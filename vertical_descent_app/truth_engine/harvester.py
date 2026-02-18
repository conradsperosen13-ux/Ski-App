import requests
import yaml
from datetime import datetime, timedelta
import pandas as pd
import logging
from sqlalchemy.orm import Session
from sqlalchemy import select
from vertical_descent_app.data_layer.models import Observations

logger = logging.getLogger(__name__)

AWDB_API_URL = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"

def load_config(path: str = "vertical_descent_app/config/stations.yaml") -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {path}: {e}")
        return {}

def fetch_awdb_data(triplet: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches daily SNOTEL data from AWDB REST API.
    Returns a DataFrame with Date, WTEQ, SNWD, TMIN, TMAX.
    """
    params = {
        "stationTriplets": triplet,
        "elements": "WTEQ,SNWD,TMIN,TMAX",
        "ordinal": "1",
        "duration": "DAILY",
        "beginDate": start_date,
        "endDate": end_date
    }

    try:
        response = requests.get(AWDB_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Parse response
        # Structure: [{"stationTriplet": "...", "data": [{"stationElement": {"elementCode": "SNWD", ...}, "values": [...]}, ...]}]

        records = {}

        if not data:
            return pd.DataFrame()

        # Iterate over stations (should be 1)
        for station in data:
            for item in station.get("data", []):
                element = item.get("stationElement", {}).get("elementCode")
                values = item.get("values", [])

                for v in values:
                    date_str = v.get("date")
                    val = v.get("value")

                    if date_str not in records:
                        records[date_str] = {"Date": date_str}

                    records[date_str][element] = val

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(list(records.values()))
        # Ensure numeric
        for col in ["WTEQ", "SNWD", "TMIN", "TMAX"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df["Date"] = pd.to_datetime(df["Date"])
        return df

    except Exception as e:
        logger.error(f"Error fetching data for {triplet}: {e}")
        return pd.DataFrame()

def ingest_observations(session: Session, config: dict, start_date: str = None, end_date: str = None):
    """
    Main logic to ingest observations for all configured stations.
    """
    stations = config.get("stations", [])
    defaults = config.get("defaults", {})
    lookback = defaults.get("lookback_hours", 24)

    # Calculate dates if not provided
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(hours=lookback)).strftime("%Y-%m-%d")

    logger.info(f"Starting ingestion for {len(stations)} stations from {start_date} to {end_date}")

    for station in stations:
        triplet = station.get("triplet")
        location_id = station.get("location_id")
        name = station.get("name")

        logger.info(f"Processing {name} ({triplet})...")

        df = fetch_awdb_data(triplet, start_date, end_date)

        if df.empty:
            logger.warning(f"No data returned for {name}")
            continue

        count = 0
        for _, row in df.iterrows():
            obs_time = row["Date"]

            # Check if exists (upsert logic or skip)
            # We use (observation_time_utc, location_id) unique constraint logic (though not strictly constrained in schema, we should check)
            # The schema has INDEX (observation_time_utc, location_id)

            existing = session.execute(
                select(Observations).where(
                    Observations.location_id == location_id,
                    Observations.observation_time_utc == obs_time
                )
            ).scalar_one_or_none()

            if existing:
                # Update? Or skip?
                # Usually SNOTEL data can be corrected. Let's update.
                existing.actual_swe_inches = row.get("WTEQ")
                existing.actual_snow_depth_inches = row.get("SNWD")
                existing.actual_temp_min_f = row.get("TMIN")
                existing.actual_temp_max_f = row.get("TMAX")
                # sensor_status_code logic if needed
            else:
                obs = Observations(
                    location_id=location_id,
                    observation_time_utc=obs_time,
                    actual_swe_inches=row.get("WTEQ"),
                    actual_snow_depth_inches=row.get("SNWD"),
                    actual_temp_min_f=row.get("TMIN"),
                    actual_temp_max_f=row.get("TMAX")
                )
                session.add(obs)
                count += 1

        try:
            session.commit()
            logger.info(f"Ingested {count} new records for {name}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to commit for {name}: {e}")
