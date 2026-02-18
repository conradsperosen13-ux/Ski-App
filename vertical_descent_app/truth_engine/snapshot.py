import pandas as pd
import json
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import logging

from vertical_descent_app.data_layer.models import ForecastHistory

logger = logging.getLogger(__name__)

def capture_forecast(
    session: Session,
    forecast_df: pd.DataFrame,
    location_id: int,
    issue_time_utc: datetime,
    band_filter: str = "Summit"
):
    """
    Serializes and persists the forecast dataframe to the forecast_history table.

    Args:
        session: SQLAlchemy session.
        forecast_df: DataFrame output from the Logic Engine (must contain 'Date', 'Model', 'Amount', 'Temp_C', 'SLR', etc.).
        location_id: The ID of the location (resort/station).
        issue_time_utc: The time the forecast was generated (in UTC).
        band_filter: The elevation band to capture (default: "Summit").
    """
    if forecast_df.empty:
        logger.warning(f"No forecast data to capture for location {location_id}")
        return

    # Ensure issue_time is UTC
    if issue_time_utc.tzinfo is None:
        issue_time_utc = issue_time_utc.replace(tzinfo=timezone.utc)
    else:
        issue_time_utc = issue_time_utc.astimezone(timezone.utc)

    # Filter by band if column exists
    if "Band" in forecast_df.columns and band_filter:
        forecast_df = forecast_df[forecast_df["Band"] == band_filter]
        if forecast_df.empty:
            logger.warning(f"No forecast data for band {band_filter} at location {location_id}")
            return

    count = 0
    for _, row in forecast_df.iterrows():
        try:
            valid_time = row['Date']
            # Convert valid_time to UTC
            if valid_time.tzinfo is None:
                valid_time_utc = valid_time.replace(tzinfo=timezone.utc)
            else:
                valid_time_utc = valid_time.astimezone(timezone.utc)

            lead_time_hours = (valid_time_utc - issue_time_utc).total_seconds() / 3600.0

            # Conversions
            temp_c = row.get('Temp_C', 0)
            temp_f = (temp_c * 9/5) + 32

            snow_inches = row.get('Amount', 0)
            slr = row.get('SLR', 1)
            swe_inches = snow_inches / slr if slr else 0

            # Construct model ID (e.g. "ECMWF_Summit") or just "ECMWF"
            model_name = row.get('Model', 'Unknown')
            band = row.get('Band', 'Unknown')

            # Create payload
            payload = {
                "Band": band,
                "Cloud_Cover": row.get('Cloud_Cover'),
                "Freezing_Level_m": row.get('Freezing_Level_m'),
                "SLR": slr
            }

            history_entry = ForecastHistory(
                location_id=location_id,
                model_id=model_name, # Storing just model name. Band is in JSON.
                                     # To differentiate bands for same location, strictly speaking location_id should map to a specific point (lat/lon/elev).
                issue_time_utc=issue_time_utc,
                valid_time_utc=valid_time_utc,
                lead_time_hours=int(lead_time_hours),
                predicted_swe_inches=float(swe_inches),
                predicted_snow_depth_inches=float(snow_inches), # Assuming incremental new snow
                predicted_temp_min_f=float(temp_f), # Hourly, so min/max are same
                predicted_temp_max_f=float(temp_f),
                serialized_payload_json=payload
            )
            session.add(history_entry)
            count += 1

        except Exception as e:
            logger.error(f"Error capturing row: {e}")
            continue

    session.commit()
    logger.info(f"Captured {count} forecast records for location {location_id}")
