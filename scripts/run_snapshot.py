import logging
import sys
import os
import asyncio
from datetime import datetime

# Ensure the app package is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vertical_descent_app.data_layer.database import init_db, SessionLocal
from vertical_descent_app.logic_engine.forecasting import run_async_forecast, RESORTS
from vertical_descent_app.truth_engine.snapshot import capture_forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Map Resort Name to Location ID (Must match stations.yaml)
LOCATION_MAP = {
    "Winter Park": 1,
    "Steamboat": 2,
    "Arapahoe Basin": 3
}

def main():
    logger.info("Starting Forecast Snapshot Run...")

    init_db()

    session = SessionLocal()
    try:
        issue_time = datetime.now()

        for name, config in RESORTS.items():
            location_id = LOCATION_MAP.get(name)
            if not location_id:
                logger.warning(f"No location ID mapping for {name}, skipping snapshot.")
                continue

            logger.info(f"Running forecast for {name}...")

            # Run forecast logic
            lat = config["lat"]
            lon = config["lon"]
            elev_config = config["elev_ft"]

            # Logic Engine call
            df = run_async_forecast(lat, lon, elev_config, name)

            if df.empty:
                logger.warning(f"No forecast data generated for {name}")
                continue

            # Capture Snapshot
            # Default to capturing 'Summit' band as that's usually where SNOTEL is relevant or snow matters most
            capture_forecast(session, df, location_id, issue_time, band_filter="Summit")

    except Exception as e:
        logger.error(f"Snapshot run failed: {e}")
    finally:
        session.close()
        logger.info("Snapshot run complete.")

if __name__ == "__main__":
    main()
