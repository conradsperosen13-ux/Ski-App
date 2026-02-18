import argparse
import logging
import sys
from datetime import datetime, timedelta
import os

# Ensure the app package is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vertical_descent_app.data_layer.database import init_db, SessionLocal
from vertical_descent_app.truth_engine.harvester import load_config, ingest_observations

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run the SNOTEL Harvester")
    parser.add_argument("--stations", nargs="+", help="Specific station triplets to fetch (overrides config)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--config", default="vertical_descent_app/config/stations.yaml", help="Path to config file")

    args = parser.parse_args()

    # Initialize DB if needed (or assume created)
    init_db()

    # Load config from default path if --config not provided or provided path
    config_path = args.config
    if not os.path.exists(config_path):
        # Fallback to absolute path relative to script if needed, or assume running from root
        # Try finding it relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        potential_path = os.path.join(project_root, config_path)
        if os.path.exists(potential_path):
            config_path = potential_path

    config = load_config(config_path)

    if args.stations:
        # Filter config stations
        stations = config.get("stations", [])
        # We need to filter based on triplet matching any in args.stations
        filtered = [s for s in stations if any(req in s["triplet"] for req in args.stations)]
        if not filtered:
             logger.warning(f"No stations found in config matching {args.stations}")
        else:
             config["stations"] = filtered

    session = SessionLocal()
    try:
        start_date = args.start
        end_date = args.end

        # Ingest
        ingest_observations(session, config, start_date, end_date)

    finally:
        session.close()

if __name__ == "__main__":
    main()
