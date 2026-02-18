import unittest
import asyncio
import os
import shutil
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from vertical_descent_app.data_layer.database import Base
from vertical_descent_app.data_layer.models import Observations, ForecastHistory, UnifiedDataPoint, VariableType, DataQuality, ForecastResponse
from vertical_descent_app.data_layer.manager import DataManager
from vertical_descent_app.truth_engine.harvester import ingest_observations
from vertical_descent_app.truth_engine.snapshot import capture_forecast

# Setup Test DB
TEST_DB_URL = "sqlite:///test_truth_engine.db"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)

    @classmethod
    def tearDownClass(cls):
        Base.metadata.drop_all(bind=engine)
        if os.path.exists("test_truth_engine.db"):
            os.remove("test_truth_engine.db")
        if os.path.exists("weather_cache.db"):
             # Clean up cache db created by DataManager
            try:
                os.remove("weather_cache.db")
            except:
                pass

    def setUp(self):
        self.db = TestingSessionLocal()

    def tearDown(self):
        self.db.close()

    @patch('vertical_descent_app.data_layer.interfaces.OpenMeteoAdapter.fetch_data')
    @patch('vertical_descent_app.data_layer.interfaces.SnotelSource.fetch_data')
    def test_datamanager_aggregation(self, mock_snotel, mock_openmeteo):
        # Setup mocks
        mock_openmeteo.return_value = ForecastResponse(
            location_id="loc1",
            generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
            points=[
                UnifiedDataPoint(
                    timestamp_utc=datetime.now(timezone.utc).replace(tzinfo=None),
                    variable=VariableType.PRECIP_SNOW,
                    value=1.0,
                    unit="in",
                    source="OpenMeteo",
                    quality=DataQuality.FORECAST
                )
            ],
            status="OK"
        )
        mock_snotel.return_value = ForecastResponse(
            location_id="loc1",
            generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
            points=[
                UnifiedDataPoint(
                    timestamp_utc=datetime.now(timezone.utc).replace(tzinfo=None),
                    variable=VariableType.SWE,
                    value=5.0,
                    unit="in",
                    source="Snotel",
                    quality=DataQuality.MEASURED
                )
            ],
            status="OK"
        )

        manager = DataManager()
        # Mocking async call
        start = datetime.now(timezone.utc).replace(tzinfo=None)
        end = start + timedelta(days=1)

        # Run async function
        response = asyncio.run(manager.get_forecast(40.0, -105.0, start, end))

        self.assertIsNotNone(response)
        self.assertEqual(len(response.points), 2)

        # Verify sources
        sources = [p.source for p in response.points]
        self.assertIn("OpenMeteo", sources)
        self.assertIn("Snotel", sources)

    @patch('requests.get')
    def test_harvester_ingest(self, mock_get):
        # Mock SNOTEL response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{
            "stationTriplet": "123:CO:SNTL",
            "data": [
                {
                    "stationElement": {"elementCode": "WTEQ"},
                    "values": [{"date": "2023-01-01", "value": 10.0}]
                },
                {
                    "stationElement": {"elementCode": "SNWD"},
                    "values": [{"date": "2023-01-01", "value": 20.0}]
                },
                 {
                    "stationElement": {"elementCode": "TMIN"},
                    "values": [{"date": "2023-01-01", "value": 5.0}]
                },
                 {
                    "stationElement": {"elementCode": "TMAX"},
                    "values": [{"date": "2023-01-01", "value": 25.0}]
                }
            ]
        }]
        mock_get.return_value = mock_resp

        config = {
            "stations": [{"triplet": "123:CO:SNTL", "name": "Test Station", "location_id": 1}],
            "defaults": {"lookback_hours": 24}
        }

        ingest_observations(self.db, config, "2023-01-01", "2023-01-01")

        # Verify DB
        obs = self.db.query(Observations).filter_by(location_id=1).first()
        self.assertIsNotNone(obs)
        self.assertEqual(obs.actual_swe_inches, 10.0)
        self.assertEqual(obs.actual_snow_depth_inches, 20.0)

    def test_snapshot_capture(self):
        # Create dummy forecast dataframe
        data = {
            "Date": [datetime.now(timezone.utc).replace(tzinfo=None)],
            "Model": ["TestModel"],
            "Amount": [5.0],
            "Temp_C": [-5.0],
            "SLR": [10.0],
            "Band": ["Summit"],
            "Cloud_Cover": [50],
            "Freezing_Level_m": [2000]
        }
        df = pd.DataFrame(data)

        capture_forecast(self.db, df, location_id=1, issue_time_utc=datetime.now(timezone.utc).replace(tzinfo=None))

        # Verify DB
        history = self.db.query(ForecastHistory).filter_by(location_id=1).first()
        self.assertIsNotNone(history)
        self.assertEqual(history.model_id, "TestModel")
        self.assertAlmostEqual(history.predicted_snow_depth_inches, 5.0)
        self.assertAlmostEqual(history.predicted_swe_inches, 0.5) # 5.0 / 10.0

if __name__ == '__main__':
    unittest.main()
