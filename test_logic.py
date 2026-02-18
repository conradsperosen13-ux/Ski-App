import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logic
from logic import (
    PointForecastEngine,
    get_noaa_forecast,
    get_snotel_data,
    parse_snowiest_raw_text,
    calculate_forecast_metrics,
    calculate_swe_ratio,
    get_raw_forecast_data,
    RESORTS,
    DEMO_DATA
)

class TestLogic(unittest.TestCase):

    def test_calculate_swe_ratio(self):
        # Test thresholds
        self.assertEqual(calculate_swe_ratio(35), 8)   # >= 32
        self.assertEqual(calculate_swe_ratio(30), 10)  # >= 28
        self.assertEqual(calculate_swe_ratio(25), 12)  # >= 24
        self.assertEqual(calculate_swe_ratio(22), 15)  # >= 20
        self.assertEqual(calculate_swe_ratio(15), 20)  # >= 10
        self.assertEqual(calculate_swe_ratio(5), 25)   # >= 0
        self.assertEqual(calculate_swe_ratio(-5), 30)  # Default

    def test_parse_snowiest_raw_text(self):
        # Test with DEMO_DATA
        df, totals = parse_snowiest_raw_text(DEMO_DATA)
        self.assertFalse(df.empty)
        self.assertIn("Model", df.columns)
        self.assertIn("Amount", df.columns)
        self.assertIn("Date", df.columns)

        # Check totals parsing
        self.assertTrue(len(totals) > 0)

        # Test with empty input
        df_empty, totals_empty = parse_snowiest_raw_text("")
        self.assertTrue(df_empty.empty)
        self.assertEqual(totals_empty, {})

    @patch('logic.requests.get')
    def test_get_noaa_forecast(self, mock_get):
        # Mock point response
        mock_point_resp = MagicMock()
        mock_point_resp.status_code = 200
        mock_point_resp.json.return_value = {
            "properties": {
                "forecastHourly": "http://fake.url/forecast",
                "relativeLocation": {
                    "properties": {
                        "elevation": {"value": 3000}
                    }
                }
            }
        }

        # Mock forecast response
        mock_forecast_resp = MagicMock()
        mock_forecast_resp.status_code = 200
        mock_forecast_resp.json.return_value = {
            "properties": {
                "periods": [
                    {
                        "startTime": "2023-01-01T12:00:00Z",
                        "temperature": 25,
                        "relativeHumidity": {"value": 80},
                        "windSpeed": "10 mph",
                        "shortForecast": "Snow"
                    }
                ]
            }
        }

        mock_get.side_effect = [mock_point_resp, mock_forecast_resp]

        df, elev_ft = get_noaa_forecast(40.0, -105.0)

        self.assertEqual(elev_ft, 3000 * 3.28084)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Temp"], 25)
        self.assertEqual(df.iloc[0]["Wind"], 10)
        self.assertEqual(df.iloc[0]["SWE_Ratio"], 12) # 25F -> 12:1

    @patch('logic.requests.get')
    def test_get_snotel_data(self, mock_get):
        # Mock HTML response
        html_content = """
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>WTEQ (in)</th>
                    <th>SNWD (in)</th>
                    <th>TAVG (F)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>2023-01-01</td>
                    <td>10.5</td>
                    <td>40</td>
                    <td>20</td>
                </tr>
                <tr>
                    <td>2023-01-02</td>
                    <td>11.0</td>
                    <td>42</td>
                    <td>18</td>
                </tr>
            </tbody>
        </table>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html_content
        mock_get.return_value = mock_resp

        df = get_snotel_data([123])

        self.assertFalse(df.empty)
        self.assertEqual(len(df), 2)
        self.assertIn("SWE", df.columns) # Renamed from WTEQ
        self.assertIn("Depth", df.columns) # Renamed from SNWD
        self.assertIn("Temp", df.columns) # Renamed from TAVG
        self.assertEqual(df.iloc[1]["SWE_Delta"], 0.5) # 11.0 - 10.5

    def test_calculate_forecast_metrics(self):
        # Create dummy dataframe
        dates = pd.date_range(start=datetime.now(), periods=5, tz="America/Denver")
        data = {
            "Date": dates,
            "Model": ["ModelA"] * 5,
            "Amount": [1.0, 5.0, 0.5, 0.0, 2.0],
            "Band": ["Summit"] * 5
        }
        df = pd.DataFrame(data)

        metrics = calculate_forecast_metrics(df, ["ModelA"])

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["total_snowfall"], 8.5)
        self.assertEqual(metrics["best_24h_amount"], 5.0)

        # Test storm detection (>= 2.0 inches)
        # 1.0 (no), 5.0 (yes), 0.5 (no), 0.0 (no), 2.0 (yes)
        # Storms logic accumulates consecutive days >= threshold?
        # Or counts individual days? Let's check logic implementation.
        # "is_heavy" = amount >= 2.0
        # storm_group based on consecutive heavy days.

        # Check 'storms' logic:
        # Day 1: 1.0 -> No (< 2.0)
        # Day 2: 5.0 -> Start Storm, Total=5.0
        # Day 3: 0.5 -> End Storm. Append {start: Day2, total: 5.0}
        # Day 4: 0.0 -> No
        # Day 5: 2.0 -> Start Storm, Total=2.0
        # End loop -> Append {start: Day5, total: 2.0}

        self.assertEqual(len(metrics["storms"]), 2)
        self.assertEqual(metrics["storms"][0]["total"], 5.0)
        self.assertEqual(metrics["storms"][1]["total"], 2.0)

    def test_PointForecastEngine_process_physics(self):
        # Mock input data from Open-Meteo
        mock_data = {
            "elevation": 2000, # Model elevation in meters (~6560 ft)
            "hourly": {
                "time": ["2023-01-01T00:00", "2023-01-01T01:00"],
                # Model A
                "precipitation_ecmwf_ifs04": [0.1, 0.2], # inches
                "temperature_2m_ecmwf_ifs04": [-5.0, -6.0], # C
                "relative_humidity_700hPa_ecmwf_ifs04": [90, 85],
                "cloud_cover_ecmwf_ifs04": [100, 100],
                "freezing_level_height_ecmwf_ifs04": [1000, 900],
            }
        }

        elev_config = {"base": 8000, "peak": 10000} # ft
        # Base ~ 2438 m, Peak ~ 3048 m
        # Delta Z (Summit) = 3048 - 2000 = 1048 m

        df = PointForecastEngine.process_physics(mock_data, elev_config, "TestResort")

        self.assertFalse(df.empty)
        # Should have data for Summit, Mid, Base bands for ECMWF
        self.assertTrue(len(df) >= 2 * 3) # 2 timestamps * 3 bands

        # Check Summit band
        summit_df = df[df["Band"] == "Summit"]
        self.assertEqual(len(summit_df), 2)

        # Check physics adjustment
        # Temp should decrease with elevation
        # -5.0 C at 2000m.
        # At 3048m (Summit), delta_z = 1048m.
        # Lapse rate 0.65 C / 100m.
        # Adj = -5.0 - (0.65 * 10.48) = -5.0 - 6.812 = -11.812 C
        processed_temp = summit_df.iloc[0]["Temp_C"]
        self.assertAlmostEqual(processed_temp, -5.0 - (0.65 * 10.48), delta=0.1)

        # Precip should increase
        # Lift factor 0.05 / 100m
        # Multiplier = 1.0 + 0.05 * 10.48 = 1.524
        # Precip = 0.1 * 1.524 = 0.1524
        # SLR: Temp is -11.8C. Kuchera or logic matrix.
        # Logic:
        # is_rain > 1.0 (No)
        # is_wet_snow <= 1.0 & > -3.0 (No)
        # is_dgz_champagne <= -12.0 & >= -18.0 & RH >= 80 (Almost, temp is -11.8)
        # So default Kuchera: 12 + (-2 - (-11.8)) = 12 + 9.8 = 21.8
        # Snow amount = 0.1524 * 21.8 ~ 3.32 inches

        processed_amount = summit_df.iloc[0]["Amount"]
        self.assertGreater(processed_amount, 0.1)

    @patch('logic.PointForecastEngine.read_from_cache')
    @patch('logic.PointForecastEngine.fetch_api_data')
    @patch('logic.PointForecastEngine.write_to_cache')
    def test_get_raw_forecast_data(self, mock_write, mock_fetch, mock_read):
        # Case 1: Cache hit
        mock_read.return_value = {"cached": True}
        data = get_raw_forecast_data(40, -105)
        self.assertEqual(data, {"cached": True})
        mock_fetch.assert_not_called()

        # Case 2: Cache miss, fetch success
        mock_read.return_value = None
        mock_fetch.return_value = {"fetched": True}
        data = get_raw_forecast_data(40, -105)
        self.assertEqual(data, {"fetched": True})
        mock_write.assert_called()

        # Case 3: Cache miss, fetch fail
        mock_read.return_value = None
        mock_fetch.return_value = None
        data = get_raw_forecast_data(40, -105)
        self.assertIsNone(data)

if __name__ == '__main__':
    unittest.main()
