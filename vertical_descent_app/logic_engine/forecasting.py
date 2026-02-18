import concurrent.futures
import logging
import re
import pytz
import time
from datetime import datetime, timedelta
from io import StringIO
import asyncio
import aiohttp
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import requests

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}
API_TIMEOUT = 30
STORM_THRESHOLD_IN = 2.0
SWE_RATIO_THRESHOLDS = [(32, 8), (28, 10), (24, 12), (20, 15), (10, 20), (0, 25)]
DEFAULT_COLD_RATIO = 30

# Cache configuration for NWP engine
CACHE_DIR = Path(".cache")
CACHE_TTL_SECONDS = 4 * 3600  # 4 Hours

DEMO_DATA = """Model / Date
Snow
18 Wed 	19 Thu 	20 Fri 	21 Sat 	22 Sun 	23 Mon 	24 Tue 	25 Wed 	26 Thu 	27 Fri 	28 Sat 	1 Sun 	 2 Mon 	 3 Tue 	 4 Wed 	 Total
ECMWF IFS 025 	 1.7" 	0.4" 	1.1" 	0.7" 	0" 	0" 	0.1" 	4" 	1.7" 	0.3" 	0.4" 	0" 	0.1" 	6" 	- 	 16"
GFS 3" 	0.9" 	0.9" 	0.2" 	0" 	0" 	0.2" 	0" 	0" 	0.9" 	0" 	0" 	0.7" 	0.1" 	0.2" 	7"
GEM 2" 	0.4" 	1.6" 	0.4" 	0" 	0" 	0" 	0.4" 	0.4" 	- 	 - 	 - 	 - 	 - 	 - 	 5"
JMA 0.9" 	0.7" 	0.2" 	0.5" 	0" 	0" 	0" 	4" 	2" 	8" 	- 	 - 	 - 	 - 	 - 	 16"
ICON 	0.2" 	0.1" 	0.8" 	0" 	0" 	0" 	0.1" 	- 	 - 	 - 	 - 	 - 	 - 	 - 	 - 	 1.1"
MeteoFrance 1.3" 	0.3" 	1.2" 	0.8" 	- 	 - 	 - 	 - 	 - 	 - 	 - 	 - 	 - 	 - 	 - 	 4"
Average 1.5" 	0.5" 	1" 	0.4" 	0" 	0" 	0.1" 	2" 	1" 	3" 	0.2" 	0" 	0.4" 	3" 	0.2" 	13"
Snowfall Prediction History (Estimated)
Model / Date
Snow
8 Sun 	 9 Mon 	 10 Tue 	11 Wed 	12 Thu 	13 Fri 	14 Sat 	15 Sun 	16 Mon 	17 Tue 	Total
ECMWF IFS 025 	 0" 	0" 	0.2" 	0.4" 	0.1" 	1.6" 	0.7" 	0" 	0" 	2" 	5"
GFS 0" 	0" 	0" 	0.1" 	0.2" 	0.4" 	0.2" 	0" 	0" 	0.4" 	1.2"
GEM 0" 	0" 	0.2" 	0.6" 	0.5" 	1.5" 	0.2" 	0" 	0" 	1.4" 	5"
JMA 0" 	0" 	0.8" 	0.4" 	0.1" 	0.1" 	0.3" 	0" 	0" 	1.1" 	3"
ICON 	0" 	0" 	0" 	0" 	0.3" 	0.8" 	0.2" 	0" 	0" 	1.1" 	3"
MeteoFrance 0" 	0" 	0.2" 	0.2" 	0.1" 	1.3" 	0.2" 	0" 	0" 	1.1" 	3"
Average 0" 	0" 	0.2" 	0.3" 	0.2" 	1" 	0.3" 	0" 	0" 	1.2" 	3"
"""

RESORTS = {
    "Winter Park": {
        "lat": 39.8859, "lon": -105.764,
        "snotel_ids": [1186, 335],
        "state": "CO",
        "snowiest_url": "https://www.snowiest.app/winter-park/snow-forecasts",
        "elevation": "9,000 – 12,060 ft",
        "elev_ft": {"base": 9000, "peak": 12060}
    },
    "Steamboat": {
        "lat": 40.4855, "lon": -106.8336,
        "snotel_ids": [457, 709, 825],
        "state": "CO",
        "snowiest_url": "https://www.snowiest.app/steamboat/snow-forecasts",
        "elevation": "6,900 – 10,568 ft",
        "elev_ft": {"base": 6900, "peak": 10568}
    },
    "Arapahoe Basin": {
        "lat": 39.6419, "lon": -105.8753,
        "snotel_ids": [602, 505],
        "state": "CO",
        "snowiest_url": "https://www.snowiest.app/arapahoe-basin/snow-forecasts",
        "elevation": "10,780 – 13,050 ft",
        "elev_ft": {"base": 10780, "peak": 13050}
    }
}

class PointForecastEngine:
    """
    Enterprise-Grade Numerical Weather Prediction (NWP) Processor.
    Integrated into Summit Terminal dashboard.
    """

    LAPSE_RATE_C_PER_100M = 0.65
    OROGRAPHIC_LIFT_FACTOR = 0.05

    MODEL_DISPLAY_MAP = {
        "ecmwf_ifs04": "ECMWF", "gfs_seamless": "GFS",
        "jma_seamless": "JMA", "icon_seamless": "ICON",
        "gem_global": "GEM", "meteofrance_seamless": "MeteoFrance"
    }

    @classmethod
    def get_cache_path(cls, lat: float, lon: float) -> Path:
        CACHE_DIR.mkdir(exist_ok=True)
        filename = f"forecast_{lat:.4f}_{lon:.4f}.json"
        return CACHE_DIR / filename

    @classmethod
    def read_from_cache(cls, lat: float, lon: float) -> Optional[Dict]:
        path = cls.get_cache_path(lat, lon)
        if not path.exists():
            return None

        last_modified = path.stat().st_mtime
        if (time.time() - last_modified) > CACHE_TTL_SECONDS:
            return None

        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            return None

    @classmethod
    def write_to_cache(cls, lat: float, lon: float, data: Dict):
        try:
            path = cls.get_cache_path(lat, lon)
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Cache Write Error: {e}")

    @staticmethod
    async def fetch_api_data(lat: float, lon: float) -> Optional[Dict]:
        models_list = [
            "ecmwf_ifs04", "gfs_seamless", "jma_seamless",
            "icon_seamless", "gem_global", "meteofrance_seamless"
        ]

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation,temperature_2m,relative_humidity_700hPa,cloud_cover,freezing_level_height",
            "models": ",".join(models_list),
            "precipitation_unit": "inch",
            "timezone": "America/Denver"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"API returned {resp.status}: {text[:200]}")
                        return None
                    return await resp.json()
        except asyncio.TimeoutError:
            logger.error("API request timed out")
            return None
        except Exception as e:
            logger.error(f"Async API Error: {e}")
            return None

    @classmethod
    def process_physics(cls, data: Dict, elev_config: Dict[str, int], resort_name: str) -> pd.DataFrame:
        """
        The core physics logic that converts NWP data to band-specific snow forecasts.
        """
        try:
            if not data:
                logger.error(f"[{resort_name}] No data provided")
                return pd.DataFrame()

            hourly = data.get("hourly", {})

            if not hourly:
                logger.error(f"[{resort_name}] No hourly data in response")
                return pd.DataFrame()

            if "time" not in hourly or not hourly["time"]:
                logger.error(f"[{resort_name}] No time data in response")
                return pd.DataFrame()

            times = pd.to_datetime(hourly["time"])
            if len(times) == 0:
                logger.error(f"[{resort_name}] Empty time array")
                return pd.DataFrame()

            times = times.tz_localize("America/Denver", ambiguous='NaT', nonexistent='shift_forward')

            model_elev_m = data.get("elevation", 0)
            if model_elev_m is None:
                model_elev_m = 0
                logger.warning(f"[{resort_name}] No elevation in response, using 0")

            # Define Bands
            base_m = elev_config["base"] * 0.3048
            peak_m = elev_config["peak"] * 0.3048
            mid_m = (base_m + peak_m) / 2.0
            bands = {"Base": base_m, "Mid": mid_m, "Summit": peak_m}

            df_list = []

            for model_id, model_label in cls.MODEL_DISPLAY_MAP.items():
                p_key = f"precipitation_{model_id}"
                t_key = f"temperature_2m_{model_id}"
                rh_key = f"relative_humidity_700hPa_{model_id}"
                cloud_key = f"cloud_cover_{model_id}"
                fl_key = f"freezing_level_height_{model_id}"

                # Check if all required keys exist
                if all(k in hourly for k in [p_key, t_key, rh_key, cloud_key, fl_key]):
                    try:
                        # Get data and ensure it's not None
                        precip_raw = hourly[p_key]
                        temp_raw_c = hourly[t_key]
                        rh_700 = hourly[rh_key]
                        cloud_cover = hourly[cloud_key]
                        freezing_level = hourly[fl_key]

                        # Check if any are None or empty
                        if not all([precip_raw, temp_raw_c, rh_700, cloud_cover, freezing_level]):
                            logger.debug(f"[{resort_name}] Some data empty for {model_label}")
                            continue

                        # Convert to numpy arrays, replacing None with 0
                        precip_raw = np.array([x if x is not None else 0 for x in precip_raw], dtype=float)
                        temp_raw_c = np.array([x if x is not None else 0 for x in temp_raw_c], dtype=float)
                        rh_700 = np.array([x if x is not None else 0 for x in rh_700], dtype=float)
                        cloud_cover = np.array([x if x is not None else 0 for x in cloud_cover], dtype=float)
                        freezing_level = np.array([x if x is not None else 0 for x in freezing_level], dtype=float)

                        # Ensure same length as times
                        if len(precip_raw) != len(times):
                            logger.warning(f"[{resort_name}] Length mismatch for {model_label}")
                            continue

                        for band_name, target_elev_m in bands.items():
                            delta_z_m = target_elev_m - model_elev_m
                            temp_adj_c = temp_raw_c - (cls.LAPSE_RATE_C_PER_100M * (delta_z_m / 100.0))
                            lift_multiplier = 1.0 + cls.OROGRAPHIC_LIFT_FACTOR * (np.maximum(0, delta_z_m) / 100.0)
                            precip_adj = precip_raw * lift_multiplier

                            # Ensure no negative precipitation
                            precip_adj = np.maximum(0, precip_adj)

                            # Dynamic SLR calculations
                            is_rain = temp_adj_c > 1.0
                            is_wet_snow = (temp_adj_c <= 1.0) & (temp_adj_c > -3.0)
                            is_dgz_champagne = (temp_adj_c <= -12.0) & (temp_adj_c >= -18.0) & (rh_700 >= 80.0)

                            # Kuchera SLR approximation
                            kuchera_ratio = np.clip(12.0 + (-2.0 - temp_adj_c), 8.0, 30.0)

                            slr = np.select(
                                [is_rain, is_wet_snow, is_dgz_champagne],
                                [0.0, 8.0, 20.0],
                                default=kuchera_ratio
                            )

                            # Calculate snow amount (convert to inches)
                            snow_amount = precip_adj * slr

                            df_list.append(pd.DataFrame({
                                "Date": times,
                                "Model": model_label,
                                "Band": band_name,
                                "Amount": snow_amount,
                                "Temp_C": temp_adj_c,
                                "SLR": slr,
                                "Cloud_Cover": cloud_cover,
                                "Freezing_Level_m": freezing_level,
                                "IsHistory": False,
                                "Source": "NWP"
                            }))
                    except Exception as e:
                        logger.error(f"[{resort_name}] Error processing {model_label}: {e}")
                        continue
                else:
                    missing = [k for k in [p_key, t_key, rh_key, cloud_key, fl_key] if k not in hourly]
                    logger.debug(f"[{resort_name}] Skipping {model_label} due to missing keys: {missing}")

            if not df_list:
                logger.warning(f"[{resort_name}] No valid model data processed")
                return pd.DataFrame()

            result = pd.concat(df_list, ignore_index=True)
            logger.info(f"[{resort_name}] Processed {len(result)} rows from {len(df_list)} models")
            return result

        except Exception as e:
            logger.error(f"Physics Processing Error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    @classmethod
    async def get_forecast_async(cls, lat: float, lon: float, elev_config: Dict[str, int], resort_name: str = "Unknown") -> pd.DataFrame:
        """
        Async entry point with caching.
        """
        cache_data = cls.read_from_cache(lat, lon)
        if cache_data:
            logger.info(f"[{resort_name}] Cache Hit (Valid < 4hr)")
            data = cache_data
        else:
            logger.info(f"[{resort_name}] Cache Miss. Fetching API...")
            data = await cls.fetch_api_data(lat, lon)
            if data:
                cls.write_to_cache(lat, lon, data)

        if not data:
            logger.warning(f"[{resort_name}] No data returned from API/Cache.")
            return pd.DataFrame()

        return await asyncio.to_thread(cls.process_physics, data, elev_config, resort_name)

def run_async_forecast(lat: float, lon: float, elev_config: Dict[str, int], resort_name: str) -> pd.DataFrame:
    """
    Synchronous wrapper for running async forecast in Streamlit.
    """
    try:
        return asyncio.run(
            PointForecastEngine.get_forecast_async(lat, lon, elev_config, resort_name)
        )
    except Exception as e:
        logger.error(f"Error running async forecast: {e}")
        return pd.DataFrame()

def get_raw_forecast_data(lat: float, lon: float) -> Optional[Dict]:
    """
    Retrieves raw forecast data from cache or API (synchronously).
    """
    # Try cache first
    data = PointForecastEngine.read_from_cache(lat, lon)
    if data:
        return data

    # Fetch from API
    try:
        # Use asyncio.run for the async fetch
        data = asyncio.run(PointForecastEngine.fetch_api_data(lat, lon))
        if data:
            PointForecastEngine.write_to_cache(lat, lon, data)
        return data
    except Exception as e:
        logger.error(f"Error fetching raw data: {e}")
        return None

def calculate_swe_ratio(temp_f):
    for threshold, ratio in SWE_RATIO_THRESHOLDS:
        if temp_f >= threshold:
            return ratio
    return DEFAULT_COLD_RATIO

_NOAA_CACHE = {"df": None, "elev": 0, "timestamp": None}

def _cache_noaa(df, elev):
    _NOAA_CACHE["df"] = df
    _NOAA_CACHE["elev"] = elev
    _NOAA_CACHE["timestamp"] = datetime.now()

def _get_cached_noaa():
    if _NOAA_CACHE["df"] is not None:
        age = (datetime.now() - _NOAA_CACHE["timestamp"]).total_seconds()
        if age < 21600:
            return _NOAA_CACHE["df"], _NOAA_CACHE["elev"]
    return None

def get_noaa_forecast(lat, lon, retries=3):
    for attempt in range(retries):
        try:
            point_resp = requests.get(
                f"https://api.weather.gov/points/{lat},{lon}",
                headers=API_HEADERS, timeout=API_TIMEOUT
            )
            point_resp.raise_for_status()
            prop = point_resp.json().get("properties", {})
            forecast_url = prop.get("forecastHourly")
            if not forecast_url:
                cached = _get_cached_noaa()
                return cached if cached else (pd.DataFrame(), 0)

            forecast_resp = requests.get(forecast_url, headers=API_HEADERS, timeout=API_TIMEOUT)
            forecast_resp.raise_for_status()
            periods = forecast_resp.json()["properties"]["periods"]

            elev_m = prop.get("relativeLocation", {}).get("properties", {}).get("elevation", {}).get("value", 0)
            elevation_ft = elev_m * 3.28084 if elev_m else 0

            data = []
            for p in periods:
                temp_f = p["temperature"]
                wind_str = str(p.get("windSpeed", "0"))
                wind_speed = 0
                if wind_str and wind_str[0].isdigit():
                    try:
                        wind_speed = int(wind_str.split()[0])
                    except:
                        wind_speed = 0
                data.append({
                    "Time": pd.to_datetime(p["startTime"]).tz_convert("UTC").tz_localize(None),
                    "Temp": temp_f,
                    "Humidity": p["relativeHumidity"]["value"],
                    "Wind": wind_speed,
                    "SWE_Ratio": calculate_swe_ratio(temp_f),
                    "Summary": p["shortForecast"]
                })
            df = pd.DataFrame(data)
            _cache_noaa(df, elevation_ft)
            return df, elevation_ft

        except Exception as e:
            logger.error(f"NOAA attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                cached = _get_cached_noaa()
                if cached:
                    # Removed st.warning as this is logic only
                    return cached
                return pd.DataFrame(), 0

def _fetch_single_snotel(sid, state="CO", retries=3):
    primary_url = (
        "https://wcc.sc.egov.usda.gov/reportGenerator/view/"
        "customSingleStationReport/daily/{}:{}:SNTL|id=%22%22|"
        "name/-29,0/WTEQ::value,WTEQ::median_1991,WTEQ::pctOfMedian_1991,"
        "SNWD::value,PREC::value,PREC::median_1991,PREC::pctOfMedian_1991,"
        "TMAX::value,TMIN::value,TAVG::value?fitToScreen=false"
    ).format(sid, state)
    fallback_url = (
        "https://wcc.sc.egov.usda.gov/reportGenerator/view/"
        "customSingleStationReport/daily/{}:{}:SNTL|id=%22%22|"
        "name/-30,0/WTEQ::value,SNWD::value,TAVG::value"
    ).format(sid, state)

    for attempt in range(retries):
        for url in [primary_url, fallback_url]:
            try:
                resp = requests.get(url, timeout=API_TIMEOUT, headers=API_HEADERS)
                resp.raise_for_status()
                tables = pd.read_html(StringIO(resp.text))

                df = None
                for tbl in tables:
                    cols = [str(c).upper() for c in tbl.columns]
                    if any('DATE' in c for c in cols) and any('WTEQ' in c or 'WATER' in c for c in cols):
                        df = tbl
                        break
                if df is None:
                    df = next((t for t in tables if not t.empty), pd.DataFrame())
                if df.empty:
                    continue

                def find_column(df, possible_names, exclude_median_pct=True):
                    for col in df.columns:
                        col_upper = str(col).upper()
                        for name in possible_names:
                            if name.upper() in col_upper:
                                if exclude_median_pct and ('MEDIAN' in col_upper or 'PCT' in col_upper):
                                    continue
                                return col
                    return None

                date_col = find_column(df, ['Date'])
                if date_col is None:
                    continue
                df.rename(columns={date_col: 'Date'}, inplace=True)

                swe_col = find_column(df, ['WTEQ', 'Snow Water Equivalent', 'SWE'], exclude_median_pct=True)
                if swe_col is None:
                    continue
                df.rename(columns={swe_col: 'SWE'}, inplace=True)

                depth_col = find_column(df, ['SNWD', 'Snow Depth'], exclude_median_pct=False)
                if depth_col:
                    df.rename(columns={depth_col: 'Depth'}, inplace=True)

                temp_col = find_column(df, ['TAVG', 'Temperature Average'], exclude_median_pct=False)
                if temp_col:
                    df.rename(columns={temp_col: 'Temp'}, inplace=True)

                if 'Date' not in df.columns or 'SWE' not in df.columns:
                    continue

                df["Date"] = pd.to_datetime(df["Date"])
                df["SWE_Delta"] = df["SWE"].diff().clip(lower=0)
                df["SiteID"] = str(sid)
                return df

            except Exception as e:
                logger.warning(f"SNOTEL {sid} failed: {e}")
                continue

        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return pd.DataFrame()

def get_snotel_data(site_ids, state="CO"):
    combined = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        # Use partial or lambda to pass state
        futures = [ex.submit(_fetch_single_snotel, sid, state) for sid in site_ids]
        results = [f.result() for f in futures]

    for df in results:
        if not df.empty:
            combined.append(df)
    if not combined:
        return pd.DataFrame()
    res = pd.concat(combined)
    res["Date"] = res["Date"].dt.strftime("%Y-%m-%d")
    return res

def _calculate_date(day_val, is_history):
    now = datetime.now()
    try:
        day_num = int(day_val)
        month, year = now.month, now.year
        if not is_history and day_num < now.day:
            month = (month % 12) + 1
            if month == 1:
                year += 1
        elif is_history and day_num > now.day + 5:
            month = month - 1 if month > 1 else 12
            if month == 12:
                year -= 1
        return f"{year}-{month:02d}-{day_num:02d}"
    except:
        return None

def _parse_model_line(line, headers, is_history, model_totals):
    parts = line.split('\t')
    if len(parts) < 2:
        return []
    model_name = parts[0].strip()
    keywords = ["ECMWF IFS 025", "ECMWF", "GFS", "GEM", "JMA", "ICON", "MeteoFrance", "Average"]
    if not any(kw in model_name for kw in keywords):
        return []

    values = parts[1:]
    numbers = []
    for v in values:
        v = v.strip().replace('"', '')
        if v in ('-', '.', ''):
            numbers.append(None)
        else:
            try:
                numbers.append(float(v))
            except:
                numbers.append(None)

    observations = []
    daily_sum = 0.0
    for i, day_val in enumerate(headers):
        if i >= len(numbers):
            break
        amount = numbers[i] if numbers[i] is not None else 0.0
        daily_sum += amount
        date_str = _calculate_date(day_val, is_history)
        if date_str:
            observations.append({
                "Date": pd.to_datetime(date_str),
                "Amount": amount,
                "Model": model_name,
                "IsHistory": is_history,
                "Source": "Snowiest"
            })

    if len(numbers) > len(headers):
        total_val = numbers[len(headers)]
        if total_val is not None:
            key = f"{model_name} ({'H' if is_history else 'F'})"
            model_totals[key] = {"declared": total_val, "calculated": daily_sum}

    return observations

def parse_snowiest_raw_text(text):
    if not text:
        return pd.DataFrame(), {}
    obs = []
    totals = {}
    sections = re.split(r"(Snowfall Prediction History)", text)

    f_lines = sections[0].split("\n")
    f_headers = []
    for line in f_lines:
        m = re.findall(r"(\d+)\s+(Sun|Mon|Tue|Wed|Thu|Fri|Sat)", line)
        if m:
            f_headers = [x[0] for x in m]
            break
    if f_headers:
        for line in f_lines:
            obs.extend(_parse_model_line(line, f_headers, False, totals))

    if len(sections) > 1:
        h_lines = sections[1].split("\n")
        h_headers = []
        for line in h_lines:
            m = re.findall(r"(\d+)\s+(Sun|Mon|Tue|Wed|Thu|Fri|Sat)", line)
            if m:
                h_headers = [x[0] for x in m]
                break
        if h_headers:
            for line in h_lines:
                obs.extend(_parse_model_line(line, h_headers, True, totals))

    if not obs:
        return pd.DataFrame(), {}
    df = pd.DataFrame(obs).dropna()
    return df, totals

def calculate_forecast_metrics(df_models, selected_models, current_depth=0, band_filter="Summit"):
    if df_models.empty or not selected_models:
        return None

    # Make a copy to avoid altering the original DataFrame
    df = df_models.copy()

    # --- Ensure Date is datetime and timezone-aware ---
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])
    if df["Date"].dt.tz is None:
        df["Date"] = df["Date"].dt.tz_localize("America/Denver", ambiguous='NaT', nonexistent='shift_forward')
    # -------------------------------------------------

    filtered = df[
        (df["Model"].isin(selected_models)) &
        (df["Band"] == band_filter)
    ] if "Band" in df.columns else df[df["Model"].isin(selected_models)]

    # Use the timezone from the data
    tz = filtered["Date"].dt.tz if not filtered.empty else pytz.timezone('America/Denver')
    now = pd.Timestamp.now(tz=tz).normalize()  # midnight today
    future = filtered[filtered["Date"] >= now]

    if future.empty:
        return None

    # Group by date and calculate statistics
    daily_stats = future.groupby("Date")["Amount"].agg(["mean", "min", "max", "std"]).reset_index()
    daily_stats = daily_stats.sort_values("Date").reset_index(drop=True)
    daily_stats["cumulative"] = daily_stats["mean"].cumsum()
    daily_stats["total_depth"] = current_depth + daily_stats["cumulative"]
    daily_stats["spread"] = daily_stats["max"] - daily_stats["min"]
    daily_stats["mean_48h"] = daily_stats["mean"].rolling(window=2, min_periods=1).sum()

    best_24h_idx  = daily_stats["mean"].idxmax()
    best_24h_val  = daily_stats.loc[best_24h_idx, "mean"]
    best_24h_date = daily_stats.loc[best_24h_idx, "Date"]

    best_48h_idx  = daily_stats["mean_48h"].idxmax()
    best_48h_val  = daily_stats.loc[best_48h_idx, "mean_48h"]
    best_48h_date = daily_stats.loc[best_48h_idx, "Date"]

    daily_stats["is_heavy"] = daily_stats["mean"] >= STORM_THRESHOLD_IN
    daily_stats["storm_group"] = (daily_stats["is_heavy"] != daily_stats["is_heavy"].shift()).cumsum()
    storm_groups = daily_stats[daily_stats["is_heavy"]].groupby("storm_group")

    major_storms = []
    for _, group in storm_groups:
        if len(group) >= 3:
            major_storms.append({
                "start": group["Date"].iloc[0],
                "end":   group["Date"].iloc[-1],
                "total": group["mean"].sum(),
                "days":  len(group)
            })
    major_storms.sort(key=lambda x: x["total"], reverse=True)

    storms = []
    in_storm, storm_start, storm_total = False, None, 0.0
    for _, row in daily_stats.iterrows():
        if row["mean"] >= STORM_THRESHOLD_IN and not in_storm:
            in_storm, storm_start, storm_total = True, row["Date"], row["mean"]
        elif row["mean"] >= STORM_THRESHOLD_IN and in_storm:
            storm_total += row["mean"]
        elif row["mean"] < STORM_THRESHOLD_IN and in_storm:
            storms.append({"start": storm_start, "total": storm_total})
            in_storm, storm_total = False, 0.0
    if in_storm:
        storms.append({"start": storm_start, "total": storm_total})

    return {
        "best_24h_date":   best_24h_date,
        "best_24h_amount": best_24h_val,
        "best_48h_date":   best_48h_date,
        "best_48h_amount": best_48h_val,
        "major_storms":    major_storms,
        "deepest_total":   daily_stats["total_depth"].max(),
        "total_snowfall":  daily_stats["cumulative"].iloc[-1],
        "avg_spread":      daily_stats["spread"].mean(),
        "storms":          storms,
        "daily_stats":     daily_stats,
    }
