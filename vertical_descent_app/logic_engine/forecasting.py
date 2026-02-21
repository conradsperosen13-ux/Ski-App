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

RESORTS = {
    "Winter Park": {
        "lat": 39.8859, "lon": -105.764,
        "snotel_ids": [1186, 335],
        "state": "CO",
        "timezone": "America/Denver",
        "snowiest_url": "https://www.snowiest.app/winter-park/snow-forecasts",
        "elevation": "9,000 – 12,060 ft",
        "elev_ft": {"base": 9000, "peak": 12060}
    },
    "Steamboat": {
        "lat": 40.4855, "lon": -106.8336,
        "snotel_ids": [457, 709, 825],
        "state": "CO",
        "timezone": "America/Denver",
        "snowiest_url": "https://www.snowiest.app/steamboat/snow-forecasts",
        "elevation": "6,900 – 10,568 ft",
        "elev_ft": {"base": 6900, "peak": 10568}
    },
    "Arapahoe Basin": {
        "lat": 39.6419, "lon": -105.8753,
        "snotel_ids": [602, 505],
        "state": "CO",
        "timezone": "America/Denver",
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
            "timezone": "UTC"
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
    def process_physics(cls, data: Dict, elev_config: Dict[str, int], resort_name: str, tz_str: str = "UTC") -> pd.DataFrame:
        """
        Enterprise-Grade Physics Engine.
        Features true fuzzy key matching and resilient array handling.
        """
        try:
            if not data:
                logger.error(f"[{resort_name}] No data provided")
                return pd.DataFrame()

            hourly = data.get("hourly", {})
            if not hourly or "time" not in hourly:
                logger.error(f"[{resort_name}] Incomplete hourly data")
                return pd.DataFrame()

            # 1. Safely parse and convert timezones
            times = pd.to_datetime(hourly["time"])
            if len(times) == 0:
                return pd.DataFrame()
            times = times.tz_localize("UTC").tz_convert(tz_str)

            model_elev_m = data.get("elevation", 0) or 0
            
            # 2. Elevation Band Setup
            base_m = elev_config["base"] * 0.3048
            peak_m = elev_config["peak"] * 0.3048
            mid_m = (base_m + peak_m) / 2.0
            bands = {"Base": base_m, "Mid": mid_m, "Summit": peak_m}

            df_list = []

            for model_id, model_label in cls.MODEL_DISPLAY_MAP.items():
                # TRUE FUZZY MATCHING: Extract 'gfs' from 'gfs_seamless'
                short_id = model_id.split('_')[0] 
                
                # Find keys that contain the variable type AND the short_id
                p_key = next((k for k in hourly.keys() if k.startswith("precipitation_") and short_id in k), None)
                t_key = next((k for k in hourly.keys() if k.startswith("temperature_2m_") and short_id in k), None)
                rh_key = next((k for k in hourly.keys() if k.startswith("relative_humidity_700hPa_") and short_id in k), None)
                cloud_key = next((k for k in hourly.keys() if k.startswith("cloud_cover_") and short_id in k), None)
                fl_key = next((k for k in hourly.keys() if k.startswith("freezing_level_height_") and short_id in k), None)

                # Only proceed if the critical thermodynamic keys exist
                if p_key and t_key and rh_key:
                    try:
                        # FAST ARRAY PROCESSING: Replace Nones with NaNs, then convert to 0
                        precip_raw = np.nan_to_num(np.array(hourly[p_key], dtype=float), nan=0.0)
                        temp_raw_c = np.nan_to_num(np.array(hourly[t_key], dtype=float), nan=0.0)
                        rh_700 = np.nan_to_num(np.array(hourly[rh_key], dtype=float), nan=0.0)
                        
                        # Resilient fallback for non-critical keys
                        cloud_cover = np.nan_to_num(np.array(hourly.get(cloud_key, [0]*len(times)), dtype=float), nan=0.0)
                        freezing_level = np.nan_to_num(np.array(hourly.get(fl_key, [0]*len(times)), dtype=float), nan=0.0)

                        for band_name, target_elev_m in bands.items():
                            delta_z_m = target_elev_m - model_elev_m
                            
                            # Physics: Thermodynamic Downscaling
                            temp_adj_c = temp_raw_c - (cls.LAPSE_RATE_C_PER_100M * (delta_z_m / 100.0))
                            
                            # Physics: Orographic Lift
                            lift_multiplier = 1.0 + cls.OROGRAPHIC_LIFT_FACTOR * (np.maximum(0, delta_z_m) / 100.0)
                            precip_adj = np.maximum(0, precip_raw * lift_multiplier)

                            # Physics: Dynamic SLR
                            is_rain = temp_adj_c > 1.0
                            is_wet_snow = (temp_adj_c <= 1.0) & (temp_adj_c > -3.0)
                            is_dgz_champagne = (temp_adj_c <= -12.0) & (temp_adj_c >= -18.0) & (rh_700 >= 80.0)
                            
                            kuchera_ratio = np.clip(12.0 + (-2.0 - temp_adj_c), 8.0, 30.0)

                            slr = np.select(
                                [is_rain, is_wet_snow, is_dgz_champagne],
                                [0.0, 8.0, 20.0],
                                default=kuchera_ratio
                            )

                            df_list.append(pd.DataFrame({
                                "Date": times,
                                "Model": model_label,
                                "Band": band_name,
                                "Amount": precip_adj * slr,
                                "Temp_C": temp_adj_c,
                                "SLR": slr,
                                "Cloud_Cover": cloud_cover,
                                "Freezing_Level_m": freezing_level,
                                "IsHistory": False,
                                "Source": "NWP"
                            }))
                    except Exception as e:
                        logger.error(f"[{resort_name}] Model processing failed for {model_label}: {e}")

            if not df_list:
                return pd.DataFrame()

            result = pd.concat(df_list, ignore_index=True)
            logger.info(f"[{resort_name}] Successfully processed {len(result)} physics-adjusted rows.")
            return result

        except Exception as e:
            logger.error(f"Critical Physics Error: {e}")
            return pd.DataFrame()

    @classmethod
    async def get_forecast_async(cls, lat: float, lon: float, elev_config: Dict[str, int], resort_name: str = "Unknown", tz_str: str = "UTC") -> pd.DataFrame:
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

        return await asyncio.to_thread(cls.process_physics, data, elev_config, resort_name, tz_str)

def run_async_forecast(lat: float, lon: float, elev_config: Dict[str, int], resort_name: str, tz_str: str = "UTC") -> pd.DataFrame:
    try:
        return asyncio.run(
            PointForecastEngine.get_forecast_async(lat, lon, elev_config, resort_name, tz_str)
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

def calculate_forecast_metrics(df_models, selected_models, current_depth=0, band_filter="Summit", tz_str="UTC"):
    if df_models.empty or not selected_models:
        return None

    df = df_models.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])
    
    if df["Date"].dt.tz is None:
        df["Date"] = df["Date"].dt.tz_localize(tz_str, ambiguous='NaT', nonexistent='shift_forward')
    else:
        df["Date"] = df["Date"].dt.tz_convert(tz_str)

    filtered = df[
        (df["Model"].isin(selected_models)) &
        (df["Band"] == band_filter)
    ] if "Band" in df.columns else df[df["Model"].isin(selected_models)]

    tz = pytz.timezone(tz_str)
    now = pd.Timestamp.now(tz=tz).normalize()
    future = filtered[filtered["Date"] >= now].copy()

    if future.empty:
        return None

    # 1. Calculate Consensus Time-Series
    # Average the models at each specific timestamp
    ts_consensus = future.groupby("Date")["Amount"].mean().reset_index()
    ts_consensus = ts_consensus.sort_values("Date").reset_index(drop=True)
    
    total_snowfall = ts_consensus["Amount"].sum()
    
    # 2. Daily Aggregation for Charts
    future["Day"] = future["Date"].dt.normalize()
    # Sum each model's daily output, THEN average across models
    daily_model_sums = future.groupby(["Day", "Model"])["Amount"].sum().reset_index()
    daily_stats = daily_model_sums.groupby("Day")["Amount"].agg(
        mean="mean", min="min", max="max", std="std"
    ).reset_index()
    daily_stats.rename(columns={"Day": "Date"}, inplace=True)
    
    daily_stats["cumulative"] = daily_stats["mean"].cumsum()
    daily_stats["total_depth"] = current_depth + daily_stats["cumulative"]
    daily_stats["spread"] = daily_stats["max"] - daily_stats["min"]

    # 3. The Adaptive Hero Logic 
    hero_context = {
        "condition": "DRY",
        "best_day": None,
        "peak_amount": 0,
        "timing_note": ""
    }

    if total_snowfall < 2.0:
        # DRY SPELL LOGIC: Determine base conditions
        hero_context["condition"] = "DRY"
        # Find the warmest day in the next 5 days for spring slush check
        if "Temp_C" in future.columns:
            warmest_day = future.groupby("Day")["Temp_C"].max().idxmax()
            max_temp = future.groupby("Day")["Temp_C"].max().max()
            
            if max_temp > 2.0 and current_depth > 20:
                hero_context["timing_note"] = "Spring Conditions Formulating"
                hero_context["best_day"] = warmest_day
            else:
                hero_context["timing_note"] = "Firm / Hardpack Conditions"
                hero_context["best_day"] = now
    else:
        # SNOW EVENT LOGIC: Find best 24h rolling window
        hero_context["condition"] = "SNOW"
        
        # Determine if data is hourly (NWP) or daily (Snowiest)
        is_hourly = (ts_consensus["Date"].dt.hour > 0).any()
        
        if is_hourly:
            ts_consensus.set_index("Date", inplace=True)
            # Fix Future Warning: Use '24h' instead of '24H'
            rolling_24h = ts_consensus["Amount"].rolling("24h").sum()
            
            # idxmax() returns the timestamp itself because Date is the index
            peak_end_time = rolling_24h.idxmax()
            peak_amount = rolling_24h.loc[peak_end_time]
            
            # Reset index AFTER we are done using the time-based index
            ts_consensus.reset_index(inplace=True)
            
            # The Operational Noon Cutoff Rule
            if peak_end_time.hour <= 12:
                hero_context["best_day"] = peak_end_time.normalize()
                hero_context["timing_note"] = f"Accumulation tapers by {peak_end_time.strftime('%H:00')}"
            else:
                hero_context["best_day"] = (peak_end_time + pd.Timedelta(days=1)).normalize()
                hero_context["timing_note"] = f"Drops {peak_amount:.1f}\" by {peak_end_time.strftime('%a %H:00')}"
                
            hero_context["peak_amount"] = peak_amount
        else:
            # Fallback for daily Snowiest data
            best_idx = daily_stats["mean"].idxmax()
            hero_context["best_day"] = daily_stats.loc[best_idx, "Date"] + pd.Timedelta(days=1)
            hero_context["peak_amount"] = daily_stats.loc[best_idx, "mean"]
            hero_context["timing_note"] = "Based on daily totals"

    return {
        "total_snowfall":  total_snowfall,
        "avg_spread":      daily_stats["spread"].mean() if not daily_stats["spread"].isna().all() else 0,
        "daily_stats":     daily_stats,
        "hero_context":    hero_context
    }
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def build_forecast_figure(stats, noaa_df, show_spaghetti, show_ribbon, df_models, selected_models, noaa_cutoff_dt, show_extended, w_end, band_filter="Summit", is_dark=True):
    fig = go.Figure()
    if stats.empty: return fig

    # 1. DAILY BARS (Snow per 24h window)
    fig.add_trace(go.Bar(
        x=stats["Date"],
        y=stats["mean"],
        name="Daily Mean",
        marker_color='rgba(45,212,191,0.3)' if is_dark else 'rgba(45,212,191,0.5)',
        hovertemplate="Daily Accum: %{y:.1f}\"<extra></extra>"
    ))

    # 2. UNCERTAINTY RIBBON
    if show_ribbon:
        fig.add_trace(go.Scatter(
            x=stats["Date"].tolist() + stats["Date"].tolist()[::-1],
            y=stats["max"].tolist() + stats["min"].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(56,189,248,0.07)' if is_dark else 'rgba(56,189,248,0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=False
        ))

    # 3. CUMULATIVE CONSENSUS LINE
    fig.add_trace(go.Scatter(
        x=stats["Date"],
        y=stats["cumulative"],
        mode='lines+markers',
        line=dict(color='#38bdf8', width=3),
        marker=dict(size=4),
        name="Total Forecast"
    ))

    # 4. SPAGHETTI (Individual Ensemble Members)
    if show_spaghetti and not df_models.empty:
        for model in selected_models:
            m_data = df_models[df_models["Model"] == model].copy()
            if "Band" in m_data.columns:
                m_data = m_data[m_data["Band"] == band_filter]
            
            # Aggregate to daily cumulative for spaghetti lines
            m_daily = m_data.groupby(m_data["Date"].dt.normalize())["Amount"].sum().cumsum().reset_index()
            fig.add_trace(go.Scatter(
                x=m_daily["Date"], y=m_daily["Amount"],
                mode='lines', line=dict(width=1),
                opacity=0.2, showlegend=False, hoverinfo="skip"
            ))

    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        legend=dict(orientation="h", y=1.1, x=0),
        xaxis=dict(range=[stats["Date"].min(), w_end] if w_end else None)
    )
    return fig