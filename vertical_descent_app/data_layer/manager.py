import sqlite3
import json
import asyncio
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from .models import ForecastResponse, UnifiedDataPoint
from .interfaces import SnotelSource, OpenMeteoAdapter, NOAAAdapter, ClimatologyAdapter, WeatherSource

logger = logging.getLogger(__name__)

# Register adapter/converter to avoid Python 3.12 DeprecationWarning
def adapt_datetime(dt):
    return dt.isoformat(sep=" ")

def convert_datetime(val):
    return datetime.fromisoformat(val.decode("utf-8"))

sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter("timestamp", convert_datetime)

# Initialize SQLite with datetime parsing capabilities
# Using PARSE_DECLTYPES forces SQLite to convert TIMESTAMP strings back to datetime objects
conn = sqlite3.connect(
    'weather_cache.db',
    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    check_same_thread=False
)

def init_db():
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS swr_cache (
            query_hash TEXT PRIMARY KEY,
            payload TEXT,
            fetched_at TIMESTAMP,
            stale_at TIMESTAMP,
            expires_at TIMESTAMP
        )
    ''')
    conn.commit()

init_db()

class DataManager:
    def __init__(self):
        # Register adapters by category
        self.forecast_adapters: List[WeatherSource] = [
            OpenMeteoAdapter(),
            NOAAAdapter()
        ]
        self.telemetry_adapters: List[WeatherSource] = [
            SnotelSource()
        ]
        self.fallback_adapter = ClimatologyAdapter()

    async def get_forecast(self, lat: float, lon: float, start: datetime, end: datetime) -> ForecastResponse:
        """The sole entry point for the Dumb UI."""
        query_hash = f"{lat}_{lon}_{start.timestamp()}_{end.timestamp()}"

        # 1. Check SQLite Cache
        cursor = conn.cursor()
        cursor.execute("SELECT payload, stale_at, expires_at FROM swr_cache WHERE query_hash=?", (query_hash,))
        row = cursor.fetchone()

        now = datetime.now(timezone.utc).replace(tzinfo=None)

        if row:
            payload_json, stale_at, expires_at = row
            try:
                cached_response = ForecastResponse.model_validate_json(payload_json)

                # Check if stale or fresh
                if now < stale_at:
                    # STATE: Fresh. Return immediately.
                    cached_response.status = "CACHE_HIT_FRESH"
                    return cached_response

                elif now < expires_at:
                    # STATE: Stale. Return immediately, but trigger background fetch.
                    cached_response.status = "CACHE_HIT_STALE"
                    # Spawn a thread to handle background revalidation so it survives the asyncio.run scope
                    threading.Thread(
                        target=self._run_background_revalidate,
                        args=(query_hash, lat, lon, start, end)
                    ).start()
                    return cached_response
            except Exception as e:
                logger.error(f"Cache deserialization failed: {e}")
                # Fall through to fetch

        # STATE: Expired or Cache Miss. Fetch synchronously.
        return await self._fetch_aggregated_data(query_hash, lat, lon, start, end)

    def _run_background_revalidate(self, query_hash, lat, lon, start, end):
        """Entry point for the background thread."""
        try:
            asyncio.run(self._fetch_aggregated_data(query_hash, lat, lon, start, end))
        except Exception as e:
            logger.warning(f"Background revalidation failed: {e}")

    async def _fetch_aggregated_data(self, query_hash, lat, lon, start, end) -> ForecastResponse:
        """
        Aggregates data from multiple sources:
        1. Forecast (OpenMeteo OR NOAA)
        2. Telemetry (Snotel)
        3. Fallback (Climatology) if Forecast fails
        """
        combined_points = []
        forecast_success = False

        # 1. Fetch Forecast
        for adapter in self.forecast_adapters:
            try:
                response = await adapter.fetch_data(lat, lon, start, end)
                if response and response.points:
                    combined_points.extend(response.points)
                    forecast_success = True
                    break # Stop at first successful forecast source
            except Exception as e:
                logger.warning(f"Forecast Adapter {adapter.__class__.__name__} failed: {e}")
                continue

        # 2. Fetch Telemetry (Independent of forecast success)
        for adapter in self.telemetry_adapters:
            try:
                response = await adapter.fetch_data(lat, lon, start, end)
                if response and response.points:
                    combined_points.extend(response.points)
            except Exception as e:
                logger.warning(f"Telemetry Adapter {adapter.__class__.__name__} failed: {e}")
                continue

        # 3. Fallback if Forecast failed
        if not forecast_success:
            try:
                logger.warning("All forecast sources failed. Using Climatology fallback.")
                response = await self.fallback_adapter.fetch_data(lat, lon, start, end)
                if response and response.points:
                    combined_points.extend(response.points)
            except Exception as e:
                logger.error(f"Fallback Adapter failed: {e}")
                # If even fallback fails, and we have no telemetry, we are in trouble.
                # But if we have telemetry, we return that.

        if not combined_points:
             raise Exception("Terminal Failure: All weather sources exhausted.")

        # Construct combined response
        final_response = ForecastResponse(
            location_id=f"{lat},{lon}",
            generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
            points=combined_points,
            status="OK"
        )

        # Success: Persist to SWR cache
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        stale_at = now + timedelta(minutes=30)
        expires_at = now + timedelta(minutes=90)

        cursor = conn.cursor()
        cursor.execute(
            "REPLACE INTO swr_cache (query_hash, payload, fetched_at, stale_at, expires_at) VALUES (?,?,?,?,?)",
            (query_hash, final_response.model_dump_json(), now, stale_at, expires_at)
        )
        conn.commit()

        return final_response
