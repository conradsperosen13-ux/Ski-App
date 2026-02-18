import sqlite3
import json
import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, List
from .models import ForecastResponse, UnifiedDataPoint
from .interfaces import SnotelSource, OpenMeteoAdapter, NOAAAdapter, ClimatologyAdapter, WeatherSource

logger = logging.getLogger(__name__)

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
        # Register the fallback chain hierarchy
        # Primary: Open-Meteo (Fast, good coverage)
        # Secondary: NOAA (Official)
        # Tertiary: SNOTEL (Ground truth, but sparse)
        # Using SNOTEL as primary for now to test Phase 1/2 transition?
        # Blueprint says: "Primary: Open-Meteo Ensemble... Secondary: NOAA Point Forecast... Tertiary: AWDB SNOTEL Telemetry."
        self.adapters: List[WeatherSource] = [
            OpenMeteoAdapter(),
            NOAAAdapter(),
            SnotelSource(),
            ClimatologyAdapter()
        ]

    async def get_forecast(self, lat: float, lon: float, start: datetime, end: datetime) -> ForecastResponse:
        """The sole entry point for the Dumb UI."""
        query_hash = f"{lat}_{lon}_{start.timestamp()}_{end.timestamp()}"

        # 1. Check SQLite Cache
        cursor = conn.cursor()
        cursor.execute("SELECT payload, stale_at, expires_at FROM swr_cache WHERE query_hash=?", (query_hash,))
        row = cursor.fetchone()

        now = datetime.utcnow()

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
        return await self._execute_fallback_chain(query_hash, lat, lon, start, end)

    def _run_background_revalidate(self, query_hash, lat, lon, start, end):
        """Entry point for the background thread."""
        try:
            asyncio.run(self._execute_fallback_chain(query_hash, lat, lon, start, end))
        except Exception as e:
            logger.warning(f"Background revalidation failed: {e}")

    async def _execute_fallback_chain(self, query_hash, lat, lon, start, end) -> ForecastResponse:
        """Iterates through adapters until one succeeds, enforcing graceful degradation."""
        last_exception = None

        for adapter in self.adapters:
            try:
                # If circuit breaker is open, this raises instantly, moving to the next adapter
                response = await adapter.fetch_data(lat, lon, start, end)

                if not response or not response.points:
                    # If adapter returned empty response (e.g. Snotel no station found), continue
                    continue

                # Success: Persist to SWR cache
                now = datetime.utcnow()
                stale_at = now + timedelta(minutes=30)
                expires_at = now + timedelta(minutes=90)

                cursor = conn.cursor()
                cursor.execute(
                    "REPLACE INTO swr_cache (query_hash, payload, fetched_at, stale_at, expires_at) VALUES (?,?,?,?,?)",
                    (query_hash, response.model_dump_json(), now, stale_at, expires_at)
                )
                conn.commit()

                return response

            except Exception as e:
                logger.warning(f"Adapter {adapter.__class__.__name__} failed: {e}")
                last_exception = e
                continue

        # If we reach here, all adapters failed
        raise Exception(f"Terminal Failure: All weather sources exhausted. Last error: {last_exception}")
