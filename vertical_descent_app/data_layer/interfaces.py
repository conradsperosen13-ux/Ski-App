import asyncio
import aiohttp
import math
import re
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from aiobreaker import CircuitBreaker
from typing import List, Optional, Dict
import pandas as pd
from .models import UnifiedDataPoint, ForecastResponse, VariableType, DataQuality
from .config import RESORTS

logger = logging.getLogger(__name__)

# Instantiate a Circuit Breaker: Trip open after 3 failures, stay open for 5 minutes (300s)
adapter_breaker = CircuitBreaker(fail_max=3, timeout_duration=300)

class WeatherSource(ABC):
    """The strict interface contract every adapter must follow."""

    @abstractmethod
    async def fetch_data(
        self, lat: float, lon: float, start: datetime, end: datetime
    ) -> ForecastResponse:
        pass

class SnotelSource(WeatherSource):
    """
    Adapter for USDA NRCS AWDB (SNOTEL) API.
    Provides hard telemetry (ground truth) for Snow Water Equivalent (WTEQ),
    Snow Depth (SNWD), and Temperature (TOBS).
    """

    BASE_URL = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"

    def _find_nearest_resort(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Locates the nearest resort in the RESORTS config to the given coordinates.
        Simple Euclidean distance is sufficient for this scale.
        """
        nearest_key = None
        min_dist = float('inf')

        for name, data in RESORTS.items():
            # Euclidean distance approximation
            dist = math.sqrt((data['lat'] - lat)**2 + (data['lon'] - lon)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_key = name

        if nearest_key:
            return RESORTS[nearest_key]
        return None

    @adapter_breaker
    async def fetch_data(self, lat: float, lon: float, start: datetime, end: datetime) -> ForecastResponse:
        resort = self._find_nearest_resort(lat, lon)
        if not resort:
            return ForecastResponse(
                location_id=f"{lat},{lon}",
                generated_at=datetime.utcnow(),
                points=[],
                status="NO_STATION_FOUND"
            )

        points: List[UnifiedDataPoint] = []
        state = resort.get("state", "CO")

        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        for station_id in resort.get("snotel_ids", []):
            triplet = f"{station_id}:{state}:SNTL"
            elements_list = ["WTEQ", "SNWD", "TOBS"]

            params = {
                "stationTriplets": triplet,
                "elements": ",".join(elements_list),
                "ordinal": "1",
                "beginDate": start_str,
                "endDate": end_str,
                "duration": "DAILY"
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.BASE_URL, params=params) as resp:
                        if resp.status != 200:
                            continue

                        data = await resp.json()

                        for station_data in data:
                            for item in station_data.get("data", []):
                                el_info = item.get("stationElement", {})
                                el_code = el_info.get("elementCode")
                                values = item.get("values", [])

                                var_type = None
                                unit = "Imperial"

                                if el_code == "WTEQ":
                                    var_type = VariableType.SWE
                                    unit = el_info.get("storedUnitCode", "in")
                                elif el_code == "SNWD":
                                    var_type = VariableType.SNOW_DEPTH
                                    unit = el_info.get("storedUnitCode", "in")
                                elif el_code == "TOBS":
                                    var_type = VariableType.TEMP_AIR
                                    unit = el_info.get("storedUnitCode", "degF")

                                if not var_type:
                                    continue

                                for v in values:
                                    val = v.get("value")
                                    date_str = v.get("date")
                                    if val is None or not date_str:
                                        continue

                                    try:
                                        dt = datetime.strptime(date_str, "%Y-%m-%d")
                                    except ValueError:
                                        try:
                                            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
                                        except ValueError:
                                            continue

                                    dp = UnifiedDataPoint(
                                        timestamp_utc=dt,
                                        variable=var_type,
                                        value=float(val),
                                        unit=unit,
                                        source=f"SNOTEL_{station_id}",
                                        quality=DataQuality.MEASURED
                                    )
                                    points.append(dp)

            except Exception as e:
                logger.error(f"Error fetching SNOTEL {triplet}: {e}")
                continue

        return ForecastResponse(
            location_id=f"{lat},{lon}",
            generated_at=datetime.utcnow(),
            points=points,
            status="OK" if points else "PARTIAL_FAILURE"
        )

class OpenMeteoAdapter(WeatherSource):
    """Concrete implementation for the Open-Meteo API."""

    @adapter_breaker
    async def fetch_data(self, lat: float, lon: float, start: datetime, end: datetime) -> ForecastResponse:
        url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,snowfall",
            "timezone": "UTC",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d")
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        raise Exception(f"Open-Meteo API Error: {response.status}")

                    data = await response.json()
        except Exception as e:
            # Re-raise for breaker
            raise e

        points: List[UnifiedDataPoint] = []

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps_c = hourly.get("temperature_2m", [])
        snows_cm = hourly.get("snowfall", [])

        for i in range(len(times)):
            t_str = times[i]
            temp_c = temps_c[i]
            snow_cm = snows_cm[i]

            if temp_c is None or snow_cm is None:
                continue

            # Normalization
            # Celsius -> Fahrenheit
            temp_f = (temp_c * 1.8) + 32

            # cm -> inches
            snow_in = snow_cm / 2.54

            ts = datetime.fromisoformat(t_str.replace("Z", "+00:00"))

            # Temperature Point
            points.append(UnifiedDataPoint(
                timestamp_utc=ts,
                variable=VariableType.TEMP_AIR,
                value=temp_f,
                unit="Fahrenheit",
                source="OpenMeteo",
                quality=DataQuality.FORECAST
            ))

            # Snowfall Point
            points.append(UnifiedDataPoint(
                timestamp_utc=ts,
                variable=VariableType.PRECIP_SNOW,
                value=snow_in,
                unit="Imperial",
                source="OpenMeteo",
                quality=DataQuality.FORECAST
            ))

        return ForecastResponse(
            location_id=f"{lat},{lon}",
            generated_at=datetime.utcnow(),
            points=points,
            status="OK"
        )

class NOAAAdapter(WeatherSource):
    """
    Concrete implementation for NOAA/NWS API.
    Handles 2-hop spatial resolution (Point -> Grid).
    """

    # In-memory cache for grid endpoints to avoid redundant lookup
    _grid_cache = {}

    @adapter_breaker
    async def fetch_data(self, lat: float, lon: float, start: datetime, end: datetime) -> ForecastResponse:
        grid_url = await self._resolve_grid_url(lat, lon)

        headers = {
            "User-Agent": "VerticalDescent/1.0 (contact@example.com)",
            "Accept": "application/geo+json"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(grid_url, headers=headers) as response:
                    if response.status != 200:
                         raise Exception(f"NOAA Grid API Error: {response.status}")

                    data = await response.json()
        except Exception as e:
            raise e

        points: List[UnifiedDataPoint] = []
        properties = data.get("properties", {})

        # Process Temperature
        temps = properties.get("temperature", {}).get("values", [])
        for item in temps:
            await self._process_noaa_item(item, VariableType.TEMP_AIR, points, lambda x: (x * 1.8) + 32) # C to F

        # Process Snowfall
        snows = properties.get("snowfallAmount", {}).get("values", [])
        for item in snows:
             await self._process_noaa_item(item, VariableType.PRECIP_SNOW, points, lambda x: x / 25.4) # mm to in

        return ForecastResponse(
            location_id=f"{lat},{lon}",
            generated_at=datetime.utcnow(),
            points=points,
            status="OK"
        )

    async def _resolve_grid_url(self, lat: float, lon: float) -> str:
        key = f"{lat},{lon}"
        if key in self._grid_cache:
            return self._grid_cache[key]

        url = f"https://api.weather.gov/points/{lat},{lon}"
        headers = {
            "User-Agent": "VerticalDescent/1.0 (contact@example.com)",
             "Accept": "application/geo+json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"NOAA Points API Error: {response.status}")
                data = await response.json()

        grid_id = data["properties"]["gridId"]
        grid_x = data["properties"]["gridX"]
        grid_y = data["properties"]["gridY"]

        grid_url = f"https://api.weather.gov/gridpoints/{grid_id}/{grid_x},{grid_y}"
        self._grid_cache[key] = grid_url
        return grid_url

    async def _process_noaa_item(self, item, variable, points_list, converter=None):
        valid_time = item["validTime"]
        value = item["value"]

        if value is None:
            return

        if converter:
            value = converter(value)

        parts = valid_time.split("/")
        start_time_str = parts[0]
        duration_str = parts[1] if len(parts) > 1 else "PT1H"

        start_dt = pd.to_datetime(start_time_str).to_pydatetime()
        duration = pd.to_timedelta(duration_str)

        hours = int(duration.total_seconds() / 3600)
        for h in range(hours):
            ts = start_dt + timedelta(hours=h)
            points_list.append(UnifiedDataPoint(
                timestamp_utc=ts,
                variable=variable,
                value=float(value),
                unit="Imperial",
                source="NOAA_Grid",
                quality=DataQuality.FORECAST
            ))

class ClimatologyAdapter(WeatherSource):
    """
    Terminal fallback adapter.
    Returns locally stored historical averages when the internet is dead.
    """
    async def fetch_data(self, lat: float, lon: float, start: datetime, end: datetime) -> ForecastResponse:
        points: List[UnifiedDataPoint] = []

        # Generate hourly points for the requested window
        delta = end - start
        hours = int(delta.total_seconds() / 3600)

        for h in range(hours + 1):
            ts = start + timedelta(hours=h)

            # Static "safe" values based on winter averages in CO
            points.append(UnifiedDataPoint(
                timestamp_utc=ts,
                variable=VariableType.TEMP_AIR,
                value=20.0, # 20F average
                unit="Fahrenheit",
                source="Climatology_Fallback",
                quality=DataQuality.FALLBACK
            ))

            points.append(UnifiedDataPoint(
                timestamp_utc=ts,
                variable=VariableType.PRECIP_SNOW,
                value=0.05, # Trace snow per hour
                unit="Imperial",
                source="Climatology_Fallback",
                quality=DataQuality.FALLBACK
            ))

        return ForecastResponse(
            location_id=f"{lat},{lon}",
            generated_at=datetime.utcnow(),
            points=points,
            status="FALLBACK_CLIMATOLOGY"
        )
