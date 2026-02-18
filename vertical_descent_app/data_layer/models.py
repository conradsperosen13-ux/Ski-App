from pydantic import BaseModel, ConfigDict
from datetime import datetime
from enum import Enum
from typing import Optional, List

class VariableType(Enum):
    TEMP_AIR = "temp_air"
    PRECIP_SNOW = "precip_snow"
    SNOW_DEPTH = "snow_depth" # Added
    SWE = "swe"
    WIND_SPEED = "wind_speed"
    CLOUD_COVER = "cloud_cover"

class DataQuality(Enum):
    MEASURED = "measured"   # Hard telemetry (SNOTEL)
    FORECAST = "forecast"   # Model output
    INTERPOLATED = "interpolated" # Gap-filled
    FALLBACK = "fallback"   # Climatology/Average (when API fails)

class UnifiedDataPoint(BaseModel):
    # Forbid extra fields to prevent schema pollution from upstream APIs
    model_config = ConfigDict(extra='forbid', strict=False)

    timestamp_utc: datetime
    variable: VariableType
    value: float
    unit: str  # e.g., "Imperial"
    source: str # e.g., "NOAA_GFS", "SNOTEL_335"
    quality: DataQuality

class ForecastResponse(BaseModel):
    location_id: str
    generated_at: datetime
    points: List[UnifiedDataPoint]
    status: str # "OK", "PARTIAL_FAILURE", "CACHE_HIT"
