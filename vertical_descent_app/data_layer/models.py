from pydantic import BaseModel, ConfigDict
from datetime import datetime
from enum import Enum
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from vertical_descent_app.data_layer.database import Base

# =============================================================================
# Pydantic Models (Data Exchange)
# =============================================================================

class VariableType(str, Enum):
    """
    str, Enum ensures the enum member IS its string value.
    This means VariableType.PRECIP_SNOW == "precip_snow" is True,
    which makes DataFrame filtering and JSON round-trips through the
    SQLite cache work correctly without silent comparison failures.
    """
    TEMP_AIR    = "temp_air"
    PRECIP_SNOW = "precip_snow"
    SNOW_DEPTH  = "snow_depth"
    SWE         = "swe"
    WIND_SPEED  = "wind_speed"
    CLOUD_COVER = "cloud_cover"

class DataQuality(str, Enum):
    """
    str, Enum ensures the enum member IS its string value.
    This means DataQuality.MEASURED == "measured" is True,
    which makes DataFrame filtering and JSON round-trips through the
    SQLite cache work correctly without silent comparison failures.
    """
    MEASURED     = "measured"      # Hard telemetry (SNOTEL)
    FORECAST     = "forecast"      # Model output
    INTERPOLATED = "interpolated"  # Gap-filled
    FALLBACK     = "fallback"      # Climatology/Average (when API fails)

class UnifiedDataPoint(BaseModel):
    # Forbid extra fields to prevent schema pollution from upstream APIs
    model_config = ConfigDict(extra='forbid', strict=False)

    timestamp_utc : datetime
    variable      : VariableType
    value         : float
    unit          : str   # e.g., "in", "Fahrenheit"
    source        : str   # e.g., "NOAA_GFS", "SNOTEL_335"
    quality       : DataQuality

class ForecastResponse(BaseModel):
    location_id  : str
    generated_at : datetime
    points       : List[UnifiedDataPoint]
    status       : str  # "OK", "PARTIAL_FAILURE", "CACHE_HIT_FRESH", "CACHE_HIT_STALE"

# =============================================================================
# SQLAlchemy Models (Database Persistence)
# =============================================================================

class Observations(Base):
    __tablename__ = "observations"

    id                       = Column(Integer,  primary_key=True, index=True)
    location_id              = Column(Integer,  index=True)
    observation_time_utc     = Column(DateTime, index=True)
    actual_swe_inches        = Column(Float)
    actual_snow_depth_inches = Column(Float)
    actual_temp_min_f        = Column(Float)
    actual_temp_max_f        = Column(Float)

class ForecastHistory(Base):
    __tablename__ = "forecast_history"

    id                          = Column(Integer,  primary_key=True, index=True)
    location_id                 = Column(Integer,  index=True)
    model_id                    = Column(String,   index=True)
    issue_time_utc              = Column(DateTime, index=True)
    valid_time_utc              = Column(DateTime, index=True)
    lead_time_hours             = Column(Integer)
    predicted_swe_inches        = Column(Float)
    predicted_snow_depth_inches = Column(Float)
    predicted_temp_min_f        = Column(Float)
    predicted_temp_max_f        = Column(Float)
    serialized_payload_json     = Column(JSON)