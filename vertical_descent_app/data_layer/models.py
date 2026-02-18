from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON, Index, BigInteger, Numeric
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class ForecastHistory(Base):
    __tablename__ = "forecast_history"

    history_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    location_id = Column(Integer, nullable=False)
    model_id = Column(String, nullable=False) # Changed to String as models are usually names/slugs
    issue_time_utc = Column(DateTime(timezone=True), nullable=False)
    valid_time_utc = Column(DateTime(timezone=True), nullable=False)
    lead_time_hours = Column(Integer, nullable=False)
    predicted_swe_inches = Column(Numeric(5, 2))
    predicted_snow_depth_inches = Column(Numeric(5, 2))
    predicted_temp_min_f = Column(Numeric(5, 2))
    predicted_temp_max_f = Column(Numeric(5, 2))
    serialized_payload_json = Column(JSON) # For future AI feature extraction

    # Index for rapid joining
    __table_args__ = (
        Index("idx_valid_time_location", "valid_time_utc", "location_id"),
    )

class Observations(Base):
    __tablename__ = "observations"

    obs_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    location_id = Column(Integer, nullable=False)
    observation_time_utc = Column(DateTime(timezone=True), nullable=False)
    actual_swe_inches = Column(Numeric(5, 2))
    actual_snow_depth_inches = Column(Numeric(5, 2))
    actual_temp_min_f = Column(Numeric(5, 2))
    actual_temp_max_f = Column(Numeric(5, 2))
    sensor_status_code = Column(Integer) # To handle missing/bad data

    # Index for rapid joining
    __table_args__ = (
        Index("idx_observation_time_location", "observation_time_utc", "location_id"),
    )

class VerificationMetrics(Base):
    __tablename__ = "verification_metrics"

    metric_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    history_id = Column(Integer, ForeignKey("forecast_history.history_id"))
    obs_id = Column(Integer, ForeignKey("observations.obs_id"))
    bias = Column(Numeric(5, 2))
    abs_error = Column(Numeric(5, 2))
    is_powder_day_hit = Column(Boolean) # For categorical scoring
    calculation_timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    forecast = relationship("ForecastHistory")
    observation = relationship("Observations")
