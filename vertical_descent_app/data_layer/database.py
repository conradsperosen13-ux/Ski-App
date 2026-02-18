from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite database URL
DATABASE_URL = "sqlite:///truth_engine.db"

# Create the engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}, # Needed for SQLite with multiple threads (e.g., Streamlit)
    echo=False # Set to True for SQL logging
)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

def init_db():
    """Initializes the database by creating all tables."""
    # Import models to ensure they are registered with Base.metadata
    from vertical_descent_app.data_layer import models
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency for getting a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
