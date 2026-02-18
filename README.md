❄️ Summit Terminal

High-Fidelity Alpine Weather Forecasting & Telemetry Dashboard

Summit Terminal is a Python-based desktop application designed to bridge the gap between coarse global weather models and specific alpine summits. Unlike standard weather apps that rely on static 10:1 snow-to-liquid ratios, Summit Terminal utilizes a proprietary Point Forecast Physics Engine to estimate snowfall based on thermodynamic downscaling, orographic lift, and dynamic microphysics.

🚀 Key Features

1. Research-Grade Physics Engine

Thermodynamic Downscaling: Automatically adjusts temperature based on the specific vertical difference between the model's grid elevation and the target peak using a standard lapse rate (-0.65°C/100m).

Orographic Enhancement: Applies a "Clausius-Clapeyron Proxy" to boost precipitation totals by 5% for every 100m of upslope vertical rise.

Dynamic Microphysics: Replaces the 10:1 ratio with a logic matrix:

Rain Phase: > 1°C

"Cascade Concrete": 1°C to -3°C (8:1 Ratio)

"Champagne Powder": -12°C to -18°C + High RH (18:1 Ratio)

Modified Kuchera: Standard cold accumulation for all other regimes.

2. The "Vibe" Engine

Translates raw meteorological data into skier-centric summaries.

Examples: "Where's your snorkel?", "DEFCON 1: It's Nuking", "Solid Reset."

Illustrative loading states provide visibility into the data fetch process (Telemetry -> Atmospherics -> Physics).

3. Multi-Source Integration

NWP Models: Live ingestion of GFS and ECMWF models via Open-Meteo (Free Tier).

SNOTEL: Direct scraping of USDA telemetry for live ground-truth (Snow Water Equivalent, Depth).

NOAA: hourly atmospheric outlooks for temperature and wind.

Manual Override: Paste raw data tables from third-party tools (like Snowiest.app) to override the internal physics engine.

🛠️ Installation & Setup

Prerequisites

Python 3.8+

Internet connection (for API calls)

1. Clone & Setup

# Create a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install streamlit pandas numpy plotly requests pytz


2. Run the Application

streamlit run summit_terminal.py


The application will open automatically in your default web browser (localhost:8501).

⚙️ Configuration

The application allows for forecast location customization directly within the code. Locate the RESORTS dictionary in summit_terminal.py:

RESORTS = {
    "Winter Park": {
        "lat": 39.8859, 
        "lon": -105.764,
        "snotel_ids": [1186, 335], # USDA Site IDs
        "snowiest_url": "...",
        "elev_ft": {"base": 9000, "peak": 12060} # Critical for Physics Engine
    },
    # Add your own resorts here...
}


Note: The elev_ft dictionary is critical. The Physics Engine uses these values to calculate the vertical "lift" required to adjust the forecast from the smoothed global grid to your specific peak.

⚠️ Limitations & Adversarial Reality Check

Grid Resolution: The Open-Meteo Free Tier uses 0.4° (ECMWF) and 0.25° (GFS) resolution. While the Physics Engine mathematically corrects for elevation, it cannot "see" hyper-localized convective micro-bursts that occur between grid points.

Inversion Blindness: The standard lapse rate (-0.65°C/100m) assumes a standard atmosphere. During strong temperature inversions (common in January mornings), the engine may predict colder summit temps than reality.

SNOTEL Latency: USDA data is real-time but can occasionally lag by 1-2 hours depending on the transmission window.

📄 License

MIT License. Free for personal use.

Data provided by Open-Meteo (CC BY 4.0), NOAA, and USDA.
