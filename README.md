# ❄️ Summit Terminal

**High-Fidelity Alpine Weather Forecasting & Telemetry Dashboard**

Summit Terminal is an advanced Python-based application designed to bridge the gap between global weather models (NWP) and specific alpine summits. By utilizing a proprietary **Point Forecast Physics Engine**, it downscales coarse model data to estimate snowfall based on thermodynamic profiles, orographic lift, and dynamic microphysics.

The system features a **Data Abstraction Layer (DAL)** that unifies real-time telemetry (SNOTEL), atmospheric forecasts (NOAA, Open-Meteo), and historical verification data into a single, cohesive dashboard.

---

## 🚀 Key Features

### 1. The Logic Engine (Forecasting)
*   **Thermodynamic Downscaling:** Adjusts temperature based on the specific vertical difference between the model's grid elevation and the target peak using a standard lapse rate (-0.65°C/100m).
*   **Orographic Enhancement:** Applies a "Clausius-Clapeyron Proxy" to boost precipitation totals by 5% for every 100m of upslope vertical rise.
*   **Dynamic Microphysics:** Calculates Snow-Liquid Ratios (SLR) based on wet-bulb temperature and humidity regimes:
    *   **Rain:** > 1°C
    *   **"Cascade Concrete":** 1°C to -3°C (8:1 Ratio)
    *   **"Champagne Powder":** -12°C to -18°C + High RH (18:1 Ratio)
    *   **Modified Kuchera:** Standard cold accumulation for other regimes.

### 2. The Truth Engine (Verification)
*   **SNOTEL Harvester:** Automatically scrapes and ingests USDA telemetry data (Snow Water Equivalent, Depth, Temperature) into a local SQLite database.
*   **Forecast Snapshots:** Captures model predictions at issue time to verify accuracy against observed reality later.

### 3. The Dashboard (UI)
*   **Live Telemetry:** Real-time view of current conditions (Base Depth, SWE).
*   **Ensemble Analysis:** Compare multiple weather models (ECMWF, GFS, GEM, ICON) side-by-side.
*   **Atmospheric Outlook:** 7-day temperature and wind forecasts from NOAA.
*   **Theme Support:** Switch between Light and Dark modes.

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.8+
*   Internet connection (for API calls)

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-repo/summit-terminal.git
cd summit-terminal

# Create a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Configuration

### SNOTEL Stations
Telemetry stations are configured in `vertical_descent_app/config/stations.yaml`. You can map USDA station triplets to internal location IDs.

```yaml
stations:
  - triplet: "1186:CO:SNTL"
    name: "Winter Park (Middle Fork Camp)"
    location_id: 1
  # Add more stations here...
```

### Resort Configuration
Forecast locations are currently defined in the `RESORTS` dictionary within the code (e.g., `vertical_descent_app/logic_engine/forecasting.py`). Each resort requires:
*   Latitude/Longitude
*   SNOTEL IDs
*   Elevation profile (Base/Peak) for the Physics Engine.

---

## 🖥️ Usage

### 1. Initialize & Ingest Data (The Truth Engine)
Before running the dashboard, populate the database with recent telemetry data.

```bash
# Fetch recent SNOTEL data (defaults to last 7 days or config)
python scripts/run_harvester.py

# Optional: Fetch a specific date range
python scripts/run_harvester.py --start 2023-10-01 --end 2023-10-31
```

### 2. Generate Forecast Snapshots (Optional)
To build a history of forecasts for verification, run the snapshot script. This is typically run via a cron job (e.g., every 6 hours).

```bash
python scripts/run_snapshot.py
```

### 3. Launch the Dashboard
Start the Streamlit application to visualize the data.

```bash
streamlit run vertical_descent_app/app.py
```

The application will open automatically in your default web browser (typically `http://localhost:8501`).

---

## 📂 Project Structure

```text
summit-terminal/
├── scripts/
│   ├── run_harvester.py    # Fetches SNOTEL data
│   └── run_snapshot.py     # Captures forecast snapshots
├── vertical_descent_app/
│   ├── app.py              # Main Streamlit Dashboard (v5)
│   ├── config/             # Configuration files (stations.yaml)
│   ├── dashboard/          # UI Components
│   ├── data_layer/         # Database models & DAL
│   ├── logic_engine/       # Physics & Forecasting logic
│   └── truth_engine/       # Harvester & Verification logic
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

*   **Grid Resolution:** Open-Meteo Free Tier uses 0.4° (ECMWF) and 0.25° (GFS) resolution. The Physics Engine corrects for elevation but cannot predict hyper-localized micro-climates.
*   **Inversion Blindness:** Standard lapse rates may overestimate summit coldness during strong inversions.
*   **SNOTEL Latency:** USDA data is real-time but can occasionally lag by 1-2 hours.

---

## 📄 License

MIT License. Free for personal use.

Data provided by [Open-Meteo](https://open-meteo.com/) (CC BY 4.0), NOAA, and USDA.
