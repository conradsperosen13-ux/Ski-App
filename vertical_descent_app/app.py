import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from datetime import datetime, timedelta
import logging
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pytz
from plotly.subplots import make_subplots
from vertical_descent_app.logic_engine import (
    PointForecastEngine, 
    run_async_forecast, 
    get_raw_forecast_data,
    calculate_swe_ratio, 
    get_noaa_forecast as logic_get_noaa_forecast,
    get_snotel_data as logic_get_snotel_data,
    calculate_forecast_metrics,
    build_forecast_figure
)

# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Summit Terminal",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 2. THEME & GLOBAL STYLES
# =============================================================================
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

is_dark = st.session_state["theme"] == "dark"

with st.sidebar:
    st.markdown(
        '<p style="font-size:0.7rem; letter-spacing:0.12em; text-transform:uppercase; '
        'opacity:0.5; margin-bottom:0.75rem;">Data Sources</p>',
        unsafe_allow_html=True
    )
    
    st.divider()
    # ... rest of sidebar
    st.markdown(
        '<p style="font-size:0.7rem; letter-spacing:0.12em; text-transform:uppercase; '
        'opacity:0.5; margin-bottom:0.75rem;">Data Sources</p>',
        unsafe_allow_html=True
    )

    use_nwp = st.toggle("🌐 Live NWP Data (Open-Meteo)", value=True)
    st.session_state["use_nwp"] = use_nwp
    st.divider()
    st.markdown(
        '<p style="font-size:0.65rem; opacity:0.3; text-align:center;">Summit Terminal · v4 · NWP Integrated</p>',
        unsafe_allow_html=True
    )

# =============================================================================
# 3. COLOUR PALETTE
# =============================================================================
if is_dark:
    BG_BASE            = "#09090b"
    BG_CARD            = "rgba(30, 30, 35, 0.55)"
    BG_CARD_HOVER      = "rgba(42, 42, 50, 0.75)"
    BORDER             = "rgba(255, 255, 255, 0.07)"
    BORDER_HOVER       = "rgba(255, 255, 255, 0.16)"
    TEXT_PRI           = "#f4f4f5"
    TEXT_SEC           = "#a1a1aa"
    TEXT_MUTED         = "#52525b"
    SIDEBAR_BG         = "rgba(9, 9, 11, 0.95)"
    PLOTLY_GRID        = "rgba(255,255,255,0.05)"
    PLOTLY_FONT        = "#71717a"
    PLOTLY_HOVER       = "#18181b"
    SHADOW             = "0 8px 32px rgba(0,0,0,0.45)"
    SHADOW_HOVER       = "0 20px 48px rgba(0,0,0,0.65)"
    NOISE_OPACITY      = "0.025"
    METRIC_LABEL_COLOR = "#52525b"
    METRIC_VAL_COLOR   = "#f4f4f5"
    BLUR_AMOUNT        = "18px"
else:
    BG_BASE            = "#f8fafc"
    BG_CARD            = "rgba(255, 255, 255, 1.0)"
    BG_CARD_HOVER      = "rgba(255, 255, 255, 1.0)"
    BORDER             = "rgba(0, 0, 0, 0.15)"
    BORDER_HOVER       = "rgba(0, 0, 0, 0.3)"
    TEXT_PRI           = "#09090b"
    TEXT_SEC           = "#52525b"
    TEXT_MUTED         = "#a1a1aa"
    SIDEBAR_BG         = "rgba(255, 255, 255, 1.0)"
    PLOTLY_GRID        = "rgba(0,0,0,0.1)"
    PLOTLY_FONT        = "#52525b"
    PLOTLY_HOVER       = "#ffffff"
    SHADOW             = "0 4px 16px rgba(0,0,0,0.08)"
    SHADOW_HOVER       = "0 12px 32px rgba(0,0,0,0.12)"
    NOISE_OPACITY      = "0.015"
    METRIC_LABEL_COLOR = "#52525b"
    METRIC_VAL_COLOR   = "#09090b"
    BLUR_AMOUNT        = "5px"

ACCENT_BLUE  = "#38bdf8"
ACCENT_TEAL  = "#2dd4bf"
ACCENT_ROSE  = "#fb7185"
ACCENT_AMBER = "#fbbf24"

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=PLOTLY_FONT, size=11),
        xaxis=dict(gridcolor=PLOTLY_GRID, zeroline=False, showline=False),
        yaxis=dict(gridcolor=PLOTLY_GRID, zeroline=False, showline=False),
        margin=dict(l=0, r=0, t=20, b=20),
        legend=dict(orientation="h", y=1.06, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=PLOTLY_HOVER,
            bordercolor=BORDER,
            font=dict(family="Inter, sans-serif", size=12, color=TEXT_PRI)
        )
    )
)

# =============================================================================
# 4. CSS  (single clean block — duplicate removed)
# =============================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;500;600;800&display=swap');

:root {{
  --bg-base:       {BG_BASE};
  --bg-card:       {BG_CARD};
  --bg-card-hover: {BG_CARD_HOVER};
  --border:        {BORDER};
  --border-hover:  {BORDER_HOVER};
  --text-pri:      {TEXT_PRI};
  --text-sec:      {TEXT_SEC};
  --text-muted:    {TEXT_MUTED};
  --shadow:        {SHADOW};
  --shadow-hover:  {SHADOW_HOVER};
  --accent-blue:   {ACCENT_BLUE};
  --accent-teal:   {ACCENT_TEAL};
  --accent-rose:   {ACCENT_ROSE};
  --accent-amber:  {ACCENT_AMBER};
  --font-display:  'Inter', sans-serif;
  --font-mono:     'JetBrains Mono', monospace;
  --radius-sm:     8px;
  --radius-md:     14px;
  --blur:          {BLUR_AMOUNT};
  --noise-opacity: {NOISE_OPACITY};
}}

html, body, [class*="css"] {{
    font-family: var(--font-display);
    color: var(--text-pri);
    background-color: var(--bg-base) !important;
    -webkit-font-smoothing: antialiased;
}}
.main {{ background: var(--bg-base) !important; }}
.block-container {{
    padding: 2.5rem 2rem 5rem !important;
    max-width: 1400px !important;
    margin: 0 auto;
}}

section[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG} !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid var(--border) !important;
}}

.glass-card {{
    position: relative;
    background: var(--bg-card);
    backdrop-filter: blur(var(--blur));
    -webkit-backdrop-filter: blur(var(--blur));
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    box-shadow: var(--shadow);
    transition: transform 0.25s cubic-bezier(0.4,0,0.2,1),
                box-shadow 0.25s cubic-bezier(0.4,0,0.2,1),
                border-color 0.25s ease;
    overflow: hidden;
    margin-bottom: 0.85rem;
}}
.glass-card::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='1'/%3E%3C/svg%3E");
    opacity: var(--noise-opacity);
    border-radius: inherit;
    pointer-events: none;
    z-index: 0;
}}
.glass-card > * {{ position: relative; z-index: 1; }}
.glass-card:hover {{
    background: var(--bg-card-hover);
    border-color: var(--border-hover);
    box-shadow: var(--shadow-hover);
    transform: translateY(-3px);
}}
.glass-card-accent {{ border-top: 2px solid var(--accent-blue); }}

.hero-stat {{
    font-size: clamp(3.5rem, 6vw, 5.5rem);
    font-weight: 800;
    line-height: 1;
    background: linear-gradient(160deg, {TEXT_PRI} 30%, {TEXT_SEC});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.04em;
}}
.hero-label {{
    font-family: var(--font-mono);
    text-transform: uppercase;
    color: var(--accent-blue);
    font-size: 0.75rem;
    letter-spacing: 0.14em;
    margin-bottom: 0.6rem;
}}
.hero-sub {{
    font-size: 1.1rem;
    color: var(--text-sec);
    font-weight: 400;
    margin-top: 0.35rem;
}}

.section-label {{
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.85rem;
    margin-top: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}}
.section-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}}

.pill {{
    display: inline-block;
    padding: 0.22rem 0.65rem;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border: 1px solid transparent;
}}
.pill-blue  {{ background:rgba(56,189,248,0.1);  color:{ACCENT_BLUE};  border-color:rgba(56,189,248,0.2); }}
.pill-teal  {{ background:rgba(45,212,191,0.1);  color:{ACCENT_TEAL};  border-color:rgba(45,212,191,0.2); }}
.pill-rose  {{ background:rgba(251,113,133,0.1); color:{ACCENT_ROSE};  border-color:rgba(251,113,133,0.2); }}
.pill-amber {{ background:rgba(251,191,36,0.1);  color:{ACCENT_AMBER}; border-color:rgba(251,191,36,0.2); }}

.live-dot {{
    height: 7px; width: 7px;
    background: {ACCENT_TEAL};
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
    animation: pulse-dot 2.4s cubic-bezier(0.4,0,0.6,1) infinite;
}}
@keyframes pulse-dot {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(45,212,191,0.4); }}
    50%       {{ box-shadow: 0 0 0 5px rgba(45,212,191,0); }}
}}

.outlook-strip {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}}
.outlook-card {{
    background: var(--bg-card);
    backdrop-filter: blur(var(--blur));
    -webkit-backdrop-filter: blur(var(--blur));
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1rem 0.85rem;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}}
.outlook-card:hover {{
    transform: translateY(-2px);
    border-color: var(--border-hover);
}}

.chart-wrap {{
    background: var(--bg-card);
    backdrop-filter: blur(var(--blur));
    -webkit-backdrop-filter: blur(var(--blur));
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem 0.5rem 0.5rem;
    box-shadow: var(--shadow);
}}

[data-testid="stMetric"] {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 1rem 1.1rem !important;
    backdrop-filter: blur(var(--blur)) !important;
}}
[data-testid="stMetricLabel"] > div {{
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: {METRIC_LABEL_COLOR} !important;
    font-weight: 600 !important;
}}
[data-testid="stMetricValue"] > div {{
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: {METRIC_VAL_COLOR} !important;
}}
[data-testid="stMetricDelta"] {{ display: none !important; }}

@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.fade-up   {{ animation: fadeUp 0.5s cubic-bezier(0.4,0,0.2,1) both; }}
.fade-up-1 {{ animation-delay: 0.05s; }}
.fade-up-2 {{ animation-delay: 0.12s; }}
.fade-up-3 {{ animation-delay: 0.20s; }}
.fade-up-4 {{ animation-delay: 0.28s; }}
.fade-up-5 {{ animation-delay: 0.36s; }}

.stSelectbox > div > div {{
    background-color: transparent !important;
    border: none !important;
    border-bottom: 1.5px solid var(--border-hover) !important;
    border-radius: 0 !important;
}}
.stSelectbox div[data-baseweb="select"] span {{
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: var(--text-pri) !important;
}}
.stMultiSelect [data-baseweb="tag"] {{
    background: rgba(56,189,248,0.1) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
}}
.stButton > button {{
    background: transparent !important;
    border: 1px solid var(--border-hover) !important;
    color: var(--text-sec) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 1rem !important;
    transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease !important;
    width: 100% !important;
}}
.stButton > button:hover {{
    background: rgba(56,189,248,0.08) !important;
    border-color: {ACCENT_BLUE} !important;
    color: {ACCENT_BLUE} !important;
}}
.stToggle label span, .stCheckbox label {{ color: var(--text-sec) !important; }}
.stDataFrame {{ border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; }}
.stTextArea textarea {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border-hover) !important;
    color: var(--text-pri) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
}}
.stCaption {{ color: var(--text-muted) !important; font-size: 0.7rem !important; letter-spacing: 0.08em !important; }}
.stExpander {{ border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; }}
.streamlit-expanderHeader {{
    background: var(--bg-card) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-sec) !important;
    font-size: 0.8rem !important;
}}

#MainMenu, footer, .stDeployButton {{ visibility: hidden; }}

[data-testid="stSidebarCollapsedControl"] {{
    display: flex !important;
    visibility: visible !important;
    left: 20px !important;
    top: 20px !important;
    z-index: 1000001 !important;
    color: white !important;
    background-color: #38bdf8 !important; /* Bright blue so we can see it */
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    justify-content: center !important;
    align-items: center !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
}}

.mountain-control {{
    background: linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(56,189,248,0.05) 100%), 
                url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 144 32'%3E%3Cpath fill='rgba(255,255,255,0.08)' d='M0,32L16,24L32,16L48,22L64,8L80,18L96,12L112,20L128,4L144,28L144,32Z'/%3E%3C/svg%3E") no-repeat bottom;
    background-size: cover;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1rem 0.5rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
}}

.mountain-control .stRadio > div {{ 
    gap: 1rem; 
    align-items: center; 
}}
/* Nested Mountain Tier Container */
.mtn-selector-wrap {{
    position: relative;
    width: 100%;
    height: 250px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
    padding-bottom: 20px;
}}

/* Shared Shape Logic */
.tier-shape {{
    position: absolute;
    width: 100%;
    background: var(--bg-card);
    border-bottom: 3px solid var(--border);
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
}}

/* PEAK: True Triangle */
.peak-shape {{
    height: 70px;
    width: 80px;
    bottom: 160px;
    clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
}}
.peak-shape.active {{ 
    background: rgba(251, 113, 133, 0.15); 
    border-bottom: 3px solid var(--accent-rose);
    filter: drop-shadow(0 0 10px var(--accent-rose));
}}

/* MID: Trapezoid */
.mid-shape {{
    height: 75px;
    width: 160px;
    bottom: 85px;
    clip-path: polygon(25% 0%, 75% 0%, 100% 100%, 0% 100%);
}}
.mid-shape.active {{ 
    background: rgba(45, 212, 191, 0.15); 
    border-bottom: 3px solid var(--accent-teal);
    filter: drop-shadow(0 0 10px var(--accent-teal));
}}

/* BASE: Wide Trapezoid */
.base-shape {{
    height: 85px;
    width: 240px;
    bottom: 0px;
    clip-path: polygon(15% 0%, 85% 0%, 100% 100%, 0% 100%);
}}
.base-shape.active {{ 
    background: rgba(56, 189, 248, 0.15); 
    border-bottom: 3px solid var(--accent-blue);
    filter: drop-shadow(0 0 10px var(--accent-blue));
}}

/* Invisible Buttons overlayed on the shapes */
.mtn-btn-overlay {{
    position: absolute;
    z-index: 10;
}}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 5. LOGGING
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# 6. NWP FORECAST ENGINE
# =============================================================================

def debug_nwp_api(lat, lon):
    """Debug function to display raw API response info in sidebar."""
    data = get_raw_forecast_data(lat, lon)
    if data:
        st.sidebar.write("✅ NWP API Connected (Cached/Live)")
        if "hourly" in data:
            keys       = list(data["hourly"].keys())
            model_keys = [k for k in keys if any(m in k for m in ["ecmwf", "gfs", "jma", "icon", "gem"])]
            st.sidebar.write(f"Hours: {len(data['hourly'].get('time', []))}")
            st.sidebar.write(f"Keys: {keys[:5]}...")
            st.sidebar.write(f"Models found: {len(model_keys)}")
            return data
        else:
            st.sidebar.error("No hourly data in response")
            return None
    else:
        st.sidebar.error("Failed to fetch raw data")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_nwp_forecast(lat: float, lon: float, elev_config: dict, resort_name: str, tz_str: str) -> pd.DataFrame:
    return run_async_forecast(lat, lon, elev_config, resort_name, tz_str)

# =============================================================================
# 7. DATA ENGINE (Cached Wrappers)
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_noaa_forecast(lat, lon, retries=3):
    return logic_get_noaa_forecast(lat, lon, retries)

@st.cache_data(ttl=86400, show_spinner=False)
def get_snotel_data(site_ids, state="CO"):
    return logic_get_snotel_data(site_ids, state)

@st.cache_data(show_spinner=False)
def get_forecast_metrics_cached(df, selected_models, current_depth, selected_band, tz_str):
    return calculate_forecast_metrics(df, selected_models, current_depth, selected_band, tz_str)

# =============================================================================
# 8. SESSION STATE
# =============================================================================
if "raw_model_data" not in st.session_state:
    st.session_state["raw_model_data"] = pd.DataFrame()
if "model_totals" not in st.session_state:
    st.session_state["model_totals"] = {}
if "nwp_data" not in st.session_state:
    st.session_state["nwp_data"] = pd.DataFrame()
if "use_nwp" not in st.session_state:
    st.session_state["use_nwp"] = True

# =============================================================================
# 9. RESORT DATABASE & SEARCH
# =============================================================================
@st.cache_data
def load_resort_db():
    try:
        df = pd.read_csv("us_resorts.csv")
        return df
    except FileNotFoundError:
        st.error("Critical Failure: us_resorts.csv not found.")
        st.stop()

df_resorts = load_resort_db()

# Top Navigation Row
top_left, top_right = st.columns([3, 1])
with top_left:
    st.markdown('<div class="section-label" style="margin-top:0;">Station Selection</div>', unsafe_allow_html=True)
    
    search_options = df_resorts["Search_String"].tolist()
    
    # Intellingent Search Bar
    selected_search = st.selectbox(
        "Search", 
        options=search_options,
        index=search_options.index("Arapahoe Basin, CO") if "Arapahoe Basin, CO" in search_options else 0,
        placeholder="Search US resorts (e.g. Steamboat)...",
        label_visibility="collapsed"
    )

with top_right:
    st.write("")
    if st.button("＋ Request Spot"):
        st.toast("Database update scheduled.", icon="💾")

# Map the search result back to the configuration dictionary
conf = df_resorts[df_resorts["Search_String"] == selected_search].iloc[0].to_dict()
selected_loc = conf["Resort_Name"]
resort_tz    = conf["Timezone"]

# Helper: Build elevation config for the NWP engine
conf["lat"] = float(conf["Lat"])
conf["lon"] = float(conf["Lon"])
conf["elev_ft"] = {
    "Summit": int(conf["Peak_ft"]),
    "Mid":    int((conf["Peak_ft"] + conf["Base_ft"]) / 2),
    "Base":   int(conf["Base_ft"])
}
# Map Snotel IDs back to list format
conf["snotel_ids"] = [x.strip() for x in str(conf["Snotel_IDs"]).split(",")]

# =============================================================================
# 10. STATION MAPPING & DATA LOADING
# =============================================================================

# 1. Map Search Result to Config Dictionary
conf = df_resorts[df_resorts["Search_String"] == selected_search].iloc[0].to_dict()
selected_loc = conf["Resort_Name"]
resort_tz    = conf["Timezone"]

# 2. Pre-process Telemetry Parameters
conf["lat"] = float(conf["Lat"])
conf["lon"] = float(conf["Lon"])
conf["elev_ft"] = {
    "Summit": int(conf["Peak_ft"]),
    "Mid":    int((conf["Peak_ft"] + conf["Base_ft"]) / 2),
    "Base":   int(conf["Base_ft"])
}
# Convert comma-string "602, 505" into list ["602", "505"]
conf["snotel_ids"] = [x.strip() for x in str(conf["Snotel_IDs"]).split(",")]

# 3. Execute Autonomous API Pipeline
with st.spinner(f"Syncing {selected_loc}…"):
    noaa_df, grid_elev = get_noaa_forecast(conf["lat"], conf["lon"])
    snotel_df = get_snotel_data(conf["snotel_ids"], conf.get("State", "CO"))

    if st.session_state.get("use_nwp", True):
        with st.spinner("Fetching live NWP ensemble..."):
            nwp_df = get_nwp_forecast(conf["lat"], conf["lon"], conf["elev_ft"], selected_loc, resort_tz)
    else:
        nwp_df = pd.DataFrame()

# 4. Finalize Dataframes
df_models = nwp_df.copy()

# 5. Extract Current SNOTEL Telemetry
current_swe, current_depth = 0.0, 0.0
if not snotel_df.empty:
    snotel_df["Date"] = pd.to_datetime(snotel_df["Date"])
    # Sort and drop NaNs to find the most recent physical snow depth
    latest = snotel_df.sort_values("Date").dropna(subset=["SWE"]).iloc[-1] if not snotel_df.empty else None
    if latest is not None:
        current_depth = float(latest.get("Depth", 0) or 0)
        current_swe   = float(latest.get("SWE", 0) or 0)

# =============================================================================
# 11. HERO SECTION (Adaptive Instrument Cluster)
# =============================================================================
st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
st.markdown("---")

# 1. Initialize State and Metrics Variable
if "selected_band" not in st.session_state:
    st.session_state.selected_band = "Mid"

metrics = None # Prevents NameError

# 2. Layout
h_col1, h_col2, h_col3 = st.columns([2.2, 4.8, 4], gap="medium")

with h_col1:
    st.markdown('<div class="section-label">Elevation Profile</div>', unsafe_allow_html=True)
    
    # Nested Button Selection Logic
    # These are standard buttons but we will position the CSS mountain behind them
    if st.button("▲ PEAK", use_container_width=True, key="btn_peak"):
        st.session_state.selected_band = "Peak"
        st.toast("Telemetry: Summit", icon="🏔️")
    if st.button("■ MID", use_container_width=True, key="btn_mid"):
        st.session_state.selected_band = "Mid"
        st.toast("Telemetry: Mid-Mountain", icon="🏔️")
    if st.button("▼ BASE", use_container_width=True, key="btn_base"):
        st.session_state.selected_band = "Base"
        st.toast("Telemetry: Base/Lodge", icon="🏔️")

    selected_band = st.session_state.selected_band

    # Render the Visual Mountain Shape
    st.markdown(f"""
    <div class="mtn-selector-wrap fade-up">
        <div class="tier-shape peak-shape {'active' if selected_band == 'Peak' else ''}"></div>
        <div class="tier-shape mid-shape {'active' if selected_band == 'Mid' else ''}"></div>
        <div class="tier-shape base-shape {'active' if selected_band == 'Base' else ''}"></div>
    </div>
    """, unsafe_allow_html=True)

# 3. Calculate Physics Metrics Immediately
if not df_models.empty:
    all_models = list(df_models["Model"].unique())
    metrics = get_forecast_metrics_cached(df_models, all_models, current_depth, selected_band, resort_tz)

with h_col2:
    # ... (Hero Value logic remains the same, but now safely uses 'metrics')
    hero_val, hero_label, hero_sub = "—", "AWAITING DATA", "Awaiting API..."
    if metrics:
        hc = metrics["hero_context"]
        hero_val = f"{hc['peak_amount']:.1f}\"" if hc["condition"] == "SNOW" else "DRY"
        hero_label = "ESTIMATED BEST DAY" if hc["condition"] == "SNOW" else "ATMOSPHERE STABLE"
        hero_sub = f"Target: {hc['best_day'].strftime('%a, %b %d')}" if hc["best_day"] else hc["timing_note"]

    st.markdown(f"""
    <div class="fade-up fade-up-1" style="text-align: center; border-left: 1px solid var(--border); border-right: 1px solid var(--border); padding: 0 1rem;">
        <div class="hero-label">{hero_label}</div>
        <div class="hero-stat">{hero_val}</div>
        <div class="hero-sub">{hero_sub}</div>
    </div>
    """, unsafe_allow_html=True)

with h_col3:
    st.markdown('<div class="section-label">Telemetry</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card glass-card-accent fade-up fade-up-2" style="padding:1rem;">', unsafe_allow_html=True)
    
    m1, m2 = st.columns(2)
    m1.metric("Base Depth", f'{current_depth:.0f}"')
    m2.metric("SWE Total", f'{current_swe:.2f}"')
    
    m3, m4 = st.columns(2)
    m3.metric("Live SLR", f"{calculate_swe_ratio(noaa_df.iloc[0]['Temp']) if not noaa_df.empty else 0}:1")
    # SAFE CHECK FOR METRICS
    m4.metric("Spread", f"{metrics['avg_spread']:.1f}\"" if metrics else "—")
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# 12. ATMOSPHERIC OUTLOOK (Hybrid Elevation-Aware)
# =============================================================================
st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label fade-up">Atmospheric Outlook</div>', unsafe_allow_html=True)

if not noaa_df.empty:
    now_naive = datetime.now()
    shorts = noaa_df[noaa_df["Time"] > now_naive].head(96).copy()
    shorts["Date"] = shorts["Time"].dt.date
    
    # Aggregate NOAA base data
    daily_noaa = shorts.groupby("Date").agg(
        Cond=("Summary", lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]),
        Wind=("Wind", "max"),
        Max_NOAA=("Temp", "max"),
        Min_NOAA=("Temp", "min")
    ).reset_index().head(4)
    
    cards_html = '<div class="outlook-strip fade-up">'
    
    for _, row in daily_noaa.iterrows():
        # Default to NOAA temps as the baseline
        max_f, min_f = row["Max_NOAA"], row["Min_NOAA"]
        
        # Only overwrite with NWP if the physics engine has successfully labeled the data
        if not nwp_df.empty and "Band" in nwp_df.columns:
            nwp_day = nwp_df[(nwp_df["Band"] == selected_band) & (nwp_df["Date"].dt.date == row["Date"])]
            if not nwp_day.empty:
                max_f = (nwp_day["Temp_C"].max() * 9/5) + 32
                min_f = (nwp_day["Temp_C"].min() * 9/5) + 32

        is_snow = "Snow" in str(row["Cond"])
        pill_cls = "pill-blue" if is_snow else "pill-rose"
        
        cards_html += f"""
        <div class="outlook-card">
            <div style="font-size:0.7rem; color:var(--text-muted);">{row['Date'].strftime('%a %d')}</div>
            <div style="font-size:1.2rem; font-weight:700;">{max_f:.0f}° / {min_f:.0f}°</div>
            <span class="pill {pill_cls}">{str(row['Cond'])[:15]}</span>
            <div style="font-size:0.7rem; margin-top:0.5rem;">💨 {row['Wind']:.0f} mph</div>
        </div>"""
    
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)
else:
    st.caption("NOAA data unavailable")

# =============================================================================
# 13. ENSEMBLE ANALYSIS
# =============================================================================
st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label fade-up fade-up-5">Ensemble Analysis</div>', unsafe_allow_html=True)

if not df_models.empty and metrics:
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2.5, 1, 1, 1], gap="small")
    with ctrl1:
        # Filter out the static Average line to prevent double plotting
        u_models = [m for m in sorted(df_models["Model"].unique()) if "Average" not in m]
        
        # Default to showing the first 3 models to keep initial noise low
        default_models = u_models[:3]
        
        selected_models = st.multiselect(
            "Active Models", u_models, default=default_models, label_visibility="collapsed"
        )
    
    # --- THESE MUST BE INDENTED ---
    with ctrl2:
        show_spread    = st.toggle("Spread",    value=True)
    with ctrl3:
        show_spaghetti = st.toggle("Spaghetti", value=False)
    with ctrl4:
        show_extended  = st.toggle("Extended",  value=False)

    if show_extended:
        st.info(
            "⚠️ **Extended forecast**: 3‑day ~95% accuracy, 5‑day ~90%, 7‑day 80‑85%, 10‑day+ ~50%. "
            "(NOAA/NASA) · NWP data includes physics-based SLR calculations."
        )

    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
    # ------------------------------
    if selected_models:
        # Pass resort_tz
        metrics_sub = get_forecast_metrics_cached(df_models, selected_models, current_depth, selected_band, resort_tz)

        if metrics_sub:
            daily_stats = metrics_sub["daily_stats"]
            noaa_cutoff = noaa_df["Time"].max() if not noaa_df.empty else None

            if not show_extended:
                # Cap at 7 days for standard view
                data_tz = pytz.timezone(resort_tz)
                cutoff = pd.Timestamp.now(tz=data_tz).normalize() + pd.Timedelta(days=7)
                windowed = daily_stats[daily_stats["Date"] <= cutoff].copy()
                w_end_param = cutoff
            else:
                # Full NWP duration (usually 10-16 days)
                windowed = daily_stats.copy()
                w_end_param = daily_stats["Date"].max()

            # Pass w_end_param to the build_forecast_figure call below
                # -------------------------------------------------------------------

            if len(daily_stats) > 1:
                date_min     = daily_stats["Date"].min().date()
                date_max     = daily_stats["Date"].max().date()
                slider_range = st.select_slider(
                    "Storm Window",
                    options=pd.date_range(date_min, date_max).date.tolist(),
                    value=(date_min, date_max),
                    format_func=lambda d: d.strftime("%b %d"),
                )
                w_start  = pd.Timestamp(slider_range[0]).tz_localize(resort_tz)
                w_end    = pd.Timestamp(slider_range[1]).tz_localize(resort_tz).replace(hour=23, minute=59)
                windowed = daily_stats[
                    (daily_stats["Date"] >= w_start) & (daily_stats["Date"] <= w_end)
                ].copy()
                windowed = daily_stats[
                    (daily_stats["Date"] >= w_start) & (daily_stats["Date"] <= w_end)
                ].copy()
                window_total = windowed["mean"].sum()
            else:
                windowed     = daily_stats.copy()
                window_total = daily_stats["mean"].sum()

            sm1, sm2, sm3, sm4 = st.columns(4)
            with sm1:
                st.metric("Window Total", f'{metrics_sub["total_snowfall"]:.1f}"')
            with sm2:
                # Use the new peak_amount from our strike window logic
                peak_val = metrics_sub["hero_context"]["peak_amount"]
                peak_day = metrics_sub["hero_context"]["best_day"]
                st.metric(
                    "Best Window",
                    f'{peak_val:.1f}"',
                    delta=peak_day.strftime("%a %b %d") if peak_day else None
                )
            with sm3:
                st.metric("Models Active", len(selected_models))
            with sm4:
                st.metric("Avg Spread", f'{metrics_sub["avg_spread"]:.1f}"')

            st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            fig = build_forecast_figure(
                stats=windowed,
                noaa_df=noaa_df,
                show_spaghetti=show_spaghetti,
                show_ribbon=show_spread,  # <--- UPDATE THIS to match your new toggle
                df_models=df_models,
                selected_models=selected_models,
                noaa_cutoff_dt=noaa_cutoff,
                show_extended=show_extended,
                w_end=w_end,        
                band_filter=selected_band,
                is_dark=is_dark,
            )
            st.plotly_chart(fig, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
    
if not nwp_df.empty and "Band" in nwp_df.columns:
    with st.expander("NWP Physics Details"):
        # ... logic stays the same, just use nwp_df
        latest_nwp = nwp_df[nwp_df["Band"] == selected_band].groupby("Model").last().reset_index()
        st.markdown("""
        **Physics Engine Parameters:**
        - Lapse Rate: 0.65°C per 100m
        - Orographic Lift Factor: 0.05 per 100m
        - Kuchera SLR: 12 + (-2 - temp_c)
        - DGZ Champaign: 18:1 SLR (Temp -12°C to -18°C, RH >80%)
        """)
        latest_nwp = nwp_df[nwp_df["Band"] == selected_band].groupby("Model").last().reset_index()
        if not latest_nwp.empty:
            cols = [c for c in ["Model", "Temp_C", "SLR", "Cloud_Cover", "Freezing_Level_m"] if c in latest_nwp.columns]
            st.dataframe(latest_nwp[cols].round(1), hide_index=True, width="stretch")

else:
    st.markdown(f"""
    <div style="padding:3rem 2rem; border:1px dashed {BORDER}; border-radius:14px;
                text-align:center; color:var(--text-muted);">
        <div style="font-size:1.5rem; margin-bottom:0.5rem;">❄</div>
        <div style="font-size:0.9rem;">No ensemble data — enable NWP in sidebar</div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# 14. SNOTEL VERIFICATION
# =============================================================================
st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">SNOTEL Verification</div>', unsafe_allow_html=True)

if not snotel_df.empty:
    sc1, sc2 = st.columns([1, 3], gap="large")

    with sc1:
        st.metric("Current Depth", f'{current_depth:.0f}"')
        st.metric("SWE Total",     f'{current_swe:.2f}"')
        elev_display = f"{grid_elev:,.0f} ft" if grid_elev else conf["elevation"]
        st.metric("Grid Elevation", elev_display)

    with sc2:
        recent = snotel_df.groupby("Date").mean(numeric_only=True).reset_index().tail(21)
        fig_s  = make_subplots(specs=[[{"secondary_y": True}]])

        if "SWE_Delta" in recent.columns:
            max_delta = recent["SWE_Delta"].max() or 0.01
            bar_cols  = [
                f"rgba(45,212,191,{min(0.25 + v / max_delta * 0.7, 0.92):.2f})"
                for v in recent["SWE_Delta"].fillna(0)
            ]
            fig_s.add_trace(go.Bar(
                x=recent["Date"], y=recent["SWE_Delta"],
                name="Daily SWE Δ",
                marker=dict(color=bar_cols, line=dict(width=0)),
                hovertemplate="<b>%{x|%b %d}</b>  SWE Δ: +%{y:.3f}\"<extra></extra>"
            ), secondary_y=False)

        if "Depth" in recent.columns:
            fig_s.add_trace(go.Scatter(
                x=recent["Date"], y=recent["Depth"],
                name="Snow Depth",
                line=dict(color=TEXT_PRI if is_dark else TEXT_SEC, width=2),
                opacity=0.7,
                hovertemplate="Depth: %{y:.0f}\"<extra></extra>"
            ), secondary_y=True)

        if "SWE" in recent.columns:
            fig_s.add_trace(go.Scatter(
                x=recent["Date"], y=recent["SWE"],
                name="SWE Total",
                line=dict(color=ACCENT_BLUE, width=1.5, dash="dot"),
                hovertemplate="SWE: %{y:.2f}\"<extra></extra>"
            ), secondary_y=False)

        layout_s = PLOTLY_TEMPLATE["layout"].to_plotly_json()
        layout_s.update(dict(
            height=260,
            margin=dict(t=10, b=10, l=0, r=0),
            yaxis2=dict(
                overlaying="y", side="right", showgrid=False, zeroline=False,
                tickfont=dict(color=TEXT_MUTED), title="Depth (in)"
            )
        ))
        fig_s.update_layout(layout_s)
        fig_s.update_yaxes(title_text="SWE (in)", secondary_y=False)
        st.plotly_chart(fig_s, width="stretch")

    with st.expander("SNOTEL Raw Records"):
        show_cols = [c for c in ["Date", "SiteID", "SWE", "SWE_Delta", "Depth", "Temp"] if c in snotel_df.columns]
        disp = snotel_df[show_cols].sort_values("Date", ascending=False).reset_index(drop=True)
        st.dataframe(disp, hide_index=True, width="stretch", height=240)

else:
    st.markdown(f"""
    <div style="padding:1.5rem; border:1px dashed {BORDER}; border-radius:10px;
                text-align:center; color:var(--text-muted); font-size:0.85rem;">
        SNOTEL data unavailable
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# 15. RAW TELEMETRY TABLES
# =============================================================================
st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Raw Telemetry</div>', unsafe_allow_html=True)

t1, t2, t3 = st.columns(3, gap="large")
with t1:
    st.caption("SNOTEL · Last 10 Days")
    if not snotel_df.empty:
        show_cols = [c for c in ["Date", "SWE", "Depth", "Temp"] if c in snotel_df.columns]
        st.dataframe(snotel_df.tail(10)[show_cols], width="stretch", hide_index=True)
    else:
        st.info("No SNOTEL data")

with t2:
    st.caption("Ensemble Daily Statistics")
    if metrics and "daily_stats" in metrics:
        disp_cols = [c for c in ["Date", "mean", "min", "max", "cumulative", "std"]
                     if c in metrics["daily_stats"].columns]
        st.dataframe(
            metrics["daily_stats"][disp_cols].round(2),
            width="stretch", hide_index=True
        )
    else:
        st.info("No forecast data")

with t3:
    st.caption("NWP Live Data")
    # Check if nwp_df exists and the 'Band' column is present 
    if not nwp_df.empty and "Band" in nwp_df.columns:
        # Match the band to the user-selected band from the slider 
        latest_nwp = nwp_df[nwp_df["Band"] == selected_band].groupby("Model").last().reset_index()
        
        # Ensure we only try to display columns that exist 
        cols = [c for c in ["Model", "Amount", "Temp_C", "SLR"] if c in latest_nwp.columns]
        st.dataframe(latest_nwp[cols].round(1), width="stretch", hide_index=True)
    else:
        st.info("No NWP data (enable in sidebar)")

st.markdown('<div style="height:3rem;"></div>', unsafe_allow_html=True)