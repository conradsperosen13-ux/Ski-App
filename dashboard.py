import concurrent.futures
import logging
import re
import pytz
import time
from datetime import datetime, timedelta
from io import StringIO
import asyncio
import aiohttp
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Summit Terminal",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="collapsed",
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
        'opacity:0.5; margin-bottom:0.75rem;">Appearance</p>',
        unsafe_allow_html=True
    )
    theme_label = "☀️  Switch to Light" if is_dark else "🌙  Switch to Dark"
    if st.button(theme_label, key="theme_btn"):
        st.session_state["theme"] = "light" if is_dark else "dark"
        st.rerun()

    st.divider()
    st.markdown(
        '<p style="font-size:0.7rem; letter-spacing:0.12em; text-transform:uppercase; '
        'opacity:0.5; margin-bottom:0.75rem;">Data Sources</p>',
        unsafe_allow_html=True
    )

    # Add NWP data toggle
    use_nwp = st.toggle("🌐 Live NWP Data (Open-Meteo)", value=True)
    st.session_state["use_nwp"] = use_nwp

    paste_input = st.text_area(
        "Paste Snowiest Table",
        height=120,
        placeholder="Paste table from snowiest.app…",
        label_visibility="collapsed"
    )
    from_paste = st.button("⬆  Parse Data", key="sidebar_parse")

    st.divider()
    st.markdown(
        '<p style="font-size:0.65rem; opacity:0.3; text-align:center;">Summit Terminal · v4 · NWP Integrated</p>',
        unsafe_allow_html=True
    )

if is_dark:
    BG_BASE       = "#09090b"
    BG_CARD       = "rgba(30, 30, 35, 0.55)"
    BG_CARD_HOVER = "rgba(42, 42, 50, 0.75)"
    BORDER        = "rgba(255, 255, 255, 0.07)"
    BORDER_HOVER  = "rgba(255, 255, 255, 0.16)"
    TEXT_PRI      = "#f4f4f5"
    TEXT_SEC      = "#a1a1aa"
    TEXT_MUTED    = "#52525b"
    SIDEBAR_BG    = "rgba(9, 9, 11, 0.95)"
    PLOTLY_GRID   = "rgba(255,255,255,0.05)"
    PLOTLY_FONT   = "#71717a"
    PLOTLY_HOVER  = "#18181b"
    SHADOW        = "0 8px 32px rgba(0,0,0,0.45)"
    SHADOW_HOVER  = "0 20px 48px rgba(0,0,0,0.65)"
    NOISE_OPACITY = "0.025"
    METRIC_LABEL_COLOR = "#52525b"
    METRIC_VAL_COLOR   = "#f4f4f5"
    BLUR_AMOUNT        = "18px"
else:
    BG_BASE       = "#f8fafc"
    BG_CARD       = "rgba(255, 255, 255, 1.0)"
    BG_CARD_HOVER = "rgba(255, 255, 255, 1.0)"
    BORDER        = "rgba(0, 0, 0, 0.15)"
    BORDER_HOVER  = "rgba(0, 0, 0, 0.3)"
    TEXT_PRI      = "#09090b"
    TEXT_SEC      = "#52525b"
    TEXT_MUTED    = "#a1a1aa"
    SIDEBAR_BG    = "rgba(255, 255, 255, 1.0)"
    PLOTLY_GRID   = "rgba(0,0,0,0.1)"
    PLOTLY_FONT   = "#52525b"
    PLOTLY_HOVER  = "#ffffff"
    SHADOW        = "0 4px 16px rgba(0,0,0,0.08)"
    SHADOW_HOVER  = "0 12px 32px rgba(0,0,0,0.12)"
    NOISE_OPACITY = "0.015"
    METRIC_LABEL_COLOR = "#52525b"
    METRIC_VAL_COLOR   = "#09090b"
    BLUR_AMOUNT        = "5px"

ACCENT_BLUE  = "#38bdf8"
ACCENT_TEAL  = "#2dd4bf"
ACCENT_ROSE  = "#fb7185"
ACCENT_AMBER = "#fbbf24"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;500;600;800&display=swap');

:root {{
  --bg-base:       {{BG_BASE}};
  --bg-card:       {{BG_CARD}};
  --bg-card-hover: {{BG_CARD_HOVER}};
  --border:        {{BORDER}};
  --border-hover:  {{BORDER_HOVER}};
  --text-pri:      {{TEXT_PRI}};
  --text-sec:      {{TEXT_SEC}};
  --text-muted:    {{TEXT_MUTED}};
  --shadow:        {{SHADOW}};
  --shadow-hover:  {{SHADOW_HOVER}};
  --accent-blue:   {{ACCENT_BLUE}};
  --accent-teal:   {{ACCENT_TEAL}};
  --accent-rose:   {{ACCENT_ROSE}};
  --accent-amber:  {{ACCENT_AMBER}};
  --font-display:  'Inter', sans-serif;
  --font-mono:     'JetBrains Mono', monospace;
  --radius-sm:     8px;
  --radius-md:     14px;
  --blur:          {{BLUR_AMOUNT}};
  --noise-opacity: {{NOISE_OPACITY}};
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
    background: {{SIDEBAR_BG}} !important;
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
    background: linear-gradient(160deg, {{TEXT_PRI}} 30%, {{TEXT_SEC}});
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
.pill-blue  {{ background:rgba(56,189,248,0.1);  color:{{ACCENT_BLUE}};  border-color:rgba(56,189,248,0.2); }}
.pill-teal  {{ background:rgba(45,212,191,0.1);  color:{{ACCENT_TEAL}};  border-color:rgba(45,212,191,0.2); }}
.pill-rose  {{ background:rgba(251,113,133,0.1); color:{{ACCENT_ROSE}};  border-color:rgba(251,113,133,0.2); }}
.pill-amber {{ background:rgba(251,191,36,0.1);  color:{{ACCENT_AMBER}}; border-color:rgba(251,191,36,0.2); }}

.live-dot {{
    height: 7px; width: 7px;
    background: {{ACCENT_TEAL}};
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
    color: {{METRIC_LABEL_COLOR}} !important;
    font-weight: 600 !important;
}}
[data-testid="stMetricValue"] > div {{
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: {{METRIC_VAL_COLOR}} !important;
}}
[data-testid="stMetricDelta"] {{
    display: none !important;
}}

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
    border-color: {{ACCENT_BLUE}} !important;
    color: {{ACCENT_BLUE}} !important;
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
[data-testid="stToolbar"] {{ display: none; }}
</style>
""", unsafe_allow_html=True)

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
# 3. CONFIGURATION & CONSTANTS
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}
API_TIMEOUT = 30
STORM_THRESHOLD_IN = 2.0
SWE_RATIO_THRESHOLDS = [(32, 8), (28, 10), (24, 12), (20, 15), (10, 20), (0, 25)]
DEFAULT_COLD_RATIO = 30

# Cache configuration for NWP engine
CACHE_DIR = Path(".cache")
CACHE_TTL_SECONDS = 4 * 3600  # 4 Hours

DEMO_DATA = """Model / Date
Snow
18 Wed 	19 Thu 	20 Fri 	21 Sat 	22 Sun 	23 Mon 	24 Tue 	25 Wed 	26 Thu 	27 Fri 	28 Sat 	1 Sun 	 2 Mon 	 3 Tue 	 4 Wed 	 Total
ECMWF IFS 025 	 1.7" 	0.4" 	1.1" 	0.7" 	0" 	0" 	0.1" 	4" 	1.7" 	0.3" 	0.4" 	0" 	0.1" 	6" 	- 	 16"
GFS 3" 	0.9" 	0.9" 	0.2" 	0" 	0" 	0.2" 	0" 	0" 	0.9" 	0" 	0" 	0.7" 	0.1" 	0.2" 	7"
GEM 2" 	0.4" 	1.6" 	0.4" 	0" 	0" 	0" 	0.4" 	0.4" 	- 	 - 	 - 	 - 	 - 	 - 	 5"
JMA 0.9" 	0.7" 	0.2" 	0.5" 	0" 	0" 	0" 	4" 	2" 	8" 	- 	 - 	 - 	 - 	 - 	 16"
ICON 	0.2" 	0.1" 	0.8" 	0" 	0" 	0" 	0.1" 	- 	 - 	 - 	 - 	 - 	 - 	 - 	 - 	 1.1"
MeteoFrance 1.3" 	0.3" 	1.2" 	0.8" 	- 	 - 	 - 	 - 	 - 	 - 	 - 	 - 	 - 	 - 	 - 	 4"
Average 1.5" 	0.5" 	1" 	0.4" 	0" 	0" 	0.1" 	2" 	1" 	3" 	0.2" 	0" 	0.4" 	3" 	0.2" 	13"
Snowfall Prediction History (Estimated)
Model / Date
Snow
8 Sun 	 9 Mon 	 10 Tue 	11 Wed 	12 Thu 	13 Fri 	14 Sat 	15 Sun 	16 Mon 	17 Tue 	Total
ECMWF IFS 025 	 0" 	0" 	0.2" 	0.4" 	0.1" 	1.6" 	0.7" 	0" 	0" 	2" 	5"
GFS 0" 	0" 	0" 	0.1" 	0.2" 	0.4" 	0.2" 	0" 	0" 	0.4" 	1.2"
GEM 0" 	0" 	0.2" 	0.6" 	0.5" 	1.5" 	0.2" 	0" 	0" 	1.4" 	5"
JMA 0" 	0" 	0.8" 	0.4" 	0.1" 	0.1" 	0.3" 	0" 	0" 	1.1" 	3"
ICON 	0" 	0" 	0" 	0" 	0.3" 	0.8" 	0.2" 	0" 	0" 	1.1" 	3"
MeteoFrance 0" 	0" 	0.2" 	0.2" 	0.1" 	1.3" 	0.2" 	0" 	0" 	1.1" 	3"
Average 0" 	0" 	0.2" 	0.3" 	0.2" 	1" 	0.3" 	0" 	0" 	1.2" 	3"
"""

RESORTS = {
    "Winter Park": {
        "lat": 39.8859, "lon": -105.764,
        "snotel_ids": [1186, 335],
        "snowiest_url": "https://www.snowiest.app/winter-park/snow-forecasts",
        "elevation": "9,000 – 12,060 ft",
        "elev_ft": {"base": 9000, "peak": 12060}
    },
    "Steamboat": {
        "lat": 40.4855, "lon": -106.8336,
        "snotel_ids": [457, 709, 825],
        "snowiest_url": "https://www.snowiest.app/steamboat/snow-forecasts",
        "elevation": "6,900 – 10,568 ft",
        "elev_ft": {"base": 6900, "peak": 10568}
    },
    "Arapahoe Basin": {
        "lat": 39.6419, "lon": -105.8753,
        "snotel_ids": [602, 505],
        "snowiest_url": "https://www.snowiest.app/arapahoe-basin/snow-forecasts",
        "elevation": "10,780 – 13,050 ft",
        "elev_ft": {"base": 10780, "peak": 13050}
    }
}

# =============================================================================
# 4. NWP FORECAST ENGINE (Integrated from new engine)
# =============================================================================

# Debug function - at top level, NOT inside any class
def debug_nwp_api(lat, lon):
    """Debug function to see raw API response"""
    import requests
    models_list = [
        "ecmwf_ifs04", "gfs_seamless", "jma_seamless",
        "icon_seamless", "gem_global", "meteofrance_seamless"
    ]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation,temperature_2m,relative_humidity_700hPa,cloud_cover,freezing_level_height",
        "models": ",".join(models_list),
        "precipitation_unit": "inch",
        "timezone": "America/Denver"
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Debug NWP API error: {e}")
        return None

class PointForecastEngine:
    """
    Enterprise-Grade Numerical Weather Prediction (NWP) Processor.
    Integrated into Summit Terminal dashboard.
    """

    LAPSE_RATE_C_PER_100M = 0.65
    OROGRAPHIC_LIFT_FACTOR = 0.05

    MODEL_DISPLAY_MAP = {
        "ecmwf_ifs04": "ECMWF", "gfs_seamless": "GFS",
        "jma_seamless": "JMA", "icon_seamless": "ICON",
        "gem_global": "GEM", "meteofrance_seamless": "MeteoFrance"
    }

    @classmethod
    def get_cache_path(cls, lat: float, lon: float) -> Path:
        CACHE_DIR.mkdir(exist_ok=True)
        filename = f"forecast_{lat:.4f}_{lon:.4f}.json"
        return CACHE_DIR / filename

    @classmethod
    def read_from_cache(cls, lat: float, lon: float) -> Optional[Dict]:
        path = cls.get_cache_path(lat, lon)
        if not path.exists():
            return None

        last_modified = path.stat().st_mtime
        if (time.time() - last_modified) > CACHE_TTL_SECONDS:
            return None

        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            return None

    @classmethod
    def write_to_cache(cls, lat: float, lon: float, data: Dict):
        try:
            path = cls.get_cache_path(lat, lon)
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Cache Write Error: {e}")

    @staticmethod
    async def fetch_api_data(lat: float, lon: float) -> Optional[Dict]:
        models_list = [
            "ecmwf_ifs04", "gfs_seamless", "jma_seamless",
            "icon_seamless", "gem_global", "meteofrance_seamless"
        ]

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation,temperature_2m,relative_humidity_700hPa,cloud_cover,freezing_level_height",
            "models": ",".join(models_list),
            "precipitation_unit": "inch",
            "timezone": "America/Denver"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"API returned {resp.status}: {text[:200]}")
                        return None
                    return await resp.json()
        except asyncio.TimeoutError:
            logger.error("API request timed out")
            return None
        except Exception as e:
            logger.error(f"Async API Error: {e}")
            return None

    @classmethod
    def process_physics(cls, data: Dict, elev_config: Dict[str, int], resort_name: str) -> pd.DataFrame:
        """
        The core physics logic that converts NWP data to band-specific snow forecasts.
        """
        try:
            if not data:
                logger.error(f"[{resort_name}] No data provided")
                return pd.DataFrame()

            hourly = data.get("hourly", {})

            if not hourly:
                logger.error(f"[{resort_name}] No hourly data in response")
                return pd.DataFrame()

            if "time" not in hourly or not hourly["time"]:
                logger.error(f"[{resort_name}] No time data in response")
                return pd.DataFrame()

            times = pd.to_datetime(hourly["time"])
            if len(times) == 0:
                logger.error(f"[{resort_name}] Empty time array")
                return pd.DataFrame()

            times = times.tz_localize("America/Denver", ambiguous='NaT', nonexistent='shift_forward')

            model_elev_m = data.get("elevation", 0)
            if model_elev_m is None:
                model_elev_m = 0
                logger.warning(f"[{resort_name}] No elevation in response, using 0")

            # Define Bands
            base_m = elev_config["base"] * 0.3048
            peak_m = elev_config["peak"] * 0.3048
            mid_m = (base_m + peak_m) / 2.0
            bands = {"Base": base_m, "Mid": mid_m, "Summit": peak_m}

            records = []

            for model_id, model_label in cls.MODEL_DISPLAY_MAP.items():
                p_key = f"precipitation_{model_id}"
                t_key = f"temperature_2m_{model_id}"
                rh_key = f"relative_humidity_700hPa_{model_id}"
                cloud_key = f"cloud_cover_{model_id}"
                fl_key = f"freezing_level_height_{model_id}"

                # Check if all required keys exist
                if all(k in hourly for k in [p_key, t_key, rh_key, cloud_key, fl_key]):
                    try:
                        # Get data and ensure it's not None
                        precip_raw = hourly[p_key]
                        temp_raw_c = hourly[t_key]
                        rh_700 = hourly[rh_key]
                        cloud_cover = hourly[cloud_key]
                        freezing_level = hourly[fl_key]

                        # Check if any are None or empty
                        if not all([precip_raw, temp_raw_c, rh_700, cloud_cover, freezing_level]):
                            logger.debug(f"[{resort_name}] Some data empty for {model_label}")
                            continue

                        # Convert to numpy arrays, replacing None with 0
                        precip_raw = np.array([x if x is not None else 0 for x in precip_raw], dtype=float)
                        temp_raw_c = np.array([x if x is not None else 0 for x in temp_raw_c], dtype=float)
                        rh_700 = np.array([x if x is not None else 0 for x in rh_700], dtype=float)
                        cloud_cover = np.array([x if x is not None else 0 for x in cloud_cover], dtype=float)
                        freezing_level = np.array([x if x is not None else 0 for x in freezing_level], dtype=float)

                        # Ensure same length as times
                        if len(precip_raw) != len(times):
                            logger.warning(f"[{resort_name}] Length mismatch for {model_label}")
                            continue

                        for band_name, target_elev_m in bands.items():
                            delta_z_m = target_elev_m - model_elev_m
                            temp_adj_c = temp_raw_c - (cls.LAPSE_RATE_C_PER_100M * (delta_z_m / 100.0))
                            lift_multiplier = 1.0 + cls.OROGRAPHIC_LIFT_FACTOR * (np.maximum(0, delta_z_m) / 100.0)
                            precip_adj = precip_raw * lift_multiplier

                            # Ensure no negative precipitation
                            precip_adj = np.maximum(0, precip_adj)

                            # Dynamic SLR calculations
                            is_rain = temp_adj_c > 1.0
                            is_wet_snow = (temp_adj_c <= 1.0) & (temp_adj_c > -3.0)
                            is_dgz_champagne = (temp_adj_c <= -12.0) & (temp_adj_c >= -18.0) & (rh_700 >= 80.0)

                            # Kuchera SLR approximation
                            kuchera_ratio = np.clip(12.0 + (-2.0 - temp_adj_c), 8.0, 30.0)

                            slr = np.select(
                                [is_rain, is_wet_snow, is_dgz_champagne],
                                [0.0, 8.0, 20.0],
                                default=kuchera_ratio
                            )

                            # Calculate snow amount (convert to inches)
                            snow_amount = precip_adj * slr

                            for i in range(len(times)):
                                records.append({
                                    "Date": times[i],
                                    "Model": model_label,
                                    "Band": band_name,
                                    "Amount": snow_amount[i],
                                    "Temp_C": temp_adj_c[i],
                                    "SLR": slr[i],
                                    "Cloud_Cover": cloud_cover[i],
                                    "Freezing_Level_m": freezing_level[i],
                                    "IsHistory": False,
                                    "Source": "NWP"
                                })
                    except Exception as e:
                        logger.error(f"[{resort_name}] Error processing {model_label}: {e}")
                        continue
                else:
                    missing = [k for k in [p_key, t_key, rh_key, cloud_key, fl_key] if k not in hourly]
                    logger.debug(f"[{resort_name}] Skipping {model_label} due to missing keys: {missing}")

            if not records:
                logger.warning(f"[{resort_name}] No valid model data processed")
                return pd.DataFrame()

            result = pd.DataFrame(records)
            logger.info(f"[{resort_name}] Processed {len(result)} rows")
            return result

        except Exception as e:
            logger.error(f"Physics Processing Error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    @classmethod
    async def get_forecast_async(cls, lat: float, lon: float, elev_config: Dict[str, int], resort_name: str = "Unknown") -> pd.DataFrame:
        """
        Async entry point with caching.
        """
        cache_data = cls.read_from_cache(lat, lon)
        if cache_data:
            logger.info(f"[{resort_name}] Cache Hit (Valid < 4hr)")
            data = cache_data
        else:
            logger.info(f"[{resort_name}] Cache Miss. Fetching API...")
            data = await cls.fetch_api_data(lat, lon)
            if data:
                cls.write_to_cache(lat, lon, data)

        if not data:
            logger.warning(f"[{resort_name}] No data returned from API/Cache.")
            return pd.DataFrame()

        return await asyncio.to_thread(cls.process_physics, data, elev_config, resort_name)

# =============================================================================
# END OF PointForecastEngine CLASS
# =============================================================================

def run_async_forecast(lat: float, lon: float, elev_config: Dict[str, int], resort_name: str) -> pd.DataFrame:
    """
    Synchronous wrapper for running async forecast in Streamlit.
    """
    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            PointForecastEngine.get_forecast_async(lat, lon, elev_config, resort_name)
        )
        return result
    except RuntimeError as e:
        # Handle case where loop is already closed or other runtime errors
        logger.error(f"Async error: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error running async forecast: {e}")
        return pd.DataFrame()
    finally:
        if loop:
            try:
                loop.close()
            except:
                pass

@st.cache_data(ttl=3600, show_spinner=False)
def get_nwp_forecast(lat: float, lon: float, elev_config: Dict[str, int], resort_name: str) -> pd.DataFrame:
    """
    Cached wrapper for NWP forecast.
    """
    return run_async_forecast(lat, lon, elev_config, resort_name)

# =============================================================================
# 5. DATA ENGINE (Existing functions)
# =============================================================================

def calculate_swe_ratio(temp_f):
    for threshold, ratio in SWE_RATIO_THRESHOLDS:
        if temp_f >= threshold:
            return ratio
    return DEFAULT_COLD_RATIO

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_noaa(lat, lon):
    retries = 3
    for attempt in range(retries):
        try:
            point_resp = requests.get(
                f"https://api.weather.gov/points/{lat},{lon}",
                headers=API_HEADERS, timeout=API_TIMEOUT
            )
            point_resp.raise_for_status()
            prop = point_resp.json().get("properties", {})
            forecast_url = prop.get("forecastHourly")
            if not forecast_url:
                raise ValueError("No forecast URL found")

            forecast_resp = requests.get(forecast_url, headers=API_HEADERS, timeout=API_TIMEOUT)
            forecast_resp.raise_for_status()
            periods = forecast_resp.json()["properties"]["periods"]

            elev_m = prop.get("relativeLocation", {}).get("properties", {}).get("elevation", {}).get("value", 0)
            elevation_ft = elev_m * 3.28084 if elev_m else 0

            data = []
            for p in periods:
                temp_f = p["temperature"]
                wind_str = str(p.get("windSpeed", "0"))
                wind_speed = 0
                if wind_str and wind_str[0].isdigit():
                    try:
                        wind_speed = int(wind_str.split()[0])
                    except:
                        wind_speed = 0
                data.append({
                    "Time": pd.to_datetime(p["startTime"]).tz_convert("America/Denver"),
                    "Temp": temp_f,
                    "Humidity": p["relativeHumidity"]["value"],
                    "Wind": wind_speed,
                    "SWE_Ratio": calculate_swe_ratio(temp_f),
                    "Summary": p["shortForecast"]
                })
            df = pd.DataFrame(data)
            return df, elevation_ft

        except Exception as e:
            logger.error(f"NOAA attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e

def get_noaa_forecast(lat, lon, retries=3):
    if "last_noaa" not in st.session_state:
        st.session_state["last_noaa"] = (pd.DataFrame(), 0)

    try:
        df, elev = _fetch_noaa(lat, lon)
        st.session_state["last_noaa"] = (df, elev)
        return df, elev
    except Exception as e:
        logger.error(f"NOAA fetch failed: {e}")
        st.warning("Using previously cached NOAA data (live unavailable)")
        return st.session_state["last_noaa"]

def _fetch_single_snotel(sid, retries=3):
    primary_url = (
        "https://wcc.sc.egov.usda.gov/reportGenerator/view/"
        "customSingleStationReport/daily/{}:CO:SNTL|id=%22%22|"
        "name/-29,0/WTEQ::value,WTEQ::median_1991,WTEQ::pctOfMedian_1991,"
        "SNWD::value,PREC::value,PREC::median_1991,PREC::pctOfMedian_1991,"
        "TMAX::value,TMIN::value,TAVG::value?fitToScreen=false"
    ).format(sid)
    fallback_url = (
        "https://wcc.sc.egov.usda.gov/reportGenerator/view/"
        "customSingleStationReport/daily/{}:CO:SNTL|id=%22%22|"
        "name/-30,0/WTEQ::value,SNWD::value,TAVG::value"
    ).format(sid)

    with requests.Session() as session:
        for attempt in range(retries):
            for url in [primary_url, fallback_url]:
                try:
                    resp = session.get(url, timeout=API_TIMEOUT, headers=API_HEADERS)
                    resp.raise_for_status()
                    tables = pd.read_html(StringIO(resp.text))

                    df = None
                    for tbl in tables:
                        cols = [str(c).upper() for c in tbl.columns]
                        if any('DATE' in c for c in cols) and any('WTEQ' in c or 'WATER' in c for c in cols):
                            df = tbl
                            break
                    if df is None:
                        df = next((t for t in tables if not t.empty), pd.DataFrame())
                    if df.empty:
                        continue

                    def find_column(df, possible_names, exclude_median_pct=True):
                        for col in df.columns:
                            col_upper = str(col).upper()
                            for name in possible_names:
                                if name.upper() in col_upper:
                                    if exclude_median_pct and ('MEDIAN' in col_upper or 'PCT' in col_upper):
                                        continue
                                    return col
                        return None

                    date_col = find_column(df, ['Date'])
                    if date_col is None:
                        continue
                    df.rename(columns={date_col: 'Date'}, inplace=True)

                    swe_col = find_column(df, ['WTEQ', 'Snow Water Equivalent', 'SWE'], exclude_median_pct=True)
                    if swe_col is None:
                        continue
                    df.rename(columns={swe_col: 'SWE'}, inplace=True)

                    depth_col = find_column(df, ['SNWD', 'Snow Depth'], exclude_median_pct=False)
                    if depth_col:
                        df.rename(columns={depth_col: 'Depth'}, inplace=True)

                    temp_col = find_column(df, ['TAVG', 'Temperature Average'], exclude_median_pct=False)
                    if temp_col:
                        df.rename(columns={temp_col: 'Temp'}, inplace=True)

                    if 'Date' not in df.columns or 'SWE' not in df.columns:
                        continue

                    df["Date"] = pd.to_datetime(df["Date"])
                    df["SWE_Delta"] = df["SWE"].diff().clip(lower=0)
                    df["SiteID"] = str(sid)
                    return df

                except Exception as e:
                    logger.warning(f"SNOTEL {sid} failed: {e}")
                    continue

        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return pd.DataFrame()

@st.cache_data(ttl=86400, show_spinner=False)
def get_snotel_data(site_ids):
    combined = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(_fetch_single_snotel, site_ids))
    for df in results:
        if not df.empty:
            combined.append(df)
    if not combined:
        return pd.DataFrame()
    res = pd.concat(combined)
    res["Date"] = res["Date"].dt.strftime("%Y-%m-%d")
    return res

# =============================================================================
# 6. PARSER (Snowiest)
# =============================================================================

def _calculate_date(day_val, is_history):
    now = datetime.now()
    try:
        day_num = int(day_val)
        month, year = now.month, now.year
        if not is_history and day_num < now.day:
            month = (month % 12) + 1
            if month == 1:
                year += 1
        elif is_history and day_num > now.day + 5:
            month = month - 1 if month > 1 else 12
            if month == 12:
                year -= 1
        return f"{year}-{month:02d}-{day_num:02d}"
    except:
        return None

def _parse_model_line(line, headers, is_history, model_totals):
    parts = line.split('\t')
    if len(parts) < 2:
        return []
    model_name = parts[0].strip()
    keywords = ["ECMWF IFS 025", "ECMWF", "GFS", "GEM", "JMA", "ICON", "MeteoFrance", "Average"]
    if not any(kw in model_name for kw in keywords):
        return []

    values = parts[1:]
    numbers = []
    for v in values:
        v = v.strip().replace('"', '')
        if v in ('-', '.', ''):
            numbers.append(None)
        else:
            try:
                numbers.append(float(v))
            except:
                numbers.append(None)

    observations = []
    daily_sum = 0.0
    for i, day_val in enumerate(headers):
        if i >= len(numbers):
            break
        amount = numbers[i] if numbers[i] is not None else 0.0
        daily_sum += amount
        date_str = _calculate_date(day_val, is_history)
        if date_str:
            # Create naive datetime at midnight and then localize
            dt = pd.Timestamp(date_str).tz_localize(None)
            dt = dt.tz_localize("America/Denver", ambiguous='NaT', nonexistent='shift_forward')
            observations.append({
                "Date": dt,
                "Amount": amount,
                "Model": model_name,
                "IsHistory": is_history,
                "Source": "Snowiest"
            })

    if len(numbers) > len(headers):
        total_val = numbers[len(headers)]
        if total_val is not None:
            key = f"{model_name} ({'H' if is_history else 'F'})"
            model_totals[key] = {"declared": total_val, "calculated": daily_sum}

    return observations

def parse_snowiest_raw_text(text):
    if not text:
        return pd.DataFrame(), {}
    obs = []
    totals = {}
    sections = re.split(r"(Snowfall Prediction History)", text)

    f_lines = sections[0].split("\n")
    f_headers = []
    for line in f_lines:
        m = re.findall(r"(\d+)\s+(Sun|Mon|Tue|Wed|Thu|Fri|Sat)", line)
        if m:
            f_headers = [x[0] for x in m]
            break
    if f_headers:
        for line in f_lines:
            obs.extend(_parse_model_line(line, f_headers, False, totals))

    if len(sections) > 1:
        h_lines = sections[1].split("\n")
        h_headers = []
        for line in h_lines:
            m = re.findall(r"(\d+)\s+(Sun|Mon|Tue|Wed|Thu|Fri|Sat)", line)
            if m:
                h_headers = [x[0] for x in m]
                break
        if h_headers:
            for line in h_lines:
                obs.extend(_parse_model_line(line, h_headers, True, totals))

    if not obs:
        return pd.DataFrame(), {}
    df = pd.DataFrame(obs).dropna()
    return df, totals

# =============================================================================
# 7. ANALYTICS ENGINE (Enhanced for multi-source)
# =============================================================================

def calculate_forecast_metrics(df_models, selected_models, current_depth=0, band_filter="Summit"):
    if df_models.empty or not selected_models:
        return None

    filtered = df_models[
        (df_models["Model"].isin(selected_models)) &
        (df_models["Band"] == band_filter)
    ] if "Band" in df_models.columns else df_models[df_models["Model"].isin(selected_models)]

    # Get the timezone from the data (if available)
    if not filtered.empty and hasattr(filtered["Date"].dt, "tz") and filtered["Date"].dt.tz is not None:
        tz = filtered["Date"].dt.tz
    else:
        tz = pytz.timezone('America/Denver')
    now = pd.Timestamp.now(tz=tz).normalize()
    future = filtered[filtered["Date"] >= now]

    if future.empty:
        return None

    # Group by date and calculate statistics
    daily_stats = future.groupby("Date")["Amount"].agg(["mean", "min", "max", "std"]).reset_index()
    daily_stats = daily_stats.sort_values("Date").reset_index(drop=True)
    daily_stats["cumulative"] = daily_stats["mean"].cumsum()
    daily_stats["total_depth"] = current_depth + daily_stats["cumulative"]
    daily_stats["spread"] = daily_stats["max"] - daily_stats["min"]
    daily_stats["mean_48h"] = daily_stats["mean"].rolling(window=2, min_periods=1).sum()

    best_24h_idx  = daily_stats["mean"].idxmax()
    best_24h_val  = daily_stats.loc[best_24h_idx, "mean"]
    best_24h_date = daily_stats.loc[best_24h_idx, "Date"]

    best_48h_idx  = daily_stats["mean_48h"].idxmax()
    best_48h_val  = daily_stats.loc[best_48h_idx, "mean_48h"]
    best_48h_date = daily_stats.loc[best_48h_idx, "Date"]

    daily_stats["is_heavy"] = daily_stats["mean"] >= STORM_THRESHOLD_IN
    daily_stats["storm_group"] = (daily_stats["is_heavy"] != daily_stats["is_heavy"].shift()).cumsum()
    storm_groups = daily_stats[daily_stats["is_heavy"]].groupby("storm_group")

    major_storms = []
    for _, group in storm_groups:
        if len(group) >= 3:
            major_storms.append({
                "start": group["Date"].iloc[0],
                "end":   group["Date"].iloc[-1],
                "total": group["mean"].sum(),
                "days":  len(group)
            })
    major_storms.sort(key=lambda x: x["total"], reverse=True)

    storms = []
    in_storm, storm_start, storm_total = False, None, 0.0
    for _, row in daily_stats.iterrows():
        if row["mean"] >= STORM_THRESHOLD_IN and not in_storm:
            in_storm, storm_start, storm_total = True, row["Date"], row["mean"]
        elif row["mean"] >= STORM_THRESHOLD_IN and in_storm:
            storm_total += row["mean"]
        elif row["mean"] < STORM_THRESHOLD_IN and in_storm:
            storms.append({"start": storm_start, "total": storm_total})
            in_storm, storm_total = False, 0.0
    if in_storm:
        storms.append({"start": storm_start, "total": storm_total})

    return {
        "best_24h_date":   best_24h_date,
        "best_24h_amount": best_24h_val,
        "best_48h_date":   best_48h_date,
        "best_48h_amount": best_48h_val,
        "major_storms":    major_storms,
        "deepest_total":   daily_stats["total_depth"].max(),
        "total_snowfall":  daily_stats["cumulative"].iloc[-1],
        "avg_spread":      daily_stats["spread"].mean(),
        "storms":          storms,
        "daily_stats":     daily_stats,
    }

def build_forecast_figure(stats, noaa_df, show_spaghetti, show_ribbon,
                           df_models, selected_models, noaa_cutoff_dt, show_extended,
                           band_filter="Summit"):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.06,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    if show_ribbon and not stats.empty:
        x_ribbon = pd.concat([stats["Date"], stats["Date"][::-1]])
        y_ribbon = pd.concat([stats["max"], stats["min"][::-1]])
        fig.add_trace(go.Scatter(
            x=x_ribbon, y=y_ribbon,
            fill="toself", fillcolor="rgba(56,189,248,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Uncertainty Range", hoverinfo="skip"
        ), row=1, col=1, secondary_y=False)

    if show_spaghetti and not df_models.empty:
        filtered = df_models[
            (df_models["Model"].isin(selected_models)) &
            (df_models["Band"] == band_filter)
        ] if "Band" in df_models.columns else df_models[df_models["Model"].isin(selected_models)]

        if not df_models.empty and "Date" in df_models.columns and hasattr(df_models["Date"].dt, "tz") and df_models["Date"].dt.tz is not None:
            tz = df_models["Date"].dt.tz
        else:
            tz = pytz.timezone('America/Denver')
        now_dt = pd.Timestamp.now(tz=tz).normalize()
        colors = ["rgba(251,113,133,0.55)", "rgba(251,191,36,0.55)",
                  "rgba(45,212,191,0.55)", "rgba(196,181,253,0.55)",
                  "rgba(134,239,172,0.55)", "rgba(249,168,212,0.55)"]
        for i, mdl in enumerate(selected_models):
            m_df = filtered[
                (filtered["Model"] == mdl) &
                (filtered["Date"] >= now_dt)
            ]
            if not m_df.empty:
                fig.add_trace(go.Scatter(
                    x=m_df["Date"], y=m_df["Amount"].fillna(0),
                    mode="lines",
                    line=dict(width=1.5, color=colors[i % len(colors)]),
                    name=mdl, opacity=0.75,
                    hovertemplate=f"<b>{mdl}</b>: %{{y:.1f}}\"<extra></extra>"
                ), row=1, col=1, secondary_y=False)

    if not stats.empty:
        max_val = stats["mean"].max() or 0.1
        bar_colors = [
            f"rgba(56,189,248,{min(0.28 + v / max_val * 0.68, 0.95):.2f})"
            for v in stats["mean"]
        ]
        fig.add_trace(go.Bar(
            x=stats["Date"], y=stats["mean"],
            name="Daily Average",
            marker=dict(color=bar_colors, line=dict(width=0)),
            hovertemplate="<b>%{x|%a %b %d}</b><br>Mean: <b>%{y:.2f}\"</b><extra></extra>"
        ), row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=stats["Date"], y=stats["cumulative"],
            name="Cumulative",
            line=dict(color=ACCENT_ROSE, width=2.5),
            fill="tozeroy", fillcolor="rgba(251,113,133,0.06)",
            hovertemplate="Cumulative: <b>%{y:.1f}\"</b><extra></extra>"
        ), row=1, col=1, secondary_y=True)

    if show_extended and noaa_cutoff_dt is not None:
        fig.add_vline(
            x=noaa_cutoff_dt,
            line=dict(color="rgba(251,191,36,0.35)", width=1.5, dash="dot"),
        )
        fig.add_annotation(
            x=noaa_cutoff_dt, y=0.97, yref="paper",
            text="NOAA end", showarrow=False,
            font=dict(size=9, color="rgba(251,191,36,0.6)"),
            bgcolor="rgba(15,23,42,0.7)" if is_dark else "rgba(255,255,255,0.85)",
            borderpad=3, xanchor="center"
        )

    if not noaa_df.empty:
        w_end = stats["Date"].max() + timedelta(days=1) if not stats.empty else datetime.now() + timedelta(days=8)
        w_window = noaa_df[noaa_df["Time"] <= w_end]
        if not w_window.empty:
            fig.add_trace(go.Scatter(
                x=w_window["Time"], y=w_window["Temp"],
                name="Temp °F",
                line=dict(color=TEXT_SEC, width=1.5),
                hovertemplate="%{y:.0f}°F<extra>Temp</extra>"
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=w_window["Time"], y=w_window["Humidity"],
                name="Humidity %",
                line=dict(color=ACCENT_TEAL, width=1.5, dash="dot"),
                hovertemplate="%{y:.0f}%<extra>Humidity</extra>"
            ), row=2, col=1)
            fig.add_hline(y=32, line=dict(color="rgba(56,189,248,0.22)", dash="dash", width=1), row=2, col=1)

    layout_update = PLOTLY_TEMPLATE["layout"].to_plotly_json()
    layout_update.update(dict(
        height=480, bargap=0.3,
        yaxis=dict(title="Snow (in)", gridcolor=PLOTLY_GRID, zeroline=False, showline=False),
        yaxis2=dict(
            overlaying="y", side="right", showgrid=False, zeroline=False,
            title="Cumul (in)",
            tickfont=dict(color="rgba(251,113,133,0.6)"),
            title_font=dict(color="rgba(251,113,133,0.6)")
        ),
        yaxis3=dict(title="°F / %", gridcolor=PLOTLY_GRID, zeroline=False, showline=False)
    ))
    fig.update_layout(layout_update)
    return fig

# =============================================================================
# 8. SESSION STATE & PARSE HANDLING
# =============================================================================
if "raw_model_data" not in st.session_state:
    st.session_state["raw_model_data"] = pd.DataFrame()
if "model_totals" not in st.session_state:
    st.session_state["model_totals"] = {}
if "nwp_data" not in st.session_state:
    st.session_state["nwp_data"] = pd.DataFrame()
if "use_nwp" not in st.session_state:
    st.session_state["use_nwp"] = True
if "last_noaa" not in st.session_state:
    st.session_state["last_noaa"] = (pd.DataFrame(), 0)

if from_paste and paste_input.strip():
    df_p, tots_p = parse_snowiest_raw_text(paste_input)
    if not df_p.empty:
        st.session_state["raw_model_data"] = df_p
        st.session_state["model_totals"] = tots_p
        st.sidebar.success(f"Parsed {len(df_p)} observations")
    else:
        st.sidebar.error("Parse failed")

# =============================================================================
# 9. RESORT SELECTOR
# =============================================================================
top_left, top_right = st.columns([3, 1])
with top_left:
    selected_loc = st.selectbox(
        "Where are we going?",
        list(RESORTS.keys()),
        index=0,
        label_visibility="visible",
    )
with top_right:
    st.write("")
    st.write("")
    if st.button("＋ Add Spot"):
        st.toast("Coming in v2.0 🚧")

conf = RESORTS[selected_loc]

with st.spinner(f"Syncing {selected_loc}…"):
    noaa_df, grid_elev = get_noaa_forecast(conf["lat"], conf["lon"])
    snotel_df = get_snotel_data(conf["snotel_ids"])

    # Fetch NWP data if enabled
    if st.session_state["use_nwp"]:
        with st.spinner("Fetching live NWP ensemble..."):
            nwp_df = get_nwp_forecast(conf["lat"], conf["lon"], conf["elev_ft"], selected_loc)
            st.session_state["nwp_data"] = nwp_df
    else:
        st.session_state["nwp_data"] = pd.DataFrame()

    if st.session_state["raw_model_data"].empty:
        demo_df, demo_tots = parse_snowiest_raw_text(DEMO_DATA)
        st.session_state["raw_model_data"] = demo_df
        st.session_state["model_totals"] = demo_tots

# After getting NOAA forecast, add this debug section
if st.session_state["use_nwp"]:
    with st.expander("🔧 NWP Debug Info", expanded=False):
        debug_data = debug_nwp_api(conf["lat"], conf["lon"])
        if debug_data and "hourly" in debug_data:
            st.write("✅ NWP API Connected")
            st.write(f"Hours: {len(debug_data['hourly'].get('time', []))}")
            st.json({
                "elevation": debug_data.get("elevation"),
                "hourly_keys": list(debug_data["hourly"].keys())[:10],
                "sample_time": debug_data["hourly"].get("time", [])[:3] if "time" in debug_data["hourly"] else None
            })

# Combine data sources
df_snowiest = st.session_state["raw_model_data"]
df_nwp = st.session_state["nwp_data"]

# Merge dataframes if both exist
if not df_nwp.empty and not df_snowiest.empty:
    df_models = pd.concat([df_snowiest, df_nwp], ignore_index=True)
elif not df_nwp.empty:
    df_models = df_nwp
else:
    df_models = df_snowiest

current_swe, current_depth = 0.0, 0.0
if not snotel_df.empty:
    snotel_df["Date"] = pd.to_datetime(snotel_df["Date"])
    latest = snotel_df.sort_values("Date").dropna(subset=["SWE"]).iloc[-1] if not snotel_df.empty else None
    if latest is not None:
        current_depth = float(latest.get("Depth", 0) or 0)
        current_swe   = float(latest.get("SWE", 0) or 0)

# =============================================================================
# 10. HERO SECTION
# =============================================================================
st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
st.markdown("---")
st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

# Band selector for NWP data
band_options = ["Summit", "Mid", "Base"] if "Band" in df_models.columns else ["Summit"]
selected_band = "Summit"
if "Band" in df_models.columns:
    selected_band = st.selectbox(
        "Elevation Band",
        options=band_options,
        index=0,
        help="Select elevation band for forecast (NWP only)"
    )

metrics = None
if not df_models.empty:
    all_models = list(df_models["Model"].unique())
    metrics = calculate_forecast_metrics(df_models, all_models, current_depth, selected_band)

hero_val   = "—"
hero_label = "NO DATA"
hero_sub   = "Load demo data, paste Snowiest table, or enable NWP"

if metrics:
    if metrics["major_storms"]:
        s = metrics["major_storms"][0]
        hero_val   = f"{s['total']:.1f}\""
        hero_label = "STORM CYCLE DETECTED"
        hero_sub   = f"{s['days']} days  ·  Begins {s['start'].strftime('%a %b %d')}"
    elif metrics["best_48h_amount"] > (metrics["best_24h_amount"] * 1.5) and metrics["best_48h_amount"] > 6:
        hero_val   = f"{metrics['best_48h_amount']:.1f}\""
        hero_label = "48H PEAK"
        hero_sub   = f"Around {metrics['best_48h_date'].strftime('%A, %b %d')}"
    elif metrics["best_24h_amount"] > 4:
        hero_val   = f"{metrics['best_24h_amount']:.1f}\""
        hero_label = "BEST POWDER DAY"
        hero_sub   = metrics["best_24h_date"].strftime("%A, %b %d")
    else:
        hero_val   = f"{metrics['total_snowfall']:.1f}\""
        hero_label = "TOTAL FORECAST"
        hero_sub   = f"Next {len(metrics['daily_stats'])} days"

h_left, h_right = st.columns([5, 3], gap="large")

with h_left:
    st.markdown(f"""
    <div class="fade-up fade-up-1" style="padding-bottom:1.5rem;">
        <div class="hero-label">{hero_label}</div>
        <div class="hero-stat">{hero_val}</div>
        <div class="hero-sub">{hero_sub}</div>
    </div>
    """, unsafe_allow_html=True)

with h_right:
    live_slr = calculate_swe_ratio(noaa_df.iloc[0]["Temp"]) if not noaa_df.empty else 0
    agree_color = ACCENT_TEAL if (metrics and metrics["avg_spread"] < 1.5) else \
                  ACCENT_AMBER if (metrics and metrics["avg_spread"] < 3) else ACCENT_ROSE
    spread_str  = f"{metrics['avg_spread']:.1f}\"" if metrics else "—"

    st.markdown(
        f'<div class="glass-card glass-card-accent fade-up fade-up-2" style="margin-bottom:0;">',
        unsafe_allow_html=True
    )
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Base Depth", f'{current_depth:.0f}"')
    with m2:
        st.metric("SWE", f'{current_swe:.2f}"')
    m3, m4 = st.columns(2)
    with m3:
        st.metric("Live SLR", f"{live_slr}:1")
    with m4:
        st.metric("Model Spread", spread_str)

    # Show data source indicators
    source_indicators = []
    if not df_snowiest.empty:
        source_indicators.append('<span class="pill pill-blue">Snowiest</span>')
    if not df_nwp.empty:
        source_indicators.append('<span class="pill pill-teal">NWP Live</span>')

    st.markdown(
        f'<div style="margin-top:0.5rem; display: flex; gap: 0.5rem;">{"".join(source_indicators)}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div style="margin-top:0.5rem; font-size:0.75rem;">'
        f'<span class="live-dot"></span>'
        f'<span style="color:{ACCENT_TEAL};">Live Telemetry · {conf["elevation"]}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# 11. ATMOSPHERIC OUTLOOK
# =============================================================================
st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label fade-up fade-up-3">Atmospheric Outlook</div>', unsafe_allow_html=True)

if not noaa_df.empty:
    now_naive = datetime.now()
    shorts = noaa_df[noaa_df["Time"] > now_naive].head(96).copy()

    if not shorts.empty:
        shorts["Date"] = shorts["Time"].dt.date
        daily_noaa = shorts.groupby("Date").agg(
            Min  =("Temp",     "min"),
            Max  =("Temp",     "max"),
            Cond =("Summary",  lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]),
            Wind =("Wind",     "max"),
            SLR  =("SWE_Ratio","mean")
        ).reset_index().head(4)

        cards_html = '<div class="outlook-strip fade-up fade-up-4">'
        for _, row in daily_noaa.iterrows():
            d_str    = row["Date"].strftime("%a %d")
            is_snow  = "Snow" in str(row["Cond"])
            pill_cls = "pill-blue" if is_snow else "pill-rose"
            cond_str = str(row["Cond"])[:18]
            slr_str  = f"  ·  SLR {row['SLR']:.0f}:1" if is_snow else ""
            cards_html += f"""
            <div class="outlook-card">
                <div style="font-size:0.72rem; letter-spacing:0.06em; text-transform:uppercase;
                            color:var(--text-muted); margin-bottom:0.4rem;">{d_str}</div>
                <div style="font-size:1.25rem; font-weight:700; margin-bottom:0.5rem;
                            color:var(--text-pri);">{row['Max']:.0f}° / {row['Min']:.0f}°</div>
                <span class="pill {pill_cls}">{cond_str}</span>
                <div style="font-size:0.7rem; color:var(--text-muted); margin-top:0.5rem;">
                    💨 {row['Wind']:.0f} mph{slr_str}
                </div>
            </div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)
else:
    st.caption("NOAA data unavailable")

# =============================================================================
# 12. ENSEMBLE ANALYSIS (Enhanced with NWP)
# =============================================================================
st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label fade-up fade-up-5">Ensemble Analysis</div>', unsafe_allow_html=True)

if not df_models.empty and metrics:
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2.5, 1, 1, 1], gap="small")
    with ctrl1:
        u_models = sorted(df_models["Model"].unique())
        default_models = [m for m in u_models if "Average" not in m]
        selected_models = st.multiselect(
            "Active Models", u_models, default=default_models[:4], label_visibility="collapsed"
        )
    with ctrl2:
        show_ribbon    = st.toggle("Ribbon",    value=True)
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

    if selected_models:
        metrics_sub = calculate_forecast_metrics(df_models, selected_models, current_depth, selected_band)

        if metrics_sub:
            daily_stats = metrics_sub["daily_stats"]
            noaa_cutoff = noaa_df["Time"].max() if not noaa_df.empty else None

            if not show_extended:
                cutoff = datetime.now() + timedelta(days=7)
                daily_stats = daily_stats[daily_stats["Date"] <= cutoff].copy()

            if len(daily_stats) > 1:
                date_min = daily_stats["Date"].min().date()
                date_max = daily_stats["Date"].max().date()
                slider_range = st.select_slider(
                    "Storm Window",
                    options=pd.date_range(date_min, date_max).date.tolist(),
                    value=(date_min, date_max),
                    format_func=lambda d: d.strftime("%b %d"),
                )
                w_start  = pd.Timestamp(slider_range[0])
                w_end    = pd.Timestamp(slider_range[1])
                windowed = daily_stats[
                    (daily_stats["Date"] >= w_start) & (daily_stats["Date"] <= w_end)
                ].copy()
                windowed["cumulative"] = windowed["mean"].cumsum()
                window_total = windowed["mean"].sum()
            else:
                windowed     = daily_stats.copy()
                window_total = daily_stats["mean"].sum()

            sm1, sm2, sm3, sm4 = st.columns(4)
            with sm1:
                st.metric("Window Total", f'{window_total:.1f}"')
            with sm2:
                st.metric("Best Day", f'{metrics_sub["best_24h_amount"]:.1f}"',
                          delta=metrics_sub["best_24h_date"].strftime("%a %b %d"))
            with sm3:
                st.metric("Models Active", len(selected_models))
            with sm4:
                st.metric("Avg Spread", f'{metrics_sub["avg_spread"]:.1f}"')

            st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            fig = build_forecast_figure(
                stats=windowed, noaa_df=noaa_df,
                show_spaghetti=show_spaghetti, show_ribbon=show_ribbon,
                df_models=df_models, selected_models=selected_models,
                noaa_cutoff_dt=noaa_cutoff, show_extended=show_extended,
                band_filter=selected_band
            )
            st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

    totals = st.session_state["model_totals"]
    if totals:
        with st.expander("Snowiest Model Integrity Check"):
            check = [
                {
                    "Model": k, "Table": v["declared"],
                    "Calc": round(v["calculated"], 1),
                    "Δ":    round(v["declared"] - v["calculated"], 1),
                    "OK":   "✅" if abs(v["declared"] - v["calculated"]) < 0.5 else "⚠️"
                }
                for k, v in totals.items()
            ]
            st.dataframe(pd.DataFrame(check), hide_index=True, width='stretch')

    # NWP Physics Details
    if not df_nwp.empty and "Band" in df_nwp.columns:
        with st.expander("NWP Physics Details"):
            st.markdown("""
            **Physics Engine Parameters:**
            - Lapse Rate: 0.65°C per 100m
            - Orographic Lift Factor: 0.05 per 100m
            - Kuchera SLR: 12 + (-2 - temp_c)
            - DGZ Champaign: 18:1 SLR (Temp -12°C to -18°C, RH >80%)
            """)

            latest_nwp = df_nwp[df_nwp["Band"] == selected_band].groupby("Model").last().reset_index()
            if not latest_nwp.empty:
                st.dataframe(
                    latest_nwp[["Model", "Temp_C", "SLR", "Cloud_Cover", "Freezing_Level_m"]].round(1),
                    hide_index=True,
                    width='stretch'
                )

else:
    st.markdown(f"""
    <div style="padding:3rem 2rem; border:1px dashed {BORDER}; border-radius:14px;
                text-align:center; color:var(--text-muted);">
        <div style="font-size:1.5rem; margin-bottom:0.5rem;">❄</div>
        <div style="font-size:0.9rem;">No ensemble data — enable NWP in sidebar or paste a Snowiest.app table</div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# 13. UPDATE MODEL DATA
# =============================================================================
st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
with st.expander("Update Model Data"):
    col1, col2 = st.columns(2)
    with col1:
        st.link_button(
            f"↗ Snowiest — {selected_loc}",
            conf["snowiest_url"],
            help="Opens Snowiest.app. Copy the table and paste below."
        )
    with col2:
        if st.button("🔄 Refresh NWP Data", help="Clear cache and fetch fresh NWP data"):
            # Clear cache for this location
            cache_path = PointForecastEngine.get_cache_path(conf["lat"], conf["lon"])
            if cache_path.exists():
                cache_path.unlink()
            st.cache_data.clear()
            st.rerun()

    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
    inline_paste = st.text_area(
        "Or paste Snowiest table here", height=100, label_visibility="collapsed",
        placeholder="Paste Snowiest table…"
    )
    if st.button("Parse", key="inline_parse"):
        df_p, tots_p = parse_snowiest_raw_text(inline_paste)
        if not df_p.empty:
            st.session_state["raw_model_data"] = df_p
            st.session_state["model_totals"] = tots_p
            st.rerun()
        else:
            st.error("Parse failed — check format")

# =============================================================================
# 14. SNOTEL VERIFICATION
# =============================================================================
st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">SNOTEL Verification</div>', unsafe_allow_html=True)

if not snotel_df.empty:
    sc1, sc2 = st.columns([1, 3], gap="large")

    with sc1:
        st.metric("Current Depth", f'{current_depth:.0f}"')
        st.metric("SWE Total", f'{current_swe:.2f}"')
        elev_display = f"{grid_elev:,.0f} ft" if grid_elev else conf["elevation"]
        st.metric("Grid Elevation", elev_display)

    with sc2:
        recent = snotel_df.groupby("Date").mean(numeric_only=True).reset_index().tail(21)
        fig_s = make_subplots(specs=[[{"secondary_y": True}]])

        if "SWE_Delta" in recent.columns:
            max_delta = recent["SWE_Delta"].max() or 0.01
            bar_cols = [
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
        st.plotly_chart(fig_s, width='stretch')

    with st.expander("SNOTEL Raw Records"):
        show_cols = ["Date", "SiteID", "SWE", "SWE_Delta"]
        if "Depth" in snotel_df.columns:
            show_cols.append("Depth")
        if "Temp" in snotel_df.columns:
            show_cols.append("Temp")
        disp = snotel_df[show_cols].sort_values("Date", ascending=False).reset_index(drop=True)
        st.dataframe(disp, hide_index=True, width='stretch', height=240)

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
        st.dataframe(snotel_df.tail(10)[show_cols], width='stretch', hide_index=True)
    else:
        st.info("No SNOTEL data")

with t2:
    st.caption("Ensemble Daily Statistics")
    if metrics and "daily_stats" in metrics:
        disp_cols = [c for c in ["Date", "mean", "min", "max", "cumulative", "std"]
                     if c in metrics["daily_stats"].columns]
        st.dataframe(
            metrics["daily_stats"][disp_cols].round(2),
            width='stretch', hide_index=True
        )
    else:
        st.info("No forecast data")

with t3:
    st.caption("NWP Live Data")
    if not df_nwp.empty and "Band" in df_nwp.columns:
        latest_nwp = df_nwp[df_nwp["Band"] == "Summit"].groupby("Model").last().reset_index()
        st.dataframe(
            latest_nwp[["Model", "Amount", "Temp_C", "SLR"]].round(1),
            width='stretch', hide_index=True
        )
    else:
        st.info("No NWP data (enable in sidebar)")

st.markdown('<div style="height:3rem;"></div>', unsafe_allow_html=True)
