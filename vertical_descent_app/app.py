import logging
import asyncio
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone

from data_layer.manager import DataManager
from data_layer.config import RESORTS
from data_layer.models import VariableType, DataQuality

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
        '<p style="font-size:0.65rem; opacity:0.3; text-align:center;">Summit Terminal · v5 · DAL Integrated</p>',
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
    color: {METRIC_LABEL_COLOR} !important;
    font-weight: 600 !important;
}}
[data-testid="stMetricValue"] > div {{
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: {METRIC_VAL_COLOR} !important;
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

STORM_THRESHOLD_IN = 2.0

# =============================================================================
# 4. ANALYTICS ENGINE (Enhanced for multi-source)
# =============================================================================

def calculate_forecast_metrics(df_models, selected_models, current_depth=0):
    if df_models.empty or not selected_models:
        return None

    # Make a copy to avoid altering the original DataFrame
    df = df_models.copy()

    filtered = df[df["Model"].isin(selected_models)]

    # Use the timezone from the data
    tz = filtered["Date"].dt.tz if not filtered.empty else None
    if tz is None:
        # Default to Denver if no TZ
        filtered["Date"] = filtered["Date"].dt.tz_localize("America/Denver", ambiguous='NaT', nonexistent='shift_forward')
        tz = filtered["Date"].dt.tz

    now = pd.Timestamp.now(tz=tz).normalize()  # midnight today
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

def build_forecast_figure(stats, show_spaghetti, show_ribbon,
                           df_models, selected_models, show_extended):
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
        df_local = df_models.copy()

        filtered = df_local[df_local["Model"].isin(selected_models)]

        colors = ["rgba(251,113,133,0.55)", "rgba(251,191,36,0.55)",
                  "rgba(45,212,191,0.55)", "rgba(196,181,253,0.55)",
                  "rgba(134,239,172,0.55)", "rgba(249,168,212,0.55)"]
        for i, mdl in enumerate(selected_models):
            m_df = filtered[filtered["Model"] == mdl]
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
# 5. RESORT SELECTOR & DATA LOADING
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

# Initialize DataManager
manager = DataManager()

# Run Data Fetching via Asyncio
@st.cache_data(ttl=60) # Short cache on UI side, let DAL handle SWR
def fetch_data_via_dal(lat, lon):
    start = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=5) # 5 days history
    end = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=10) # 10 days forecast

    try:
        response = asyncio.run(manager.get_forecast(lat, lon, start, end))
        return response
    except Exception as e:
        logger.error(f"DAL Error: {e}")
        return None

with st.spinner(f"Syncing {selected_loc}…"):
    response = fetch_data_via_dal(conf["lat"], conf["lon"])

# Process Response into DataFrames for UI
df_models = pd.DataFrame()
snotel_df = pd.DataFrame()
current_depth = 0.0
current_swe = 0.0

if response and response.points:
    # Convert UnifiedDataPoints to DataFrame
    data = []
    for p in response.points:
        data.append({
            "Date": p.timestamp_utc,
            "Variable": p.variable,
            "Value": p.value,
            "Source": p.source,
            "Quality": p.quality
        })

    raw_df = pd.DataFrame(data)

    # 1. Prepare df_models (Forecast Snow)
    # Filter for Forecast quality and Precp Snow variable
    forecast_snow = raw_df[
        (raw_df["Variable"] == VariableType.PRECIP_SNOW) &
        (raw_df["Quality"].isin([DataQuality.FORECAST, DataQuality.FALLBACK]))
    ].copy()

    if not forecast_snow.empty:
        forecast_snow.rename(columns={"Value": "Amount", "Source": "Model"}, inplace=True)
        # Normalize date to day if needed, or keep hourly?
        # UI expects 'Date' column.
        df_models = forecast_snow[["Date", "Model", "Amount"]]

    # 2. Prepare snotel_df (Measured Telemetry)
    # Filter for Measured quality
    measured = raw_df[
        (raw_df["Quality"] == DataQuality.MEASURED)
    ].copy()

    if not measured.empty:
        # Pivot to get SWE, Depth, Temp columns
        # Pivot table requires unique index/columns.
        # Create a 'Date' column that is just the date part for daily SNOTEL?
        # Or keep timestamp.
        # Snotel source is "SNOTEL_ID".
        # We might have multiple stations.
        # Let's take the first one or average?
        # For simplicity, we just use the dataframe as is, but we need columns SWE, Depth, Temp.

        # We need to pivot 'Variable' to columns.
        pivot_df = measured.pivot_table(index="Date", columns="Variable", values="Value", aggfunc='first').reset_index()

        # Rename columns
        # Columns might be: VariableType.SWE, VariableType.SNOW_DEPTH, etc.
        # We map them to string names
        col_map = {
            VariableType.SWE: "SWE",
            VariableType.SNOW_DEPTH: "Depth",
            VariableType.TEMP_AIR: "Temp"
        }
        pivot_df.rename(columns=col_map, inplace=True)

        # Calculate SWE Delta
        if "SWE" in pivot_df.columns:
            pivot_df["SWE_Delta"] = pivot_df["SWE"].diff().clip(lower=0)

        snotel_df = pivot_df

        # Get current depth/swe
        if not snotel_df.empty:
            latest = snotel_df.sort_values("Date").iloc[-1]
            current_depth = latest.get("Depth", 0.0)
            current_swe = latest.get("SWE", 0.0)

# =============================================================================
# 6. HERO SECTION
# =============================================================================
st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
st.markdown("---")
st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

metrics = None
if not df_models.empty:
    all_models = list(df_models["Model"].unique())
    metrics = calculate_forecast_metrics(df_models, all_models, current_depth)

hero_val   = "—"
hero_label = "NO DATA"
hero_sub   = "System offline or no forecast data"

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
        # Live SLR removed as logic is now handled by adapters or simplified
        st.metric("DAL Status", "ACTIVE")
    with m4:
        st.metric("Model Spread", spread_str)

    st.markdown(
        f'<div style="margin-top:0.5rem; font-size:0.75rem;">'
        f'<span class="live-dot"></span>'
        f'<span style="color:{ACCENT_TEAL};">Data Abstraction Layer · {conf["elevation"]}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# 7. ENSEMBLE ANALYSIS
# =============================================================================
st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label fade-up fade-up-5">Ensemble Analysis</div>', unsafe_allow_html=True)

if not df_models.empty and metrics:
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2.5, 1, 1, 1], gap="small")
    with ctrl1:
        u_models = sorted(df_models["Model"].unique())
        selected_models = st.multiselect(
            "Active Models", u_models, default=u_models, label_visibility="collapsed"
        )
    with ctrl2:
        show_ribbon    = st.toggle("Ribbon",    value=True)
    with ctrl3:
        show_spaghetti = st.toggle("Spaghetti", value=False)
    with ctrl4:
        show_extended  = st.toggle("Extended",  value=False)

    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

    if selected_models:
        metrics_sub = calculate_forecast_metrics(df_models, selected_models, current_depth)

        if metrics_sub:
            daily_stats = metrics_sub["daily_stats"]

            if not show_extended:
                cutoff = datetime.now().astimezone() + timedelta(days=7)
                daily_stats = daily_stats[daily_stats["Date"] <= cutoff].copy()

            sm1, sm2, sm3, sm4 = st.columns(4)
            with sm1:
                st.metric("Window Total", f'{metrics_sub["total_snowfall"]:.1f}"')
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
                stats=daily_stats,
                show_spaghetti=show_spaghetti, show_ribbon=show_ribbon,
                df_models=df_models, selected_models=selected_models,
                show_extended=show_extended
            )
            st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style="padding:3rem 2rem; border:1px dashed {BORDER}; border-radius:14px;
                text-align:center; color:var(--text-muted);">
        <div style="font-size:1.5rem; margin-bottom:0.5rem;">❄</div>
        <div style="font-size:0.9rem;">No forecast data available from DAL</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 8. SNOTEL VERIFICATION
# =============================================================================
st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">SNOTEL Verification</div>', unsafe_allow_html=True)

if not snotel_df.empty:
    sc1, sc2 = st.columns([1, 3], gap="large")

    with sc1:
        st.metric("Current Depth", f'{current_depth:.0f}"')
        st.metric("SWE Total", f'{current_swe:.2f}"')

    with sc2:
        recent = snotel_df.sort_values("Date").tail(21)
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
else:
    st.info("No SNOTEL data available")
