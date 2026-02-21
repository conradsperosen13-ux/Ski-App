import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz

ACCENT_ROSE  = "#fb7185"
ACCENT_TEAL  = "#2dd4bf"
TEXT_SEC      = "#a1a1aa"
PLOTLY_FONT   = "#71717a"
PLOTLY_GRID   = "rgba(255,255,255,0.05)"
PLOTLY_HOVER  = "#18181b"
BORDER        = "rgba(255, 255, 255, 0.07)"
TEXT_PRI      = "#f4f4f5"

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

def build_forecast_figure(stats, noaa_df, show_spaghetti, show_ribbon,
                           df_models, selected_models, noaa_cutoff_dt, show_extended,
                           w_end, band_filter="Summit", is_dark=True):

    # --- 1. GLOBAL TIMEZONE DETECTION ---
    if not stats.empty:
        tz = stats["Date"].dt.tz
    elif not df_models.empty:
        tz = df_models["Date"].dt.tz
    else:
        tz = pytz.timezone('UTC')
    
    # --- 2. PREPARE RAW DATA FOR PERCENTILES AND SPAGHETTI ---
    df_local = df_models.copy()
    if not df_local.empty:
        if not pd.api.types.is_datetime64_any_dtype(df_local["Date"]):
            df_local["Date"] = pd.to_datetime(df_local["Date"])
        if df_local["Date"].dt.tz is None:
            df_local["Date"] = df_local["Date"].dt.tz_localize("America/Denver", ambiguous='NaT', nonexistent='shift_forward')
            
        filtered = df_local[
            (df_local["Model"].isin(selected_models)) &
            (df_local["Band"] == band_filter)
        ] if "Band" in df_local.columns else df_local[df_local["Model"].isin(selected_models)]
    else:
        filtered = pd.DataFrame()

    # --- 3. SUBPLOT SETUP (Dual axes on both rows) ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.06,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]] # Updated to support dual axes on row 2
    )

    now_dt = pd.Timestamp.now(tz=tz).normalize()

 # 4. SPAGHETTI LINES (Cumulative, Resampled, Smoothed)
    if show_spaghetti and not filtered.empty:
        # High contrast palette for dense ensemble charts
        colors = [
            "#38bdf8", # Bright Sky Blue
            "#fb7185", # Vibrant Rose
            "#a3e635", # Electric Lime
            "#facc15", # Bright Yellow
            "#c084fc", # Vivid Purple
            "#fb923c", # Bright Orange
            "#2dd4bf", # Neon Teal
            "#f472b6", # Hot Pink
            "#60a5fa", # True Blue
            "#e879f9", # Bright Fuchsia
            "#4ade80", # Mint Green
            "#f87171"  # Coral Red
        ]
        
        for i, mdl in enumerate(selected_models):
            m_df = filtered[
                (filtered["Model"] == mdl) &
                (filtered["Date"] >= now_dt) &
                (filtered["Date"] <= w_end)
            ].sort_values("Date")
            
            if not m_df.empty:
                orig_max_date = m_df["Date"].max()
                
                # Resample to 6 hour granularity
                m_resampled = m_df.set_index("Date").resample("6H").sum(numeric_only=True).reset_index()
                m_resampled = m_resampled[m_resampled["Date"] <= orig_max_date]
                
                # Calculate cumulative growth
                y_val = m_resampled["Amount"].cumsum()

                fig.add_trace(go.Scatter(
                    x=m_resampled["Date"], y=y_val,
                    mode="lines",
                    line=dict(
                        width=2.5, 
                        color=colors[i % len(colors)],
                        shape="spline", 
                        smoothing=0.4 # Low tension prevents artificial dips on cumulative data
                    ),
                    name=mdl,
                    opacity=0.85,
                    hovertemplate=f"<b>{mdl}</b> (6h): %{{y:.1f}}\"<extra></extra>"
                ), row=1, col=1, secondary_y=True)
    # --- 5. MAIN STATS BARS AND ERROR LOGIC ---
    if not stats.empty:
        # Calculate 10th and 90th percentile error bars
        if show_ribbon and not filtered.empty:
            daily_p90 = filtered.groupby(filtered["Date"].dt.date)["Amount"].quantile(0.9)
            daily_p10 = filtered.groupby(filtered["Date"].dt.date)["Amount"].quantile(0.1)
            
            p90_mapped = stats["Date"].dt.date.map(daily_p90).fillna(stats["mean"])
            p10_mapped = stats["Date"].dt.date.map(daily_p10).fillna(stats["mean"])
            
            y_upper = (p90_mapped - stats["mean"]).clip(lower=0)
            y_lower = (stats["mean"] - p10_mapped).clip(lower=0)
            
            err_dict = dict(
                type='data', symmetric=False,
                array=y_upper, arrayminus=y_lower,
                color="rgba(255,255,255,0.45)" if is_dark else "rgba(0,0,0,0.45)",
                thickness=1.5, width=4
            )
        else:
            err_dict = None

        max_val = stats["mean"].max() or 0.1
        bar_colors = [
            f"rgba(56,189,248,{min(0.28 + v / max_val * 0.68, 0.95):.2f})"
            for v in stats["mean"]
        ]
        fig.add_trace(go.Bar(
            x=stats["Date"], y=stats["mean"],
            name="Daily Average",
            marker=dict(color=bar_colors, line=dict(width=0)),
            error_y=err_dict, # Apply the statistical spread here
            hovertemplate="<b>%{x|%a %b %d}</b><br>Mean: <b>%{y:.2f}\"</b><extra></extra>"
        ), row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=stats["Date"], y=stats["cumulative"],
            name="Cumulative",
            line=dict(color=ACCENT_ROSE, width=2.5),
            fill="tozeroy", fillcolor="rgba(251,113,133,0.06)",
            hovertemplate="Cumulative: <b>%{y:.1f}\"</b><extra></extra>"
        ), row=1, col=1, secondary_y=True)

    # --- 6. EXTENDED FORECAST CUTOFF ---
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

    # --- 7. NOAA OUTLOOK (DUAL AXES) ---
    if not noaa_df.empty:
        noaa_plot = noaa_df.copy()
        if noaa_plot["Time"].dt.tz is None:
            noaa_plot["Time"] = noaa_plot["Time"].dt.tz_localize("UTC").dt.tz_convert(tz)
        w_window = noaa_plot[noaa_plot["Time"] <= w_end]
        
        if not w_window.empty:
            fig.add_trace(go.Scatter(
                x=w_window["Time"], y=w_window["Temp"],
                name="Temp °F",
                line=dict(color=TEXT_SEC, width=1.5),
                hovertemplate="%{y:.0f}°F<extra>Temp</extra>"
            ), row=2, col=1, secondary_y=False) # Bound to left axis
            
            fig.add_trace(go.Scatter(
                x=w_window["Time"], y=w_window["Humidity"],
                name="Humidity %",
                line=dict(color=ACCENT_TEAL, width=1.5, dash="dot"),
                hovertemplate="%{y:.0f}%<extra>Humidity</extra>"
            ), row=2, col=1, secondary_y=True) # Bound to right axis
            
            fig.add_hline(y=32, line=dict(color="rgba(56,189,248,0.22)", dash="dash", width=1), row=2, col=1)

    # --- 8. LAYOUT AND STYLING ---
    layout_update = PLOTLY_TEMPLATE["layout"].to_plotly_json()
    layout_update.update(dict(
        height=480, bargap=0.3,
        yaxis=dict(title="Snow (in)", gridcolor=PLOTLY_GRID, zeroline=False, showline=False),
        yaxis2=dict(
            overlaying="y", side="right", showgrid=False, zeroline=False,
            title="Cumul (in)", tickfont=dict(color="rgba(251,113,133,0.6)"),
            title_font=dict(color="rgba(251,113,133,0.6)")
        ),
        # Dedicated Temperature Axis
        yaxis3=dict(
            title="Temp °F", gridcolor=PLOTLY_GRID, zeroline=False, showline=False,
            tickfont=dict(color=TEXT_SEC), title_font=dict(color=TEXT_SEC)
        ),
        # Dedicated Humidity Axis
        yaxis4=dict(
            title="Humidity %", overlaying="y3", side="right", showgrid=False, zeroline=False,
            tickfont=dict(color=ACCENT_TEAL), title_font=dict(color=ACCENT_TEAL),
            range=[0, 105] # Fix humidity scale
        )
    ))
    fig.update_layout(layout_update)
    return fig