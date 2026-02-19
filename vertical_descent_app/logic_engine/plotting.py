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
                           band_filter="Summit", is_dark=True):

    # Adjust colors based on theme if passed
    # Note: Global constants in logic.py are fixed, but we could make them dynamic or arguments.
    # For now, I'm using the constants defined above which match the dark theme in dashboard.py roughly.
    # To fully support theming in logic, we might need to pass colors as args.

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
        # Make a local copy to ensure date handling
        df_local = df_models.copy()

        # --- Ensure Date is datetime and timezone-aware ---
        if not pd.api.types.is_datetime64_any_dtype(df_local["Date"]):
            df_local["Date"] = pd.to_datetime(df_local["Date"])
        if df_local["Date"].dt.tz is None:
            df_local["Date"] = df_local["Date"].dt.tz_localize("America/Denver", ambiguous='NaT', nonexistent='shift_forward')
        # -------------------------------------------------

        filtered = df_local[
            (df_local["Model"].isin(selected_models)) &
            (df_local["Band"] == band_filter)
        ] if "Band" in df_local.columns else df_local[df_local["Model"].isin(selected_models)]

        # Use the timezone from the data
        if not filtered.empty:
            tz = filtered["Date"].dt.tz
        else:
            # Fallback to UTC if no data exists to prevent a NameError
            tz = pytz.timezone('UTC')
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
