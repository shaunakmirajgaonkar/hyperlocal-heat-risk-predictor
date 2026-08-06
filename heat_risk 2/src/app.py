"""
Hyperlocal Heat-Risk Predictor Dashboard
100% local Dash application. No external API calls at runtime.
Run: python app.py  -> open http://127.0.0.1:8050
"""

import io
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from heat_model import (
    generate_city_grid, compute_heat_index, train_surrogate_model,
    daily_hourly_forecast, safe_work_windows, safest_route
)

# ---------------- Data prep (all local/offline) ----------------
GRID = generate_city_grid(n_points=400, grid_size=20)
DEFAULT_TEMP, DEFAULT_HOUR = 36, 14
RISK = compute_heat_index(GRID, air_temp_c=DEFAULT_TEMP, hour=DEFAULT_HOUR)
MODEL, MODEL_COLS = train_surrogate_model(RISK)
SEGMENT_IDS = RISK["segment_id"].tolist()

COLORS = {
    "bg": "#F7F4EE",
    "panel": "#FFFFFF",
    "panel2": "#FBF9F4",
    "border": "#E4DFD3",
    "text": "#2B2620",
    "muted": "#7A7367",
    "amber": "#E08A2C",
    "ember": "#E4572E",
    "safe": "#2E9E6C",
    "caution": "#D9A017",
    "danger": "#E4572E",
    "extreme": "#B3001B",
}

RISK_COLORSCALE = [
    [0.0, "#2E9E6C"],
    [0.25, "#8FD19E"],
    [0.5, "#F4C95D"],
    [0.75, "#F08A4B"],
    [1.0, "#C1272D"],
]

FONT = "'Space Grotesk', 'Segoe UI', sans-serif"
MONO = "'JetBrains Mono', 'Courier New', monospace"

app = dash.Dash(__name__, title="Hyperlocal Heat-Risk Predictor")
server = app.server

# ---------------- Layout helpers ----------------

def stat_card(label, value, sub, accent):
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "letterSpacing": "1.5px", "textTransform": "uppercase",
                                "color": COLORS["muted"], "fontWeight": "600"}),
        html.Div(value, style={"fontSize": "34px", "fontWeight": "700", "color": accent,
                                "fontFamily": MONO, "lineHeight": "1.1", "marginTop": "4px"}),
        html.Div(sub, style={"fontSize": "12px", "color": COLORS["muted"], "marginTop": "4px"}),
    ], style={
        "background": COLORS["panel"], "border": f"1px solid {COLORS['border']}",
        "borderRadius": "10px", "padding": "16px 18px", "flex": "1", "minWidth": "150px",
        "boxShadow": "0 1px 3px rgba(43,38,32,0.06)",
    })


def section(title, children, extra_style=None):
    style = {
        "background": COLORS["panel"], "border": f"1px solid {COLORS['border']}",
        "borderRadius": "12px", "padding": "20px", "marginBottom": "20px",
        "boxShadow": "0 1px 3px rgba(43,38,32,0.06)",
    }
    if extra_style:
        style.update(extra_style)
    return html.Div([
        html.Div(title, style={
            "fontSize": "13px", "letterSpacing": "1.5px", "textTransform": "uppercase",
            "color": COLORS["amber"], "fontWeight": "700", "marginBottom": "14px",
            "borderBottom": f"1px solid {COLORS['border']}", "paddingBottom": "10px"
        }),
        children
    ], style=style)


app.layout = html.Div([
    dcc.Store(id="risk-store"),

    # ---------- Header ----------
    html.Div([
        html.Div([
            html.Div("HEATLINE", style={
                "fontFamily": FONT, "fontWeight": "800", "fontSize": "26px",
                "letterSpacing": "1px", "color": COLORS["text"]
            }),
            html.Div("Hyperlocal Heat-Risk Predictor \u2014 street-level safety for outdoor workers",
                      style={"fontSize": "13px", "color": COLORS["muted"], "marginTop": "2px"}),
        ]),
        html.Div([
            html.Span("\u25CF ", style={"color": COLORS["safe"]}),
            html.Span("100% LOCAL \u00b7 NO EXTERNAL API", style={
                "fontSize": "11px", "letterSpacing": "1px", "color": COLORS["muted"], "fontFamily": MONO
            }),
        ], style={"alignSelf": "center"})
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "22px 32px", "borderBottom": f"2px solid {COLORS['amber']}",
        "background": f"linear-gradient(90deg, #FFFFFF 0%, {COLORS['bg']} 100%)"
    }),

    html.Div([
        # ---------- Controls ----------
        html.Div([
            html.Div("CONDITIONS", style={
                "fontSize": "12px", "letterSpacing": "1.5px", "color": COLORS["amber"],
                "fontWeight": "700", "marginBottom": "12px"
            }),
            html.Label("Ambient Air Temperature (\u00b0C)", style={"fontSize": "12px", "color": COLORS["muted"]}),
            dcc.Slider(id="temp-slider", min=25, max=48, step=0.5, value=DEFAULT_TEMP,
                       marks={t: f"{t}\u00b0" for t in range(25, 49, 5)},
                       tooltip={"placement": "bottom", "always_visible": False}),
            html.Div(style={"height": "18px"}),
            html.Label("Hour of Day", style={"fontSize": "12px", "color": COLORS["muted"]}),
            dcc.Slider(id="hour-slider", min=0, max=23, step=1, value=DEFAULT_HOUR,
                       marks={h: f"{h}h" for h in range(0, 24, 3)},
                       tooltip={"placement": "bottom", "always_visible": False}),
            html.Div(style={"height": "22px"}),
            html.Label("Route Planner", style={"fontSize": "12px", "color": COLORS["muted"]}),
            html.Div([
                dcc.Dropdown(id="start-seg", options=SEGMENT_IDS, value=SEGMENT_IDS[0],
                             placeholder="Start segment", style={"marginBottom": "8px", "color": "#111"}),
                dcc.Dropdown(id="end-seg", options=SEGMENT_IDS, value=SEGMENT_IDS[-1],
                             placeholder="End segment", style={"color": "#111"}),
            ]),
            html.Button("FIND SAFEST ROUTE", id="route-btn", n_clicks=0, style={
                "marginTop": "14px", "width": "100%", "padding": "12px", "border": "none",
                "borderRadius": "8px", "background": COLORS["amber"], "color": "#FFFFFF",
                "fontWeight": "700", "letterSpacing": "1px", "cursor": "pointer", "fontSize": "13px"
            }),
        ], style={
            "background": COLORS["panel"], "border": f"1px solid {COLORS['border']}",
            "borderRadius": "12px", "padding": "20px", "width": "290px", "flexShrink": "0",
            "height": "fit-content", "boxShadow": "0 1px 3px rgba(43,38,32,0.06)",
        }),

        # ---------- Main column ----------
        html.Div([
            html.Div(id="stat-cards", style={"display": "flex", "gap": "14px", "marginBottom": "20px",
                                              "flexWrap": "wrap"}),

            section("Street-Level Thermal Scan", dcc.Graph(id="grid-map", config={"displayModeBar": False})),

            html.Div([
                html.Div([section("24-Hour Risk Forecast",
                                   dcc.Graph(id="hourly-chart", config={"displayModeBar": False}))],
                         style={"flex": "1.3", "minWidth": "0"}),
                html.Div([section("Safe Work Windows", html.Div(id="work-windows"))],
                         style={"flex": "1", "minWidth": "0"}),
            ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}),

            section("Safest Walking Route", dcc.Graph(id="route-map", config={"displayModeBar": False})),

        ], style={"flex": "1", "minWidth": "0"})
    ], style={"display": "flex", "gap": "20px", "padding": "24px 32px", "alignItems": "flex-start"}),

    html.Div("Synthetic microclimate simulation for demonstration \u2014 tree cover, surface material, "
             "shading and traffic heat are modeled locally; no live sensor or weather feed is used.",
             style={"textAlign": "center", "color": COLORS["muted"], "fontSize": "11px",
                    "padding": "18px", "borderTop": f"1px solid {COLORS['border']}"})

], style={
    "background": COLORS["bg"], "minHeight": "100vh", "fontFamily": FONT, "color": COLORS["text"],
    "margin": "0", "paddingBottom": "10px"
})


# ---------------- Callbacks ----------------

@app.callback(
    Output("risk-store", "data"),
    Output("stat-cards", "children"),
    Output("grid-map", "figure"),
    Output("hourly-chart", "figure"),
    Output("work-windows", "children"),
    Input("temp-slider", "value"),
    Input("hour-slider", "value"),
)
def update_dashboard(air_temp, hour):
    risk_df = compute_heat_index(GRID, air_temp_c=air_temp, hour=hour)

    # ---- stat cards ----
    avg_risk = risk_df["risk_score"].mean()
    max_risk = risk_df["risk_score"].max()
    extreme_pct = (risk_df["risk_level"] == "Extreme").mean() * 100
    safest_seg = risk_df.loc[risk_df["risk_score"].idxmin(), "segment_id"]

    def accent_for(v):
        if v < 25: return COLORS["safe"]
        if v < 50: return COLORS["caution"]
        if v < 75: return COLORS["danger"]
        return COLORS["extreme"]

    cards = [
        stat_card("District Avg Risk", f"{avg_risk:.0f}", "0\u2013100 heat index", accent_for(avg_risk)),
        stat_card("Peak Segment Risk", f"{max_risk:.0f}", "hottest street segment", accent_for(max_risk)),
        stat_card("Extreme Zones", f"{extreme_pct:.0f}%", "of segments at Extreme", COLORS["extreme"]),
        stat_card("Safest Segment", safest_seg, "coolest right now", COLORS["safe"]),
    ]

    # ---- grid heat map ----
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=risk_df["x"], y=risk_df["y"], mode="markers",
        marker=dict(
            size=14, color=risk_df["risk_score"], colorscale=RISK_COLORSCALE,
            cmin=0, cmax=100, showscale=True,
            colorbar=dict(title="Risk", tickfont=dict(color=COLORS["muted"]),
                          title_font=dict(color=COLORS["muted"]), thickness=14, len=0.8,
                          outlinewidth=0),
            line=dict(width=1, color="rgba(255,255,255,0.15)")
        ),
        customdata=np.stack([risk_df["segment_id"], risk_df["surface"], risk_df["risk_level"].astype(str),
                              risk_df["local_temp_c"], risk_df["tree_cover"] * 100], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>Risk level: %{customdata[2]}<br>"
            "Local temp: %{customdata[3]}\u00b0C<br>Surface: %{customdata[1]}<br>"
            "Tree cover: %{customdata[4]:.0f}%<extra></extra>"
        )
    ))
    fig.update_layout(
        template=None, plot_bgcolor=COLORS["panel2"], paper_bgcolor=COLORS["panel"],
        margin=dict(l=10, r=10, t=10, b=10), height=430,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        font=dict(color=COLORS["text"], family=FONT),
    )

    # ---- hourly forecast ----
    hourly = daily_hourly_forecast(risk_df, base_temp=air_temp)
    bar_colors = [accent_for(v) for v in hourly["avg_risk"]]
    hfig = go.Figure()
    hfig.add_trace(go.Bar(
        x=hourly["hour"], y=hourly["avg_risk"], marker_color=bar_colors,
        marker_line_width=0,
        hovertemplate="Hour %{x}:00<br>Risk: %{y:.0f}<extra></extra>",
        name="Risk"
    ))
    hfig.add_vline(x=hour, line_dash="dash", line_color=COLORS["amber"], line_width=2)
    hfig.update_layout(
        plot_bgcolor=COLORS["panel2"], paper_bgcolor=COLORS["panel"],
        margin=dict(l=10, r=10, t=10, b=10), height=280,
        xaxis=dict(title="Hour", color=COLORS["muted"], gridcolor=COLORS["border"], dtick=2),
        yaxis=dict(title="Avg Risk", range=[0, 100], color=COLORS["muted"], gridcolor=COLORS["border"]),
        font=dict(color=COLORS["text"], family=FONT), showlegend=False,
    )

    # ---- safe work windows ----
    hourly_map = dict(zip(hourly["hour"], hourly["avg_risk"]))
    windows = safe_work_windows(hourly_map)
    level_style = {
        "Safe": COLORS["safe"], "Caution": COLORS["caution"], "Unsafe": COLORS["danger"]
    }
    win_rows = []
    for s, e, level in windows:
        win_rows.append(html.Div([
            html.Div(f"{s:02d}:00 \u2013 {e:02d}:00", style={"fontFamily": MONO, "fontSize": "13px",
                                                              "color": COLORS["text"], "width": "110px"}),
            html.Div(level, style={
                "background": level_style[level] + "22", "color": level_style[level],
                "border": f"1px solid {level_style[level]}", "borderRadius": "6px",
                "padding": "3px 10px", "fontSize": "11px", "fontWeight": "700", "letterSpacing": "0.5px"
            })
        ], style={"display": "flex", "alignItems": "center", "gap": "12px", "padding": "8px 0",
                   "borderBottom": f"1px solid {COLORS['border']}"}))

    guidance = html.Div([
        html.Div("RECOMMENDATION", style={"fontSize": "11px", "color": COLORS["muted"],
                                           "letterSpacing": "1px", "marginTop": "14px", "marginBottom": "6px"}),
        html.Div(_recommendation_text(windows), style={"fontSize": "13px", "color": COLORS["text"],
                                                        "lineHeight": "1.5"})
    ])

    return risk_df.to_json(date_format="iso", orient="split"), cards, fig, hfig, win_rows + [guidance]


def _recommendation_text(windows):
    safe_windows = [(s, e) for s, e, lvl in windows if lvl == "Safe"]
    if safe_windows:
        parts = [f"{s:02d}:00\u2013{e:02d}:00" for s, e in safe_windows]
        return f"Schedule strenuous outdoor work during: {', '.join(parts)}. Avoid peak hours entirely if possible."
    return "No fully safe window today \u2014 use Caution-level hours, hydrate every 20 min, and take shade breaks."


@app.callback(
    Output("route-map", "figure"),
    Input("route-btn", "n_clicks"),
    State("start-seg", "value"),
    State("end-seg", "value"),
    State("risk-store", "data"),
)
def update_route(n_clicks, start_id, end_id, risk_json):
    if risk_json is None:
        risk_df = RISK
    else:
        risk_df = pd.read_json(io.StringIO(risk_json), orient="split")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=risk_df["x"], y=risk_df["y"], mode="markers",
        marker=dict(size=10, color=risk_df["risk_score"], colorscale=RISK_COLORSCALE,
                    cmin=0, cmax=100, opacity=0.35, showscale=False),
        hoverinfo="skip"
    ))

    if start_id and end_id and start_id != end_id:
        route = safest_route(risk_df, start_id, end_id)
        avg_route_risk = route["risk_score"].mean()
        direct_risk = risk_df[risk_df["segment_id"].isin([start_id, end_id])]["risk_score"].mean()

        fig.add_trace(go.Scatter(
            x=route["x"], y=route["y"], mode="lines+markers",
            line=dict(color=COLORS["amber"], width=3),
            marker=dict(size=11, color=route["risk_score"], colorscale=RISK_COLORSCALE, cmin=0, cmax=100,
                        line=dict(width=2, color=COLORS["amber"])),
            hovertemplate="<b>%{text}</b><br>Risk: %{marker.color:.0f}<extra></extra>",
            text=route["segment_id"], name="Route"
        ))
        fig.add_annotation(x=route["x"].iloc[0], y=route["y"].iloc[0], text="START",
                            showarrow=True, arrowhead=2, font=dict(color=COLORS["safe"], size=12),
                            arrowcolor=COLORS["safe"])
        fig.add_annotation(x=route["x"].iloc[-1], y=route["y"].iloc[-1], text="END",
                            showarrow=True, arrowhead=2, font=dict(color=COLORS["ember"], size=12),
                            arrowcolor=COLORS["ember"])
        title_note = f"Route avg risk: {avg_route_risk:.0f}  \u00b7  {len(route)} segments"
    else:
        title_note = "Select two different segments and click Find Safest Route"

    fig.update_layout(
        plot_bgcolor=COLORS["panel2"], paper_bgcolor=COLORS["panel"],
        margin=dict(l=10, r=10, t=36, b=10), height=420,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        font=dict(color=COLORS["text"], family=FONT),
        title=dict(text=title_note, font=dict(size=13, color=COLORS["muted"]), x=0.01)
    )
    return fig


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
