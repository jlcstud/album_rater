#!/usr/bin/env python3
"""
Minimal Dash click interaction testbed.
- Renders a simple bar chart (1..10) with fixed y values
- On click of a bar, updates that bar's height and color
- Updates title with the index clicked

This file intentionally avoids any external assets, custom JS, or
complex layout/callback patterns from the main app to isolate whether
click events reach Dash callbacks.
"""
import json
from typing import List

import dash
from dash import Dash, dcc, html, Input, Output, State

# Build a simple figure with 10 bars
x_vals = list(range(1, 11))
y_vals = [i % 5 + 5 for i in x_vals]  # some values in 5..9
base_colors = ["#2aa9ff"] * len(x_vals)

fig = {
    "data": [
        {
            "type": "bar",
            "x": x_vals,
            "y": y_vals,
            "marker": {"color": base_colors, "line": {"color": "#8fd4ff", "width": 2}},
            "name": "Test Bars",
        }
    ],
    "layout": {
        "title": "Click a bar to update",
        "paper_bgcolor": "#111418",
        "plot_bgcolor": "#111418",
        "font": {"color": "#dfe2e7"},
        "margin": {"l": 40, "r": 20, "t": 50, "b": 40},
        # Match main app: prevent zoom/pan via fixedrange and dragmode False
        "yaxis": {"range": [0, 12], "dtick": 1, "fixedrange": True},
        "xaxis": {"tickmode": "array", "tickvals": x_vals, "ticktext": x_vals, "fixedrange": True},
        "dragmode": False,
    },
}

# NOTE: external_scripts/styles intentionally omitted to avoid side-effects
# Also ignore the default ./assets directory to avoid custom JS interfering
app = Dash(__name__, assets_ignore=".*")  # no external stylesheets, ignore assets
server = app.server

app.layout = html.Div(
    [
        dcc.Graph(
            id="test-graph",
            figure=fig,
            # Match main app: disable zoom interactions but keep events enabled
            config={
                "displayModeBar": False,
                "scrollZoom": False,
                "doubleClick": False,
                "staticPlot": False,
            },
        ),
        # Mirror of clickData for visibility
        html.Pre(id="debug-click", style={"whiteSpace": "pre-wrap", "fontSize": "12px"}),
    ],
    style={"padding": "16px", "backgroundColor": "#0e0f12", "minHeight": "100vh"},
)


@app.callback(
    Output("test-graph", "figure"),
    Output("debug-click", "children"),
    Input("test-graph", "clickData"),
    State("test-graph", "figure"),
    prevent_initial_call=True,
)
def on_click(clickData, current_fig):
    # Pass-through debug so we can see raw event data
    debug_text = json.dumps(clickData, indent=2)

    # Defensive: if clickData missing expected fields, do nothing
    try:
        pt = clickData["points"][0]
        x_clicked = pt.get("x")
        point_index = pt.get("pointIndex")  # usually 0-based index
    except Exception:
        return dash.no_update, debug_text

    if x_clicked is None or point_index is None:
        return dash.no_update, debug_text

    # Copy current series to avoid in-place mutation side effects
    series = current_fig["data"][0].copy()

    # Update y and color at the clicked index
    y = list(series.get("y", []))
    colors = list(series.get("marker", {}).get("color", []))

    # Toggle behavior: if already boosted, reset; else boost
    boosted = y[point_index] >= 10
    y[point_index] = 6 if boosted else 10
    colors[point_index] = "#ffc857" if not boosted else "#2aa9ff"

    # Rebuild series and figure
    new_series = series
    new_series["y"] = y
    new_marker = dict(series.get("marker", {}))
    new_marker["color"] = colors
    new_series["marker"] = new_marker

    new_layout = dict(current_fig.get("layout", {}))
    new_layout["title"] = f"Clicked x={x_clicked} (index {point_index})"

    new_fig = {"data": [new_series], "layout": new_layout}
    return new_fig, debug_text


if __name__ == "__main__":
    # Run on a separate port to avoid colliding with main app
    app.run_server(debug=True, port=8060)
