# app.py
from dash import Dash, html, dcc, Input, Output, State, no_update
from dash_extensions import EventListener
from dash_svg import Svg, Line, Text, Rect, Polyline, Polygon
import numpy as np
import time
from bisect import bisect_right

# ======== Config ========
NUM_POINTS = 12
Y_MIN, Y_MAX = 0.0, 10.0
INIT = np.array([5, 5, 5, 8.4, 7.2, 3.7, 7.5, 3.0, 7.6, 6.3, 5.0, 5.0], dtype=float)

# Toggle drag behavior
USE_DRAG = False  # set False to disable dragging entirely (click-to-set still works)

# Per-bar widths (len == NUM_POINTS). Example: all 1.0 (uniform); change as you like.
BAR_WIDTHS = np.array([1.0]*NUM_POINTS, dtype=float)
BAR_WIDTHS = np.array([1,4,2,4.5,3.11, 1,2,1.5, 3, 2.5, 1, 2.25], dtype=float)
# e.g. try: BAR_WIDTHS = np.array([1,4,2,4.5,3.11, 1,2,1.5, 3, 2.5, 1, 2.25], float)

# Throttle during drag
MAX_UPS = 20
MIN_INTERVAL = 1.0 / MAX_UPS

# SVG canvas + plot area
SVG_W, SVG_H = 1000, 460
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 60, 20, 46, 52  # extra top for labels
PLOT_W = SVG_W - MARGIN_L - MARGIN_R
PLOT_H = SVG_H - MARGIN_T - MARGIN_B

CURVE_COLOR = "#6b83ff"

app = Dash(__name__)
server = app.server

# ======== Geometry helpers (variable-width bars) ========
def build_geometry(widths):
    w = np.asarray(widths, float)
    assert w.ndim == 1 and len(w) == NUM_POINTS and np.all(w > 0), \
        "BAR_WIDTHS must be positive and match NUM_POINTS"

    lefts  = np.empty(NUM_POINTS, float)
    rights = np.empty(NUM_POINTS, float)

    # first bar starts half a bar to the left of its center
    lefts[0]  = -w[0] / 2.0
    rights[0] = lefts[0] + w[0]
    for i in range(1, NUM_POINTS):
        lefts[i]  = rights[i - 1]
        rights[i] = lefts[i] + w[i]

    centers = 0.5 * (lefts + rights)

    # ✅ Correct domain: exactly spans the bars (no extra padding)
    L = lefts[0]          # == centers[0] - w[0]/2
    R = rights[-1]        # == centers[-1] + w[-1]/2
    return L, R, lefts, rights, centers


DOMAIN_L, DOMAIN_R, LEFTS, RIGHTS, CENTERS = build_geometry(BAR_WIDTHS)

def x_to_px(x):
    """Map data-x to pixel-x given variable domain."""
    return MARGIN_L + (x - DOMAIN_L) * (PLOT_W / (DOMAIN_R - DOMAIN_L))

def y_to_px(y):
    return MARGIN_T + (Y_MAX - y) * (PLOT_H / (Y_MAX - Y_MIN))

# ======== Step curve builders ========
def build_step_points(yvals):
    """
    Step curve (nearest-neighbour): horizontal across each bar's width,
    vertical jump at each bar boundary.
    Returns list of (x,y) in data coords.
    """
    pts = []
    y = list(map(float, yvals))
    # start at left edge of bar 0
    pts.append((LEFTS[0], y[0]))
    for i in range(NUM_POINTS):
        # horizontal to the right edge of bar i
        pts.append((RIGHTS[i], y[i]))
        # vertical jump to next bar level
        if i < NUM_POINTS - 1:
            pts.append((RIGHTS[i], y[i+1]))
    return pts

def render_svg_children(yvals):
    elems = []

    # y grid & labels
    for v in range(int(Y_MIN), int(Y_MAX) + 1):
        y = y_to_px(v)
        elems += [
            Line(x1=MARGIN_L, y1=y, x2=MARGIN_L + PLOT_W, y2=y,
                 stroke="#334955", strokeWidth=1, style={"pointerEvents": "none", "opacity": 0.35}),
            Text(str(v), x=MARGIN_L - 10, y=y + 4, fill="#9fb3bf",
                 textAnchor="end", style={"fontSize": "12px", "pointerEvents": "none"}),
        ]

    # vertical guides at bar centers + bottom labels (1..N)
    for i in range(NUM_POINTS):
        cx = x_to_px(CENTERS[i])
        elems += [
            Line(x1=cx, y1=MARGIN_T, x2=cx, y2=MARGIN_T + PLOT_H,
                 stroke="#22313b", strokeWidth=1, style={"pointerEvents": "none", "opacity": 0.25}),
            Text(str(i + 1), x=cx, y=MARGIN_T + PLOT_H + 20, fill="#9fb3bf",
                 textAnchor="middle", style={"fontSize": "12px", "pointerEvents": "none"}),
        ]

    # top labels = current y-values (centered on bars)
    for i, yv in enumerate(yvals):
        elems.append(
            Text(f"{float(yv):.1f}", x=x_to_px(CENTERS[i]), y=MARGIN_T - 14, fill="#cbd6dd",
                 textAnchor="middle", style={"fontSize": "12px", "pointerEvents": "none"})
        )

    # border
    elems.append(
        Rect(x=MARGIN_L, y=MARGIN_T, width=PLOT_W, height=PLOT_H,
             fill="none", stroke="#44606f", strokeWidth=2, style={"pointerEvents": "none"})
    )

    # step curve + under-fill
    step_pts = build_step_points(yvals)

    # Under-fill: start at domain left on baseline, go up to level at LEFTS[0],
    # follow the step, then drop to baseline at domain right.
    poly_pts = [(DOMAIN_L, Y_MIN), (LEFTS[0], Y_MIN), (LEFTS[0], float(yvals[0]))] \
            + step_pts[1:] + [(DOMAIN_R, Y_MIN)]

    poly_points_attr = " ".join(f"{x_to_px(xd):.3f},{y_to_px(yd):.3f}" for xd, yd in poly_pts)
    elems.append(
        Polygon(points=poly_points_attr, fill=CURVE_COLOR, stroke="none",
                style={"pointerEvents": "none", "fillOpacity": 0.5})
    )


    line_points_attr = " ".join(f"{x_to_px(xd):.3f},{y_to_px(yd):.3f}" for xd, yd in step_pts)
    elems.append(
        Polyline(
            points=line_points_attr,
            fill="none", stroke=CURVE_COLOR, strokeWidth=6,
            strokeLinejoin="miter", strokeLinecap="butt",
            style={"pointerEvents": "none"}
        )
    )
    return elems

# ======== Event handling (click + optional drag) ========
def point_from_offsets(offsetX, offsetY):
    """Return (col_idx_0based, y_value) if inside plot, else None."""
    if offsetX is None or offsetY is None:
        return None
    px, py = float(offsetX), float(offsetY)
    if not (MARGIN_L <= px <= MARGIN_L + PLOT_W and MARGIN_T <= py <= MARGIN_T + PLOT_H):
        return None

    # pixel -> data x
    x_data = DOMAIN_L + (px - MARGIN_L) * (DOMAIN_R - DOMAIN_L) / PLOT_W

    # which bar: find first right-edge > x_data
    idx = bisect_right(RIGHTS.tolist(), x_data)
    idx = min(max(idx, 0), NUM_POINTS - 1)

    # pixel y -> data y
    rely = (py - MARGIN_T) / PLOT_H
    y_val = Y_MAX - rely * (Y_MAX - Y_MIN)
    y_val = float(min(max(y_val, Y_MIN), Y_MAX))
    return idx, y_val

LISTEN_EVENTS = [
    {"event": "pointerdown",  "props": ["type", "offsetX", "offsetY", "buttons"]},
    {"event": "pointermove",  "props": ["type", "offsetX", "offsetY", "buttons"]},
    {"event": "pointerup",    "props": ["type", "offsetX", "offsetY", "buttons"]},
    {"event": "pointerleave", "props": ["type", "offsetX", "offsetY", "buttons"]},
]

app.layout = html.Div(
    style={"backgroundColor": "#0b1a23", "minHeight": "100vh", "color": "white", "padding": "12px"},
    children=[
        html.H3("Click or drag a bar to set its value (nearest-neighbour / step)"),
        dcc.Store(id="y-store", data=INIT.tolist()),
        dcc.Store(id="drag-store", data={"dragging": False, "idx": None, "t": 0.0}),
        EventListener(
            id="svg-events",
            events=LISTEN_EVENTS,
            logging=False,
            children=Svg(
                id="chart-svg",
                width=SVG_W, height=SVG_H,
                viewBox=f"0 0 {SVG_W} {SVG_H}",
                style={
                    "background": "#0b1a23",
                    "userSelect": "none",
                    "touchAction": "none",
                    "cursor": "crosshair",
                },
                children=render_svg_children(INIT),
            ),
        ),
    ],
)

@app.callback(
    Output("y-store", "data"),
    Output("chart-svg", "children"),
    Output("drag-store", "data"),
    Input("svg-events", "n_events"),
    State("svg-events", "event"),
    State("y-store", "data"),
    State("drag-store", "data"),
    prevent_initial_call=True,
)
def handle_pointer(n_events, evt, ydata, drag):
    if not evt:
        return ydata, no_update, drag

    etype = evt.get("type")
    buttons = int(evt.get("buttons") or 0)
    res = point_from_offsets(evt.get("offsetX"), evt.get("offsetY"))

    # Click or drag start
    if etype == "pointerdown" and res is not None:
        idx, y_new = res
        ydata = list(ydata)
        ydata[idx] = y_new
        if USE_DRAG:
            drag = {"dragging": True, "idx": idx, "t": time.perf_counter()}
        else:
            drag = {"dragging": False, "idx": None, "t": 0.0}
        return ydata, render_svg_children(ydata), drag

    # Dragging (only if enabled)
    if USE_DRAG and etype == "pointermove" and drag.get("dragging") and drag.get("idx") is not None and (buttons & 1):
        if res is None:
            return ydata, no_update, drag  # moved outside plot
        now = time.perf_counter()
        last = float(drag.get("t", 0.0))
        if (now - last) < MIN_INTERVAL:       # throttle
            return ydata, no_update, drag

        idx_drag = drag["idx"]
        _, y_new = res
        ydata = list(ydata)
        ydata[idx_drag] = y_new
        drag = {**drag, "t": now}
        return ydata, render_svg_children(ydata), drag

    # End drag on release/leave
    if etype in ("pointerup", "pointerleave"):
        if drag.get("dragging"):
            drag = {"dragging": False, "idx": None, "t": 0.0}
            return ydata, no_update, drag
        return ydata, no_update, drag

    return ydata, no_update, drag

if __name__ == "__main__":
    app.run_server(debug=True)
