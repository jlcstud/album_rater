#!/usr/bin/env python3
import os, json, math, random, base64, io, re
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

try:
    import numpy as np
except:
    raise ValueError("If numpy is not found, active 'ar-env' with conda")
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
import dash
from dash import Dash, html, dcc, Input, Output, State, MATCH, ALL, callback_context
import dash_bootstrap_components as dbc

# --------- Config / Paths ----------
DATA_DIR = Path("output_spotify")         # where albums.json & images/ live
JSON_PATH = DATA_DIR / "albums.json"      # will now also hold ratings inline
# Removed separate ratings.json – ratings & ignored arrays now stored per album object.

THUMB_DIR = DATA_DIR / "thumbs"
THUMB_DIR.mkdir(parents=True, exist_ok=True)

# --------- Utilities ----------
def load_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    return {}

def save_json(path: Path, obj: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)

def ms_to_mmss(ms:int) -> str:
    s = ms//1000
    m, s = divmod(s, 60)
    return f"{m}:{s:02d}"

def album_duration_ms(album:dict) -> int:
    return sum(t["duration_ms"] or 0 for t in album.get("tracks", []))

def image_to_data_uri(path: Path, size:Tuple[int,int]=None) -> str:
    img = Image.open(path).convert("RGB")
    if size: img.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def get_or_make_thumb(png_path: Path, size=(128,128)) -> Path:
    tpath = THUMB_DIR / (png_path.stem + f"_{size[0]}x{size[1]}.png")
    if not tpath.exists():
        try:
            img = Image.open(png_path).convert("RGB")
            img.thumbnail(size, Image.LANCZOS)
            img.save(tpath, format="PNG")
        except Exception:
            return png_path
    return tpath

def norm_text(s:str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[,.;:!?'\"()\[\]{}]", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" the ", " ")  # mild normalization
    return s.strip()

# optional fuzzy score (RapidFuzz) → fallback to difflib if not installed
try:
    from rapidfuzz import fuzz
    def fuzzy_ratio(a,b): return fuzz.token_set_ratio(a, b)
except Exception:
    from difflib import SequenceMatcher
    def fuzzy_ratio(a,b): return int(100*SequenceMatcher(None, a, b).ratio())

def highlight_match(text:str, query:str) -> List:
    """Return list of spans where the matched fragment is bold."""
    t, q = text, query
    iloc = t.lower().find(q.lower())
    if iloc == -1:
        return [text]
    return [t[:iloc], html.Strong(t[iloc:iloc+len(q)]), t[iloc+len(q):]]

def luminance(rgb):
    r,g,b = [x/255 for x in rgb]
    return 0.2126*r + 0.7152*g + 0.0722*b

# ---- Color Theming ----
DEFAULT_THEME = {
    "bg_superdark": "#0e0f12",
    "bg_dark":      "#15171c",
    "text":         "#dfe2e7",
    "muted":        "#9aa3ad",
    # accent fallbacks (pleasant blue/cyan/purple/amber/teal)
    "accents": ["#2aa9ff", "#00d0c5", "#7f67ff", "#ffc857", "#22b573"]
}

def dominant_colors(img, n=5, std=0.0):
    # --- Load image ---
    if isinstance(img, str):
        pil = Image.open(img).convert("RGB")
        arr = np.asarray(pil)
    elif isinstance(img, np.ndarray):
        arr = img
        # Coerce to RGB uint8 if needed
        if arr.ndim != 3 or arr.shape[2] != 3:
            # Try to interpret via PIL for safety (handles grayscale/alpha)
            pil = Image.fromarray(
                (np.clip(arr, 0, 255)).astype(np.uint8) if arr.dtype != np.uint8 else arr
            ).convert("RGB")
            arr = np.asarray(pil)
        else:
            # If float-like in [0,1], scale to [0,255]
            if np.issubdtype(arr.dtype, np.floating):
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).round().astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).round().astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        raise TypeError("img must be a file path (str) or an RGB numpy array (H, W, 3).")

    # --- Optional Gaussian blur ---
    if std > 0:
        # Work in float, blur per-channel, then back to uint8
        arr_f = arr.astype(np.float32)
        # Apply same sigma to each channel
        for c in range(3):
            arr_f[..., c] = gaussian_filter(arr_f[..., c], sigma=std, mode="reflect")
        arr = np.clip(arr_f, 0, 255).round().astype(np.uint8)

    # --- Flatten pixels for clustering ---
    pixels = arr.reshape(-1, 3).astype(np.float32)

    # --- K-means clustering ---
    # Note: using n_init=10 for broad sklearn compatibility.
    km = KMeans(n_clusters=n, n_init=10)
    labels = km.fit_predict(pixels)  # (N,)
    centers = km.cluster_centers_      # (n, 3) float

    # --- Order clusters by size (descending) ---
    counts = np.bincount(labels, minlength=n)
    order = np.argsort(-counts)  # largest first
    ordered_centers = centers[order]

    # --- Return uint8 RGB colors ---
    colors = np.clip(np.rint(ordered_centers), 0, 255).astype(np.uint8)
    return colors

def theme_from_cover(cover_png: Path) -> Dict:
    # Try to get 5 bright-ish colors from cover; fallback to default accents
    try:
        img = Image.open(cover_png).convert("RGB")
        arr = np.array(img.resize((128,128)))
        cols = dominant_colors(arr)
        good = []
        for c in cols:
            if luminance(c) > 0.35:   # reject too dark
                good.append("#%02x%02x%02x" % tuple(int(v) for v in c))
        if len(good) < 3:
            accents = DEFAULT_THEME["accents"]
        else:
            accents = good[:5]
    except Exception:
        accents = DEFAULT_THEME["accents"]
    th = DEFAULT_THEME.copy()
    th["accents"] = accents
    return th

# --------- Data loading / indexing ----------
albums_doc = load_json(JSON_PATH)
ALBUMS: Dict[str,dict] = {a["album_id"]: a for a in albums_doc.get("albums", [])}
ALL_ALBUMS = list(ALBUMS.values())

# Ratings are embedded per-album under optional keys:
#   album["ratings"]: list of length == len(tracks) with float|None
#   album["ignored"]: list of bool of same length
# We keep a transient RATINGS mirror for minimal changes to rest of code, but
# it lazily reads from ALBUMS when accessed and writes back to albums.json.
RATINGS: Dict[str,dict] = {}

def _persist_albums_inline():
    """Write ALBUMS (with any inline ratings/ignored) back to albums.json.
    Keeps non-album top-level metadata from original document if present."""
    # Reconstruct the original document shape
    doc = albums_doc.copy()
    # Replace the albums list with updated album dicts from ALBUMS (preserving order of original list)
    ordered = []
    for a in albums_doc.get("albums", []):
        aid = a.get("album_id")
        if aid in ALBUMS:
            ordered.append(ALBUMS[aid])
    # Include any albums that might have been added dynamically (unlikely here)
    existing_ids = {a["album_id"] for a in ordered}
    for aid, a in ALBUMS.items():
        if aid not in existing_ids:
            ordered.append(a)
    doc["albums"] = ordered
    save_json(JSON_PATH, doc)

def ensure_rating_struct(album_id:str):
    """Ensure RATINGS mirror has proper shape, lazily reading from album.
    If album lacks keys, treat as all None / all False but DON'T persist until
    the first user modification (dynamic behavior)."""
    alb = ALBUMS[album_id]
    n = len(alb["tracks"])
    if album_id not in RATINGS or len(RATINGS[album_id].get("ratings", [])) != n:
        # Pull existing if present & length matches; else synthesize ephemeral
        r = alb.get("ratings")
        ig = alb.get("ignored")
        if not isinstance(r, list) or len(r) != n:
            r = [None]*n
        if not isinstance(ig, list) or len(ig) != n:
            ig = [False]*n
        RATINGS[album_id] = {"ratings": r, "ignored": ig}

def _write_back_album(album_id:str):
    """Write the in-memory RATINGS mirror for a single album back inline and persist."""
    alb = ALBUMS[album_id]
    state = RATINGS[album_id]
    alb["ratings"] = state["ratings"]
    alb["ignored"] = state["ignored"]
    _persist_albums_inline()

def rated_album_mean(album_id:str) -> Tuple[float,float,int]:
    """Return (mean, duration-weighted mean, count_used) excluding ignored and None."""
    alb = ALBUMS[album_id]
    ensure_rating_struct(album_id)
    r = RATINGS[album_id]["ratings"]
    ign = RATINGS[album_id]["ignored"]
    vals, weights = [], []
    for i, tr in enumerate(alb["tracks"]):
        if ign[i] or r[i] is None: continue
        vals.append(r[i])
        weights.append(tr["duration_ms"] or 0)
    if not vals: return (None, None, 0)
    mean = sum(vals)/len(vals)
    wsum = sum(weights) or 1
    wmean = sum(v*w for v,w in zip(vals,weights))/wsum
    return (round(mean,2), round(wmean,2), len(vals))

def overall_rank(album_id:str, key="mean") -> Tuple[int,int]:
    """Rank among albums with at least one rated track."""
    scores = []
    for aid in ALBUMS.keys():
        m, wm, c = rated_album_mean(aid)
        if c>0:
            scores.append((aid, wm if key=="wmean" else m))
    scores = [(aid, s) for (aid,s) in scores if s is not None]
    scores.sort(key=lambda x: x[1], reverse=True)
    idx = next((i for i,(aid,_) in enumerate(scores) if aid==album_id), None)
    if idx is None: return (None, len(scores))
    return (idx+1, len(scores))

# Search index
def build_search_index():
    idx = {"albums":[], "artists":{}, "tracks":[]}  # tracks store (text, album_id, ix)
    for a in ALL_ALBUMS:
        idx["albums"].append((a["album_name"], a["album_id"]))
        for art in a["artists"].split(","):
            k = norm_text(art)
            idx["artists"].setdefault(k, set()).add(a["album_id"])
        for i,t in enumerate(a["tracks"]):
            idx["tracks"].append((t["title"], a["album_id"], i))
    return idx
SEARCH_INDEX = build_search_index()

def search_library(query:str) -> List[dict]:
    q = query.strip()
    if not q: return []

    # Year-only search
    if re.fullmatch(r"\d{4}", q):
        year = q
        res = [a for a in ALL_ALBUMS if a.get("year")==year]
        return [{"type":"album", "album_id":a["album_id"], "why":f"Year {year}"} for a in res]

    nq = norm_text(q)

    # Helper scorers
    def score_album(a):
        name = norm_text(a["album_name"])
        s = 0
        if nq in name: s = 120
        else: s = fuzzy_ratio(nq, name)
        return s

    def score_artist(a):
        best = 0
        for art in a["artists"].split(","):
            s = fuzzy_ratio(nq, norm_text(art))
            if nq in norm_text(art): s = 120
            best = max(best, s)
        return best

    def score_tracks(a):
        best = 0
        ix = None
        for i,t in enumerate(a["tracks"]):
            nm = norm_text(t["title"])
            s = fuzzy_ratio(nq, nm)
            if nq in nm: s = 120
            if s>best:
                best, ix = s, i
        return best, ix

    # Score all albums by three channels, with priority Album>Artist>Song
    scored = []
    for a in ALL_ALBUMS:
        sa = score_album(a)
        sar = score_artist(a)
        st, ix = score_tracks(a)
        cat = "album" if sa>=max(sar,st) else ("artist" if sar>=st else "song")
        sc = {"album":sa, "artist":sar, "song":st}[cat]
        scored.append((sc, cat, a, ix))

    # threshold/filter
    scored = [x for x in scored if x[0]>=50]
    scored.sort(key=lambda x: (x[1]!="album", x[1]!="artist", -x[0]))  # album first, then artist, then song, then score

    results = []
    for sc, cat, a, ix in scored[:40]:
        why = {"album":"Album match", "artist":"Artist match", "song":"Song match"}[cat]
        results.append({"type":"album", "album_id":a["album_id"], "why":why, "song_ix":ix, "query":q})
    return results

# --------- Dash App ----------
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True,
           external_scripts=["/assets/drag.js"])
server = app.server
from flask import request, jsonify

def top_nav():
    return dbc.Navbar(
        dbc.Container([
            # Home button - use button with explicit JS navigation
            html.A(
                dbc.Button("Home", color="secondary", className="me-2", id="home-btn"),
                href="/",
                className="nav-home-link direct-link",
                style={"textDecoration":"none", "display":"inline-block", "padding":"0", "margin":"0"}
            ),

            # centered search bar
            dbc.Input(id="search-input", type="text", placeholder="Search albums & artists…",
                      className="search-input", debounce=True),
        ], fluid=True),
        color="dark", dark=True, className="mb-3"
    )

def album_card_line(result):
    a = ALBUMS[result["album_id"]]
    thumb = get_or_make_thumb(Path(a["cover_png"]), (64,64))
    img_uri = image_to_data_uri(thumb)

    # highlight matched part in title/artist if possible
    q = result.get("query","")
    title_children = highlight_match(a["album_name"], q) if q else a["album_name"]
    artist_children = highlight_match(a["artists"], q) if q else a["artists"]

    return dbc.ListGroupItem(
        dbc.Row([
            dbc.Col(html.Img(src=img_uri, style={"height":"48px", "width":"48px", "borderRadius":"6px"}), width="auto"),
            dbc.Col([
                html.Div(title_children, className="search-title"),
                html.Div(artist_children, className="search-sub"),
                html.Div(result["why"], className="why-pill"),
            ]),
            dbc.Col(html.Div(a.get("year") or "", className="search-year"), width=2, style={"textAlign":"right"})
        ], className="align-items-center"),
        action=True,
        href=f"/album/{a['album_id']}",
        className="search-result-item",
        style={"textDecoration":"none"}
    )

UNRATED_ALBUMS = [a for a in ALL_ALBUMS if rated_album_mean(a["album_id"])[2]==0]
random.shuffle(UNRATED_ALBUMS)
UNRATED_PICKS = UNRATED_ALBUMS[:3]

def home_stats(theme):
    # stats
    n_albums = len(ALL_ALBUMS)
    total_tracks = sum(len(a["tracks"]) for a in ALL_ALBUMS)
    avg_len_min = round(np.mean([album_duration_ms(a)/60000 for a in ALL_ALBUMS]), 2) if n_albums else 0
    avg_tracks = round(np.mean([len(a["tracks"]) for a in ALL_ALBUMS]), 2) if n_albums else 0
    total_rated = 0
    for a in ALL_ALBUMS:
        aid = a["album_id"]
        # treat missing as all None
        rs = a.get("ratings") or []
        total_rated += sum(1 for v in rs if v is not None)
    avg_album_rating = np.nan
    means = []
    for aid in ALBUMS.keys():
        m, wm, c = rated_album_mean(aid)
        if c>0 and m is not None: means.append(m)
    if means: avg_album_rating = round(float(np.mean(means)), 2)
    avg_album_rating = "" if np.isnan(avg_album_rating) else avg_album_rating

    # 3 random unrated covers
    #unrated = [a for a in ALL_ALBUMS if rated_album_mean(a["album_id"])[2]==0]
    #random.shuffle(unrated)
    covers = []
    for a in UNRATED_PICKS:
        p = get_or_make_thumb(Path(a["cover_png"]), (256,256))
        covers.append(
            html.A(
                html.Img(src=image_to_data_uri(p), className="home-cover"),
                href=f"/album/{a['album_id']}",
                className="direct-link",
                style={"textDecoration":"none"}
            )
        )

    cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Albums in library"), html.H2(n_albums)])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Total tracks saved"), html.H2(total_tracks)])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Mean album length (min)"), html.H2(avg_len_min)])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Mean #tracks/album"), html.H2(avg_tracks)])), md=3),
    ], className="gy-3")

    cards2 = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Total rated tracks"), html.H2(total_rated)])), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Mean album rating"), html.H2(avg_album_rating or "—")])), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Unrated album picks"), html.Div(covers, className="home-covers")])), md=4),
    ], className="gy-3")

    return html.Div([cards, html.Br(), cards2])

def layout_home():
    theme = DEFAULT_THEME
    return html.Div([
        top_nav(),
        dbc.Container(id="home-body", children=[home_stats(theme)], fluid=True)
    ])

def layout_search(results:List[dict]):
    """Legacy search layout function - modal approach now used instead."""
    search_items = []
    for r in results:
        a = ALBUMS[r["album_id"]]
        thumb = get_or_make_thumb(Path(a["cover_png"]), (64,64))
        img_uri = image_to_data_uri(thumb)

        # highlight matched part in title/artist if possible
        q = r.get("query","")
        title_children = highlight_match(a["album_name"], q) if q else a["album_name"]
        artist_children = highlight_match(a["artists"], q) if q else a["artists"]

        search_items.append(
            html.Div(
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(html.Img(src=img_uri, style={"height":"48px", "width":"48px", "borderRadius":"6px"}), width="auto"),
                            dbc.Col([
                                html.Div(title_children, className="search-title"),
                                html.Div(artist_children, className="search-sub"),
                                html.Div(r["why"], className="why-pill"),
                            ]),
                            dbc.Col(html.Div(a.get("year") or "", className="search-year"), width=2, style={"textAlign":"right"})
                        ], className="align-items-center"),
                    ]),
                    className="search-result-item mb-2",
                    id={"type": "search-result-card", "album_id": a["album_id"]},
                ),
                className="search-result-container",
            )
        )

    return html.Div([
        top_nav(),
        dbc.Container([
            html.Div(search_items, id="search-results") if results
            else html.Div("No matches.", className="muted")
        ], fluid=True)
    ])

def album_graph(album_id:str, theme:Dict):
    alb = ALBUMS[album_id]
    ensure_rating_struct(album_id)
    R = RATINGS[album_id]["ratings"]
    I = RATINGS[album_id]["ignored"]

    x = list(range(1, len(alb["tracks"])+1))
    y = [r if (r is not None and not I[i]) else 0 for i,r in enumerate(R)]

    # labels inside bars (vertical)
    texts = []
    for t in alb["tracks"]:
        s = t["title"]
        s = s[:20]  # truncate
        texts.append(s.upper())

    # Colors: use accent[0] for active, greys for ignored
    main = theme["accents"][0]
    colors = [main if not I[i] else "#4c4f58" for i in range(len(x))]
    lines = ["#8fd4ff" if not I[i] else "#3a3d44" for i in range(len(x))]

    # Overlay scatter points (transparent) to capture click position vertically.
    # This is no longer needed with the new JS-based drag/click handler.

    fig = {
        "data": [
            {
                "type":"bar", "x": x, "y": y, "name":"Ratings",
                "marker": {"color": colors, "line":{"width":2.5, "color": lines}},
                "hoverinfo": "skip",
                "text": texts,
                "textposition": "inside",
                "textangle": 90,
                "insidetextanchor": "start",
                "cliponaxis": False,
                "customdata": [[i, int(I[i])] for i in range(len(x))]
            }
        ],
        "layout": {
            "uirevision": album_id,  # keep state on updates
            "paper_bgcolor": DEFAULT_THEME["bg_superdark"],
            "plot_bgcolor": DEFAULT_THEME["bg_dark"],
            "font": {"color": DEFAULT_THEME["text"]},
            "margin": {"l":50, "r":30, "t":20, "b":80},
            "yaxis": {
                "range":[0,10], "dtick":1, "gridcolor":"#2a2d33", "gridwidth":2,
                "title":"Rating (0–10)",
                "fixedrange": True
            },
            "xaxis": {
                "showgrid": True, "gridcolor":"#22252b", "tickmode":"array",
                "tickvals": x, "ticktext": x, "tickfont":{"size":12},
                "fixedrange": True
            },
            "bargap": 0,
            "dragmode": False
        }
    }
    # Disable all default interactions; we use custom JS for this.
    graph_config = {
        "displayModeBar": False,
        "scrollZoom": False,
        "doubleClick": False,
        "staticPlot": False  # Must be False for events to fire
    }
    return dcc.Graph(
        id={"type":"album-graph","album":album_id},
        figure=fig, config=graph_config, style={"height":"460px"}
    )

def album_ignore_strip(album_id:str, theme:Dict):
    alb = ALBUMS[album_id]
    ensure_rating_struct(album_id)
    I = RATINGS[album_id]["ignored"]
    btns = []
    for i in range(len(alb["tracks"])):
        col = ("secondary" if not I[i] else "dark")
        btns.append(
            dbc.Button(str(i+1), id={"type":"ignore-btn","album":album_id,"ix":i},
                       color=col, className="ignore-pill me-1 mb-1")
        )
    return html.Div(btns, className="d-flex flex-wrap")

def layout_album(album_id:str):
    alb = ALBUMS[album_id]
    theme = theme_from_cover(Path(alb["cover_png"]))
    ensure_rating_struct(album_id)

    cover_big = image_to_data_uri(Path(alb["cover_png"]))

    mean, wmean, c = rated_album_mean(album_id)
    rank, denom = overall_rank(album_id, key="wmean")
    rank_block = html.Div([
        html.Div("Rank", className="muted small"),
        html.Div(f"{rank} / {denom}" if rank else "—", className="rank-box")
    ], style={"textAlign":"right"})

    # right header with means
    stat_cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Mean"), html.H3(mean if mean is not None else "—")])), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Duration-weighted"), html.H3(wmean if wmean is not None else "—")])), md=5),
        dbc.Col(rank_block, md=3),
    ], className="gy-2")

    tracks_table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th("#"), html.Th("Title"), html.Th("Duration")])),
            html.Tbody([
                html.Tr([html.Td(i+1), html.Td(t["title"]), html.Td(ms_to_mmss(t["duration_ms"]))]) 
                for i,t in enumerate(alb["tracks"])
            ])
        ],
        bordered=False, hover=True, size="sm", responsive=True, className="mt-3"
    )

    return html.Div([
        top_nav(),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Img(src=cover_big, className="cover-big"),
                    html.H4(alb["album_name"]), html.Div(alb["artists"], className="muted"),
                    html.Div(alb.get("year",""), className="muted"),
                    html.Hr(), tracks_table
                ], md=3),
                dbc.Col([
                    stat_cards, html.Br(),
                    album_graph(album_id, theme),
                    html.Div("Click or drag across bars to rate (snap 0.1).", className="muted mb-2"),
                    html.Div("Click a number to ignore/include a track:", className="muted"),
                    album_ignore_strip(album_id, theme),
                    dcc.Store(id={"type":"album-state","album":album_id}, data=RATINGS.get(album_id))
                ], md=9)
            ])
        ], fluid=True)
    ])

# --------- Layout Root ---------
# Provide a single global dcc.Location for routing + search state.
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    # Search results modal that will be shown/hidden via callbacks
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Search Results"), close_button=True),
            dbc.ModalBody(id="search-results-content"),
        ],
        id="search-results-modal",
        size="lg",
        is_open=False,
        centered=True,
    ),
    # 'page' will be filled by router callback (now allowed on initial call)
    html.Div(id="page")
])

# --------- Callbacks ---------

@app.callback(
    Output("page","children"),
    Input("url","pathname")
)
def router(pathname:str):
    if not pathname or pathname == "/":
        return [layout_home()]
    if pathname.startswith("/album/"):
        aid = pathname.split("/album/")[1]
        if aid in ALBUMS: return [layout_album(aid)]
    # else treat as search (never reached in our links, but safe)
    return [layout_home()]

# Clear search box when navigating to an album so residual query doesn't interfere
@app.callback(
    Output("search-input", "value", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call=True
)
def clear_search_on_album(pathname):
    if pathname and pathname.startswith("/album/"):
        return ""
    return dash.no_update

# search as-you-type
@app.callback(
    Output("search-results-modal", "is_open"),
    Output("search-results-content", "children"),
    Input("search-input", "value"),
    prevent_initial_call=True
)
def do_search(q):
    # Check if we have a context and if this was triggered by input change
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, dash.no_update
    
    # Show search results on any page; if query cleared, hide modal
    if q is None or q.strip() == "":
        return False, dash.no_update
    
    # Perform search and return results
    results = search_library(q)
    
    if not results:
        return True, html.Div("No matches.", className="muted")
    
    # Create the search result items using our new modal-friendly format
    search_items = []
    for r in results:
        a = ALBUMS[r["album_id"]]
        thumb = get_or_make_thumb(Path(a["cover_png"]), (64,64))
        img_uri = image_to_data_uri(thumb)

        # highlight matched part in title/artist if possible
        q_highlight = r.get("query","")
        title_children = highlight_match(a["album_name"], q_highlight) if q_highlight else a["album_name"]
        artist_children = highlight_match(a["artists"], q_highlight) if q_highlight else a["artists"]

        # Create a card that navigates to the album when clicked
        search_items.append(
            html.Div(
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(html.Img(src=img_uri, style={"height":"48px", "width":"48px", "borderRadius":"6px"}), width="auto"),
                            dbc.Col([
                                html.Div(title_children, className="search-title"),
                                html.Div(artist_children, className="search-sub"),
                                html.Div(r["why"], className="why-pill"),
                            ]),
                            dbc.Col(html.Div(a.get("year") or "", className="search-year"), width=2, style={"textAlign":"right"})
                        ], className="align-items-center"),
                    ]),
                    className="search-result-item mb-2",
                    id={"type": "search-result-card", "album_id": a["album_id"], "index": len(search_items)},
                ),
                className="search-result-container",
            )
        )
    
    return True, html.Div(search_items, id="search-results-list")

# Handle clicks on search result cards
@app.callback(
    Output("url", "pathname"),
    Input({"type": "search-result-card", "album_id": ALL, "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def handle_search_result_click(n_clicks_list):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    # Extract which card was clicked from the trigger ID
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    try:
        album_id = json.loads(triggered_id)["album_id"]
        
        # Get the index of the clicked item
        clicked_index = json.loads(triggered_id).get("index", -1)
        
        # Only check for None if we can identify which item was clicked
        if clicked_index >= 0 and clicked_index < len(n_clicks_list):
            # If this specific item hasn't been clicked (n_clicks is None), don't navigate
            if n_clicks_list[clicked_index] is None:
                #print(f"Ignoring initialization event for index {clicked_index}")
                return dash.no_update
        
        # Navigate to album page (modal will be closed by the URL change callback)
        print(f"Search result clicked - navigating to album {album_id}")
        return f"/album/{album_id}"
    except Exception as e:
        print(f"Error in search click handler: {e}")
        return dash.no_update

# Combined callback to manage search modal visibility
@app.callback(
    Output("search-results-modal", "is_open", allow_duplicate=True),
    [
        Input("search-results-modal", "is_open"),
        Input("url", "pathname")
    ],
    prevent_initial_call=True
)
def manage_search_modal(is_open, pathname):
    # Get triggered component
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    # If URL changed, close the modal
    trigger_id = ctx.triggered[0]["prop_id"]
    if "url.pathname" in trigger_id:
        return False
    
    return dash.no_update


# Toggle ignore buttons
@app.callback(
    Output({"type":"album-state","album":MATCH}, "data", allow_duplicate=True),
    Output({"type":"ignore-btn","album":MATCH,"ix":ALL}, "color"),
    Output({"type":"album-graph","album":MATCH}, "figure", allow_duplicate=True),
    Input({"type":"ignore-btn","album":MATCH,"ix":ALL}, "n_clicks"),
    State({"type":"album-state","album":MATCH}, "data"),
    State({"type":"album-graph","album":MATCH}, "figure"),
    prevent_initial_call=True
)
def toggle_ignore(n_clicks_list, state, fig):
    album_id = callback_context.outputs_list[0]["id"]["album"]
    ensure_rating_struct(album_id)
    alb = ALBUMS[album_id]
    if state is None:
        ensure_rating_struct(album_id)
        state = RATINGS[album_id]
    I = state["ignored"]; R = state["ratings"]
    triggered = [p["prop_id"] for p in callback_context.triggered]
    if triggered and ".n_clicks" in triggered[0]:
        # find which button toggled
        btn_id = dash.callback_context.triggered_id
        i = btn_id["ix"]
        I[i] = not I[i]
    # update colors for buttons
    colors = ["secondary" if not I[i] else "dark" for i in range(len(I))]
    # update fig colors and y-values
    main = DEFAULT_THEME["accents"][0]
    fig["data"][0]["marker"]["color"] = [main if not I[i] else "#4c4f58" for i in range(len(I))]
    fig["data"][0]["marker"]["line"]["color"] = ["#8fd4ff" if not I[i] else "#3a3d44" for i in range(len(I))]
    fig["data"][0]["y"] = [0 if (I[i] or R[i] is None) else R[i] for i in range(len(I))]

    RATINGS[album_id] = {"ratings":R, "ignored":I}
    _write_back_album(album_id)
    return RATINGS[album_id], colors, fig

if __name__ == "__main__":
    # Simple API to update ratings via JS drag interactions
    @server.route('/api/rate', methods=['POST'])
    def api_rate():
        data = request.get_json(force=True) or {}
        album_id = data.get('album_id')
        ratings = data.get('ratings')
        ignored = data.get('ignored')
        if not album_id or album_id not in ALBUMS:
            return jsonify({'ok': False, 'error': 'bad album_id'}), 400
        ensure_rating_struct(album_id)
        cur = RATINGS[album_id]
        if isinstance(ratings, list) and len(ratings)==len(cur['ratings']):
            cur['ratings'] = [None if v is None else float(max(0,min(10,v))) for v in ratings]
        if isinstance(ignored, list) and len(ignored)==len(cur['ignored']):
            cur['ignored'] = [bool(x) for x in ignored]
        RATINGS[album_id] = cur
        _write_back_album(album_id)
        return jsonify({'ok': True})
    app.run_server(debug=True)
