#!/usr/bin/env python3
import os, json, math, random, base64, io, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np 
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
import dash
from dash import Dash, html, dcc, Input, Output, State, MATCH, ALL, callback_context, no_update
import dash_bootstrap_components as dbc
from dash_extensions import EventListener
from dash_svg import Svg, Line, Text, Rect, Polyline, Polygon
from bisect import bisect_right as _br

# --------- Config / Paths ----------
DATA_DIR = Path("output_spotify")         # where albums.json & images/ live
JSON_PATH = DATA_DIR / "albums.json"      # static album metadata
RATINGS_DIR = DATA_DIR / "ratings"        # per-album lightweight rating files
RATINGS_DIR.mkdir(parents=True, exist_ok=True)

THUMB_DIR = DATA_DIR / "thumbs"
THUMB_DIR.mkdir(parents=True, exist_ok=True)

# --- Global Rating Mean Mode ---
# If True, all album mean ratings and ranks use duration-weighted means.
# If False, they use uniform (unweighted) means.
# Toggle this boolean to switch globally.
MEAN_WEIGHTED = True

TIER_SPECS = [
    ("S", "#ff7f7e", 2),  # top ~2 of 40
    ("A", "#ffbf7f", 4),  # next ~4 of 40
    ("B", "#feff7f", 8),  # next ~8 of 40
    ("C", "#beff7f", 12), # mid ~12 of 40
    ("D", "#7fbfff", 8),  # next ~8 of 40
    ("E", "#807fff", 4),  # next ~4 of 40
    ("F", "#ff7ffe", 2),  # bottom ~2 of 40
]
TIER_TOTAL_WEIGHT = sum(weight for _tier, _color, weight in TIER_SPECS)

TIERLIST_ROW_OPTIONS = list(range(1, 21)) + [25, 30, 35, 40, 45, 50]
TIERLIST_ALBUMS_PER_ROW = 16
TIERLIST_USE_LARGE_THUMBS = True

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
    """Derive theme from cover art.

        Logic:
            - Keep all 5 clustered colors (ordered) in th['accents_bar'].
            - Valid colors: luminance > 0.25 (threshold updated from 0.30).
            - Preferred primary: the first VALID color whose saturation > 0.5 (HSV S component).
            - If no valid color has saturation > 0.5, choose the first valid color (most dominant passing luminance).
            - If no colors pass luminance, fall back to DEFAULT_THEME accents (but still expose raw bar colors if present).
            - Build th['accents'] starting with chosen primary followed by remaining distinct colors.
    """
    th = DEFAULT_THEME.copy()
    try:
        img = Image.open(cover_png).convert("RGB")
        arr = np.array(img.resize((128,128)))
        cols = dominant_colors(arr)  # ndarray shape (5,3)
        # Convert all to hex (keep all 5 regardless of brightness)
        bar_hex = ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in cols]
        # Helper saturation (avoid importing colorsys for tiny calc)
        def _sat(carr):
            r,g,b = [x/255.0 for x in carr]
            mx = max(r,g,b); mn = min(r,g,b)
            if mx == 0: return 0.0
            return (mx - mn)/mx
        valid = []  # (index, color_array, hex, sat)
        for idx,c in enumerate(cols):
            if luminance(c) > 0.25:
                valid.append((idx, c, bar_hex[idx], _sat(c)))
        primary = None
        # First try: first valid with saturation > 0.5
        for idx,c,hx,sat in valid:
            if sat > 0.5:
                primary = hx
                break
        # Fallback: first valid regardless of saturation
        if primary is None and valid:
            primary = valid[0][2]
        if primary is None:
            # fallback fully
            th["accents"] = DEFAULT_THEME["accents"]
            th["accents_bar"] = bar_hex if bar_hex else DEFAULT_THEME["accents"]
        else:
            # Build accents starting with primary then remaining unique colors (preserve order but skip duplicate primary)
            remaining = [h for h in bar_hex if h != primary]
            accents = [primary] + remaining
            th["accents"] = accents[:5]
            th["accents_bar"] = bar_hex
    except Exception:
        th["accents"] = DEFAULT_THEME["accents"]
        th["accents_bar"] = DEFAULT_THEME["accents"]
    return th

# --------- Data loading / indexing ----------
albums_doc = load_json(JSON_PATH)
ALBUMS: Dict[str,dict] = {a["album_id"]: a for a in albums_doc.get("albums", [])}
ALL_ALBUMS = list(ALBUMS.values())

RATINGS: Dict[str,dict] = {}

def rating_file(album_id:str) -> Path:
    return RATINGS_DIR / f"{album_id}.json"

def load_album_ratings(album_id:str, n_tracks:int) -> Tuple[list,list]:
    path = rating_file(album_id)
    if path.exists():
        try:
            data = load_json(path)
            r = data.get("ratings")
            ig = data.get("ignored")
            if isinstance(r, list) and isinstance(ig, list) and len(r)==n_tracks and len(ig)==n_tracks:
                return r, ig
        except Exception:
            pass
    # fallback legacy inline
    alb = ALBUMS[album_id]
    r_inline = alb.get("ratings")
    ig_inline = alb.get("ignored")
    if isinstance(r_inline, list) and isinstance(ig_inline, list) and len(r_inline)==n_tracks and len(ig_inline)==n_tracks:
        return r_inline, ig_inline
    return [None]*n_tracks, [False]*n_tracks

def save_album_ratings(album_id:str):
    st = RATINGS[album_id]
    path = rating_file(album_id)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({"ratings": st["ratings"], "ignored": st["ignored"]}, f, ensure_ascii=False)
    tmp.replace(path)

def delete_album_artifacts(album_id:str):
    """Remove album from in-memory structures, delete ratings file, remove entry from albums.json, and delete thumbnails & cover image.

    NOTE: This mutates global ALBUMS / ALL_ALBUMS and writes back updated albums.json.
    """
    global ALL_ALBUMS, ALBUMS
    alb = ALBUMS.get(album_id)
    if not alb:
        return False
    # 1. Delete rating file if exists
    rf = rating_file(album_id)
    if rf.exists():
        try: rf.unlink()
        except Exception: pass
    # 2. Delete thumbnails referencing this cover
    cover_path = Path(alb.get("cover_png",""))
    if cover_path.exists():
        try: cover_path.unlink()
        except Exception: pass
    stem = cover_path.stem
    # remove generated thumbs (pattern: <stem>_WxH.png)
    if THUMB_DIR.exists():
        for p in THUMB_DIR.glob(f"{stem}_*x*.png"):
            try: p.unlink()
            except Exception: pass
    # 3. Remove from albums document and write back
    albums_list = [a for a in ALL_ALBUMS if a.get("album_id") != album_id]
    # Persist to JSON_PATH
    doc = {"albums": albums_list}
    tmp = JSON_PATH.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    tmp.replace(JSON_PATH)
    # 4. Update in-memory maps
    ALBUMS = {a["album_id"]: a for a in albums_list}
    ALL_ALBUMS = list(ALBUMS.values())
    # 5. Remove RATINGS cache if present
    RATINGS.pop(album_id, None)
    return True

def ensure_rating_struct(album_id:str):
    alb = ALBUMS[album_id]
    n = len(alb["tracks"])
    if album_id not in RATINGS:
        r, ig = load_album_ratings(album_id, n)
        if len(r)!=n: r = [None]*n
        if len(ig)!=n: ig = [False]*n
        RATINGS[album_id] = {"ratings": r, "ignored": ig}

def _write_back_album(album_id:str):
    save_album_ratings(album_id)

def rated_album_mean(album_id:str) -> Tuple[float,float,int]:
    """Return (mean, duration-weighted mean, count_used) excluding ignored and None.

    Note: This function returns both variants for completeness, but the rest of the app
    should rely on selected_album_mean() for the single, globally-chosen metric.
    """
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

def selected_album_mean(album_id:str) -> Tuple[float,int]:
    """Return (mean_for_current_mode, count_used) based on MEAN_WEIGHTED toggle."""
    mu, mw, c = rated_album_mean(album_id)
    if c == 0:
        return (None, 0)
    return ((mw if MEAN_WEIGHTED else mu), c)

def overall_rank(album_id:str, key: str = "mean") -> Tuple[int,int]:
    """Rank among albums with at least one rated track using selected mean.

    Note: 'key' is ignored; retained for backward compatibility."""
    scores = []
    for aid in ALBUMS.keys():
        mv, c = selected_album_mean(aid)
        if c>0 and mv is not None:
            scores.append((aid, mv))
    scores.sort(key=lambda x: x[1], reverse=True)
    idx = next((i for i,(aid,_) in enumerate(scores) if aid==album_id), None)
    if idx is None:
        return (None, len(scores))
    return (idx+1, len(scores))

def compute_album_rank_single(target_album_id:str) -> Tuple[int,int]:
    """Compute overall rank for the target album using the globally selected mean.

    Only considers albums with at least one rated (non-ignored) track and with a non-None mean.
    Ranks are among fully rated albums for stability, matching prior behavior.
    Returns (rank_position, total_albums_considered)."""
    rows = _rank_rows()
    total = len(rows)
    if not rows:
        return (None, 0)
    pos = next((i for i,(aid, _) in enumerate(rows) if aid == target_album_id), None)
    if pos is None:
        return (None, total)
    return (pos + 1, total)


def _rank_rows() -> List[Tuple[str, float]]:
    """Return fully rated albums with their selected means sorted descending."""
    rows: List[Tuple[str, float]] = []
    for aid, _alb in ALBUMS.items():
        mean_val, cnt = selected_album_mean(aid)
        if cnt > 0 and mean_val is not None and _is_fully_rated(aid):
            rows.append((aid, mean_val))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def tier_for_rank(rank_index:int, total:int) -> Tuple[Optional[str], Optional[str]]:
    """Return (tier_label, color) for the given zero-based rank index."""
    if total <= 0 or rank_index is None:
        return (None, None)
    # Ensure index stays within bounds
    rank_index = max(0, min(rank_index, total - 1))
    position_fraction = rank_index / total
    cumulative = 0.0
    last_label, last_color = TIER_SPECS[-1][0], TIER_SPECS[-1][1]
    for label, color, weight in TIER_SPECS:
        cumulative += weight / TIER_TOTAL_WEIGHT
        if position_fraction < cumulative:
            return (label, color)
    return (last_label, last_color)


def compute_rank_context(album_id:str) -> Dict[str, object]:
    rows = _rank_rows()
    total = len(rows)
    rank_idx = next((i for i,(aid, _) in enumerate(rows) if aid == album_id), None)
    if rank_idx is None:
        return {"rank": None, "total": total, "tier": None, "tier_color": None}
    tier_label, tier_color = tier_for_rank(rank_idx, total)
    return {
        "rank": rank_idx + 1,
        "total": total,
        "tier": tier_label,
        "tier_color": tier_color
    }


def _render_tier_badge(label:str, color:str, *, size:int, font_size:str, radius:str="8px", class_name:str="tier-badge") -> html.Div:
    style = {
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "width": f"{size}px",
        "height": f"{size}px",
        "borderRadius": radius,
        "border": f"2px solid {color}",
        "color": color,
        "fontWeight": 700,
        "fontSize": font_size,
        "textTransform": "uppercase",
        "letterSpacing": "0.05em",
        "background": "transparent"
    }
    return html.Div(label, className=class_name, style=style)


def render_rank_summary(rank_info:Dict[str, object]) -> html.Div:
    rank = rank_info.get("rank") if rank_info else None
    total = rank_info.get("total") if rank_info else None
    tier = rank_info.get("tier") if rank_info else None
    tier_color = rank_info.get("tier_color") if rank_info else None
    tier_label = tier or "?"
    badge_color = tier_color or "#6f7782"
    rank_text = str(rank) if rank else "?"
    total_text = str(total) if total else "?"
    badge_size = 54
    badge = _render_tier_badge(tier_label, badge_color, size=badge_size, font_size="1.6rem", radius="10px")
    return html.Div([
        badge,
        html.Div([
            html.Span(rank_text, className="rank-number"),
            html.Span("/", className="rank-separator"),
            html.Span(total_text, className="rank-total")
        ], className="rank-row", style={
            "width": f"{badge_size}px",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "fontSize": "0.85rem",
            "fontWeight": "600"
        })
    ], className="rank-summary")

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
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)
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
            # Albums button - navigate to albums table
            html.A(
                dbc.Button("Albums", color="secondary", className="me-2", id="albums-btn"),
                href="/albums",
                className="nav-albums-link direct-link",
                style={"textDecoration":"none", "display":"inline-block", "padding":"0", "margin":"0"}
            ),
            html.A(
                dbc.Button("Tierlist", color="secondary", className="me-2", id="tierlist-btn"),
                href="/tierlist",
                className="nav-tierlist-link direct-link",
                style={"textDecoration":"none", "display":"inline-block", "padding":"0", "margin":"0"}
            ),
            # Settings button (gear icon) - appears next to Home
            html.A(
                dbc.Button("⚙", color="secondary", className="me-2", id="settings-btn", title="Settings"),
                href="/settings",
                className="nav-settings-link direct-link",
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

    year_text = a.get("year") or ""

    return dbc.ListGroupItem(
        dbc.Row([
            dbc.Col(html.Img(src=img_uri, style={"height":"48px", "width":"48px", "borderRadius":"6px"}), width="auto"),
            dbc.Col([
                html.Div(title_children, className="search-title"),
                html.Div(artist_children, className="search-sub"),
                html.Div(result["why"], className="why-pill"),
            ]),
            dbc.Col(
                html.Span(year_text, className="search-year"),
                width=2,
                style={"textAlign":"right"}
            )
        ], className="align-items-center"),
        action=True,
        href=f"/album/{a['album_id']}",
        className="search-result-item",
        style={"textDecoration":"none"}
    )

UNRATED_ALBUMS = [a for a in ALL_ALBUMS if selected_album_mean(a["album_id"])[1]==0]
random.shuffle(UNRATED_ALBUMS)
UNRATED_PICKS = UNRATED_ALBUMS[:4]

def home_stats(theme):
    # stats
    n_albums = len(ALL_ALBUMS)
    total_tracks = sum(len(a["tracks"]) for a in ALL_ALBUMS)
    avg_len_min = round(np.mean([album_duration_ms(a)/60000 for a in ALL_ALBUMS]), 2) if n_albums else 0
    avg_tracks = round(np.mean([len(a["tracks"]) for a in ALL_ALBUMS]), 2) if n_albums else 0
    total_rated = 0
    # Ensure RATINGS loaded for each album on demand
    for a in ALL_ALBUMS:
        aid = a["album_id"]
        ensure_rating_struct(aid)
        rs = RATINGS[aid]["ratings"]
        igs = RATINGS[aid]["ignored"]
        total_rated += sum(1 for v,ig in zip(rs,igs) if v is not None and not ig)
    avg_album_rating = np.nan
    means = []
    for aid in ALBUMS.keys():
        mv, c = selected_album_mean(aid)
        if c>0 and mv is not None:
            means.append(mv)
    if means:
        avg_album_rating = round(float(np.mean(means)), 2)
    avg_album_rating = "" if np.isnan(avg_album_rating) else avg_album_rating

    # Fully rated albums: all tracks either have a rating (not None) or are ignored.
    fully_rated = 0
    for a in ALL_ALBUMS:
        aid = a["album_id"]
        ensure_rating_struct(aid)
        rs = RATINGS[aid]["ratings"]
        igs = RATINGS[aid]["ignored"]
        all_done = True
        for r, ig in zip(rs, igs):
            if (r is None) and (not ig):
                all_done = False
                break
        if all_done:
            fully_rated += 1

    # 3 random unrated covers
    #unrated = [a for a in ALL_ALBUMS if selected_album_mean(a["album_id"])[1]==0]
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

    # Add shuffle button & container for rerolling unrated picks
    shuffle_btn = dbc.Button("Shuffle", id="shuffle-unrated", size="sm", color="secondary", className="ms-2")
    unrated_header = html.Div([
        html.H5("Unrated album picks", className="mb-0"),
        shuffle_btn
    ], className="d-flex align-items-center justify-content-between mb-2")
    cards2 = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Total rated tracks"), html.H2(total_rated)])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Mean album rating"), html.H2(avg_album_rating or "—")])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Fully rated albums"), html.H2(fully_rated)])), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            unrated_header,
            html.Div(id="unrated-picks", children=covers, className="home-covers")
        ])), md=3),
    ], className="gy-3")

    return html.Div([cards, html.Br(), cards2])

def layout_home():
    theme = DEFAULT_THEME
    return html.Div([
        top_nav(),
        dbc.Container(id="home-body", children=[home_stats(theme)], fluid=True)
    ])

def _is_fully_rated(aid:str) -> bool:
    ensure_rating_struct(aid)
    rs = RATINGS[aid]["ratings"]
    ig = RATINGS[aid]["ignored"]
    for r, g in zip(rs, ig):
        if (r is None) and (not g):
            return False
    return True

def _compute_all_album_stats():
    # rows: (album_id, album_name, artists, sel_mean, duration_ms, rated_count, fully)
    rows = []
    for aid, alb in ALBUMS.items():
        mv, c = selected_album_mean(aid)
        dur_ms = sum(t.get("duration_ms") or 0 for t in alb["tracks"])
        fully = _is_fully_rated(aid)
        rows.append((aid, alb["album_name"], alb["artists"], mv, dur_ms, c, fully))
    # Ranks only among fully rated albums with at least one rated non-ignored track and value not None
    rated = [(aid, mv) for (aid,_n,_a,mv,_d,c,fully) in rows if fully and c>0 and mv is not None]
    rated.sort(key=lambda x: x[1], reverse=True)
    rank_map = {aid: i+1 for i,(aid,_) in enumerate(rated)}
    return rows, rank_map

def _format_album_duration(ms:int) -> str:
    secs = ms//1000
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h>0 else f"{m}:{s:02d}"

# Default sort: Rating high -> low.
# Note: For rating column the code interprets asc=True as reverse sort (high-to-low).
ALBUM_TABLE_SORT_DEFAULT = {"col":"rating","asc":True}

def render_albums_table(sort_state:dict):
    sort_col = (sort_state or {}).get("col","album")
    asc = (sort_state or {}).get("asc", True)
    rows, rank_map = _compute_all_album_stats()
    rank_rows = _rank_rows()
    tier_map = {}
    total_ranked = len(rank_rows)
    for idx, (aid_rank, _mean) in enumerate(rank_rows):
        tier_label, _tier_color = tier_for_rank(idx, total_ranked)
        tier_map[aid_rank] = tier_label or "—"
    # rows: (aid, name, artists, mv, dur_ms, c, fully)
    # Sorting key mapping (within group)
    def inner_sort_key(r):
        aid, name, artists, mv, dur_ms, c, fully = r
        if sort_col == "album":
            return name.lower()
        if sort_col == "artist":
            return artists.lower()
        if sort_col == "rating":
            return -1e9 if mv is None else mv
        if sort_col == "dur_total":
            return dur_ms
        return name.lower()
    # For rating columns asc=True => high->low; for others asc=True => low->high
    if sort_col in ("rating",):
        def transform(seq):
            return sorted(seq, key=inner_sort_key, reverse=True if asc else False)
    else:
        def transform(seq):
            return sorted(seq, key=inner_sort_key, reverse=False if asc else True)
    # Separate fully vs partial
    fully_rows = [r for r in rows if r[6]]
    partial_rows = [r for r in rows if not r[6]]
    fully_sorted = transform(fully_rows)
    partial_sorted = transform(partial_rows)
    rows_sorted = fully_sorted + partial_sorted
    # Recompute index numbering based on sorted list
    body_trs = []
    for idx, (aid, name, artists, mv, dur_ms, c, fully) in enumerate(rows_sorted, start=1):
        alb = ALBUMS[aid]
        thumb = get_or_make_thumb(Path(alb["cover_png"]), (64,64))
        img_uri = image_to_data_uri(thumb)
        # Selected rating cell
        if mv is None:
            rating_cell = "—"
        else:
            rk = rank_map.get(aid) if fully else None
            pill = html.Span(str(rk), className="rank-pill") if rk else ""
            rating_cell = html.Span([f"{mv:.2f}", pill])
        tier_label = tier_map.get(aid, "—") if fully else "—"
        body_trs.append(html.Tr([
            html.Td(idx),
            html.Td(html.Div([
                html.Img(src=img_uri, style={"width":"42px","height":"42px","borderRadius":"6px","objectFit":"cover"}),
                html.A(name, href=f"/album/{aid}", className="album-link")
            ], className="album-cell")),
            html.Td(artists),
            html.Td(rating_cell),
            html.Td(tier_label),
            html.Td(_format_album_duration(dur_ms))
        ]))
    # Header arrows
    def hdr(label, col_key):
        arrow = ""
        if sort_col == col_key:
            arrow = " ▲" if asc else " ▼"
        return html.Th(label + arrow, id={"type":"albums-header","col":col_key})
    thead = html.Thead(html.Tr([
        hdr("#","index"),
        hdr("Album","album"),
        hdr("Artist","artist"),
        hdr("Rating","rating"),
        html.Th("Tier"),
        hdr("Duration","dur_total")
    ]))
    table = html.Table([
        thead,
        html.Tbody(body_trs)
    ], className="album-table")
    return table

def layout_albums():
    return html.Div([
        top_nav(),
        dbc.Container([
            html.H3("Albums"),
            dcc.Store(id="albums-sort", data=ALBUM_TABLE_SORT_DEFAULT),
            html.Div(id="albums-table-container", children=[render_albums_table(ALBUM_TABLE_SORT_DEFAULT)])
        ], fluid=True)
    ])

def layout_settings():
    # Single switch: Use uniform mean instead of duration-weighted
    use_uniform_default = [] if MEAN_WEIGHTED else ["uniform"]
    switch = dbc.Checklist(
        options=[{"label": "Use uniform mean (instead of duration-weighted)", "value": "uniform"}],
        value=use_uniform_default,
        id="mean-mode-switch",
        switch=True,
    )
    per_row_dropdown = dcc.Dropdown(
        options=[{"label": str(opt), "value": opt} for opt in TIERLIST_ROW_OPTIONS],
        value=TIERLIST_ALBUMS_PER_ROW,
        id="tierlist-per-row-dropdown",
        clearable=False,
        style={"width": "220px"}
    )
    thumb_switch = dbc.Checklist(
        options=[{"label": "Use 256×256 thumbnails (otherwise 64×64)", "value": "large"}],
        value=["large"] if TIERLIST_USE_LARGE_THUMBS else [],
        id="tierlist-thumb-switch",
        switch=True,
    )
    return html.Div([
        top_nav(),
        dbc.Container([
            html.H3("Settings"),
            html.Div([
                html.Label("Album mean calculation", className="form-label"),
                switch,
                html.Div("Toggle to switch the album rating mean calculation.", className="muted")
            ], className="mb-4"),
            html.Div([
                html.Label("Tierlist albums per row", className="form-label"),
                per_row_dropdown,
                html.Div("Controls how many album thumbnails appear in each tier row.", className="muted")
            ], className="mb-4"),
            html.Div([
                html.Label("Tierlist thumbnail size", className="form-label"),
                thumb_switch,
                html.Div("Switch off to use compact 64×64 thumbnails.", className="muted")
            ], className="mb-4"),
            # hidden sink for side-effect callback
            html.Div(id="settings-status", style={"display":"none"})
        ], fluid=True)
    ])


def layout_tierlist():
    rows = _rank_rows()
    total = len(rows)
    per_row = TIERLIST_ALBUMS_PER_ROW or 1
    per_row = max(1, min(per_row, max(TIERLIST_ROW_OPTIONS)))
    thumb_dim = 256 if TIERLIST_USE_LARGE_THUMBS else 64
    grid_style = {
        "display": "grid",
        "gridTemplateColumns": f"repeat({per_row}, minmax(0, 1fr))",
        "gap": "2px",
        "justifyItems": "center",
        "padding": "3px 4px"
    }

    if total == 0:
        tier_rows = [html.Div("No fully ranked albums yet. Update ranks to populate tiers.", className="muted tierlist-empty")]
    else:
        tier_group_map = {label: [] for label, _color, _weight in TIER_SPECS}
        for idx, (aid, _mean) in enumerate(rows):
            tier_label, _tier_color = tier_for_rank(idx, total)
            if tier_label is None:
                continue
            tier_group_map.setdefault(tier_label, []).append(aid)

        tier_rows = []
        thumb_size = (thumb_dim, thumb_dim)
        for label, color, _weight in TIER_SPECS:
            album_ids = tier_group_map.get(label, [])
            album_cells = []
            for aid in album_ids:
                alb = ALBUMS.get(aid)
                if not alb:
                    continue
                cover_path = Path(alb.get("cover_png", ""))
                if not cover_path.exists():
                    continue
                thumb_path = get_or_make_thumb(cover_path, thumb_size)
                img_uri = image_to_data_uri(thumb_path)
                album_cells.append(
                    html.Div(
                        html.A(
                            html.Img(src=img_uri, className="tierlist-album-image", style={"width": "100%"}),
                            href=f"/album/{aid}",
                            className="tierlist-album-link direct-link",
                            style={"textDecoration": "none"}
                        ),
                        className="tierlist-album-item"
                    )
                )
            if not album_cells:
                content = html.Div("No albums in this tier yet.", className="muted tierlist-empty-tier")
            else:
                content = html.Div(album_cells, className="tierlist-album-grid", style=grid_style)

            tier_rows.append(
                html.Div([
                    html.Div(label, className="tierlist-label"),
                    content
                ], className="tierlist-row", style={"--tier-color": color})
            )

    return html.Div([
        top_nav(),
        dbc.Container([
            html.Div(tier_rows, className="tierlist-rows")
        ], fluid=True)
    ], className="tierlist-page")

# --- Shuffle callback for unrated picks ---
@app.callback(
    Output("unrated-picks", "children"),
    Input("shuffle-unrated", "n_clicks"),
    prevent_initial_call=True
)
def shuffle_unrated(n):
    # Collect current unrated albums (no rated non-ignored tracks)
    unrated = []
    for a in ALL_ALBUMS:
        aid = a["album_id"]
        _m, c = selected_album_mean(aid)
        if c == 0:  # no rated tracks
            unrated.append(a)
    if not unrated:
        return html.Div("All albums have at least one rating.", className="muted")
    random.shuffle(unrated)
    picks = unrated[:4]
    covers = []
    for a in picks:
        p = get_or_make_thumb(Path(a["cover_png"]), (256,256))
        covers.append(html.A(
            html.Img(src=image_to_data_uri(p), className="home-cover"),
            href=f"/album/{a['album_id']}",
            className="direct-link",
            style={"textDecoration":"none"}
        ))
    if len(picks) < 4:
        covers.append(html.Div(f"Only {len(picks)} unrated album(s) remaining.", className="muted mt-2"))
    return covers

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

        year_text = a.get("year") or ""

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
                            dbc.Col(
                                html.Span(year_text, className="search-year"),
                                width=2,
                                style={"textAlign":"right"}
                            )
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

###############################
# SVG Rating Component Helpers
###############################

SVG_W, SVG_H = 840, 368  # widened 20% (700->840) and reduced height 20% (460->368)
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 60, 24, 70, 52
PLOT_W = SVG_W - MARGIN_L - MARGIN_R
PLOT_H = SVG_H - MARGIN_T - MARGIN_B
Y_MIN, Y_MAX = 0.0, 10.0

# We only need pointerdown now (legacy drag interaction removed to avoid needless callbacks).
LISTEN_EVENTS = [
    {"event": "pointerdown",  "props": ["type", "offsetX", "offsetY", "buttons"]}
]

def _build_geometry(widths: list):
    w = np.asarray(widths, float)
    n = len(w)
    lefts = np.empty(n, float)
    rights = np.empty(n, float)
    lefts[0] = -w[0]/2.0
    rights[0] = lefts[0] + w[0]
    for i in range(1,n):
        lefts[i] = rights[i-1]
        rights[i] = lefts[i] + w[i]
    centers = 0.5*(lefts+rights)
    return lefts[0], rights[-1], lefts, rights, centers

def _x_to_px(x, domain_l, domain_r):
    return MARGIN_L + (x - domain_l) * (PLOT_W / (domain_r - domain_l))

def _y_to_px(y):
    return MARGIN_T + (Y_MAX - y) * (PLOT_H / (Y_MAX - Y_MIN))


def _render_svg_children(album_id:str, ratings:list, ignored:list, widths:list, track_titles:list, theme:Dict):
    domain_l, domain_r, LEFTS, RIGHTS, CENTERS = _build_geometry(widths)
    elems = []
    # y grid
    for v in range(int(Y_MIN), int(Y_MAX)+1):
        y = _y_to_px(v)
        elems += [
            Line(x1=MARGIN_L, y1=y, x2=MARGIN_L+PLOT_W, y2=y,
                 stroke="#334955", strokeWidth=1, style={"pointerEvents":"none", "opacity":0.35}),
            Text(str(v), x=MARGIN_L-10, y=y+4, fill="#9fb3bf", textAnchor="end", style={"fontSize":"12px", "pointerEvents":"none"}),
        ]
    # Internal graph title (moved from external controls row)
    # Title positioned above the plot border, offset left to align with plot not y-axis labels
    title_y = MARGIN_T - 52  # raise higher so it clears numeric rating labels
    elems.append(Text(
        "Album Rating Graph",
        x=MARGIN_L + 8,
        y=title_y + 18,  # baseline tweak so visual block centers in reserved band
        fill="#cbd6dd",
        textAnchor="start",
        style={"fontSize":"18px", "fontWeight":"600", "pointerEvents":"none"}
    ))
    # bottom index toggle band
    index_band_top = MARGIN_T + PLOT_H + 4
    index_band_h = 26
    for i in range(len(ratings)):
        left = LEFTS[i]; right = RIGHTS[i]
        px_l = _x_to_px(left, domain_l, domain_r)
        px_r = _x_to_px(right, domain_l, domain_r)
        w = px_r - px_l
        bg = "#3d4246" if not ignored[i] else "#25292c"
        stroke = "#5a6166" if not ignored[i] else "#353a3d"
        elems.append(Rect(x=px_l+1, y=index_band_top, width=w-2, height=index_band_h, fill=bg, stroke=stroke, strokeWidth=1,
                          style={"cursor":"pointer", "rx":4, "ry":4}))
        cx = (px_l+px_r)/2.0
        txt_color = "#dde2e6" if not ignored[i] else "#7e868d"
        elems.append(Text(str(i+1), x=cx, y=index_band_top + index_band_h/2 + 1, fill=txt_color, textAnchor="middle", style={"fontSize":"13px", "pointerEvents":"none", "dominantBaseline":"middle"}))
    # We'll add separators after bars so they render on top.
    accent = theme["accents"][0]
    # independent bar rectangles & numeric labels
    for i,(r,ign) in enumerate(zip(ratings, ignored)):
        # Visual states:
        #  - Ignored: very transparent (alpha ~0.33), treat value at actual rating (or 0) but display very faint.
        #  - Null (unrated): show placeholder bar up to 5.0 with 0.5 alpha accent.
        #  - Normal rated: full accent.
        val = 5.0 if r is None else r
        left = LEFTS[i]; right = RIGHTS[i]
        cx = (left+right)/2.0
        bar_x = _x_to_px(left, domain_l, domain_r)
        bar_x2 = _x_to_px(right, domain_l, domain_r)
        bar_w = bar_x2 - bar_x
        y_val = _y_to_px(val)
        if r is None:
            # constant dark grey with 0.5 opacity placeholder to 5.0
            elems.append(Rect(x=bar_x, y=y_val, width=bar_w, height=(MARGIN_T+PLOT_H - y_val),
                              fill="#44484d", stroke="none", strokeWidth=0, style={"fillOpacity":0.5}))
        else:
            fill_opacity = 0.33 if ign else 1.0
            elems.append(Rect(x=bar_x, y=y_val, width=bar_w, height=(MARGIN_T+PLOT_H - y_val),
                              fill=accent, stroke="none", strokeWidth=0, style={"fillOpacity":fill_opacity}))
        if r is not None and not ign:
            elems.append(Text(f"{r:.1f}", x=_x_to_px(cx, domain_l, domain_r), y=MARGIN_T-14, fill="#cbd6dd", textAnchor="middle", style={"fontSize":"12px", "pointerEvents":"none"}))
    # rotated titles centered horizontally over each bar (after rotation becomes vertical). Our coords: center each bar.
    y_base = MARGIN_T + PLOT_H - 6
    for i, title in enumerate(track_titles):
        # Truncate very long titles: if >30 chars, show first 28 + '..'
        title_disp = title[:28] + ".." if len(title) > 30 else title
        cx = (LEFTS[i]+RIGHTS[i])/2.0
        px_center = _x_to_px(cx, domain_l, domain_r)
        # Align so pivot (bottom middle of bar) is left edge of text after rotation; vertically center glyphs via dominantBaseline
        elems.append(Text(
            title_disp,
            x=px_center,
            y=y_base,
            fill="#000000",
            textAnchor="start",
            style={"fontSize":"16px", "fontWeight":"bold", "pointerEvents":"none", "dominantBaseline":"middle", "alignmentBaseline":"middle"},
            transform=f"rotate(-90 {px_center} {y_base})"
        ))
    # border
    elems.append(Rect(x=MARGIN_L, y=MARGIN_T, width=PLOT_W, height=PLOT_H, fill="none", stroke="#44606f", strokeWidth=2, style={"pointerEvents":"none"}))
    # separators on top (full height)
    for b in RIGHTS[:-1]:
        px = _x_to_px(b, domain_l, domain_r)
        elems.append(Line(x1=px, y1=MARGIN_T, x2=px, y2=MARGIN_T+PLOT_H, stroke="#000000", strokeWidth=2, style={"pointerEvents":"none", "opacity":0.85}))
    # removed polygon & step curve for performance
    return elems

def _point_from_offsets(offsetX, offsetY, widths):
    if offsetX is None or offsetY is None:
        return None
    px, py = float(offsetX), float(offsetY)
    # Allow clicks in plot area for rating & below in index band for ignore toggle
    in_plot = (MARGIN_L <= px <= MARGIN_L + PLOT_W and MARGIN_T <= py <= MARGIN_T + PLOT_H)
    in_index = (MARGIN_L <= px <= MARGIN_L + PLOT_W and MARGIN_T + PLOT_H + 4 <= py <= MARGIN_T + PLOT_H + 30)
    if not (in_plot or in_index):
        return None
    domain_l, domain_r, LEFTS, RIGHTS, CENTERS = _build_geometry(widths)
    idx = _br(RIGHTS.tolist(), domain_l + (px - MARGIN_L) * (domain_r - domain_l) / PLOT_W)
    idx = min(max(idx,0), len(widths)-1)
    if in_index:
        return (idx, None)  # signal ignore toggle
    rely = (py - MARGIN_T)/PLOT_H
    y_val = Y_MAX - rely*(Y_MAX - Y_MIN)
    return idx, float(min(max(y_val, Y_MIN), Y_MAX))

def svg_rating_component(album_id:str, theme:Dict):
    alb = ALBUMS[album_id]
    ensure_rating_struct(album_id)
    ratings = RATINGS[album_id]["ratings"]
    ignored = RATINGS[album_id]["ignored"]
    n = len(ratings)
    # --- Duration-based bar widths (simplified) ---
    # Goal: widths proportional to track duration so horizontal axis reflects time.
    # Only adjustment: enforce a 30s minimum so ultra-short tracks remain clickable.
    # No upper cap, no compression.
    def _compute_duration_widths(tracks):
        secs = []
        for tr in tracks:
            d_ms = tr.get("duration_ms") or 0
            secs.append(d_ms/1000.0)
        if not secs:
            return [1.0]*len(tracks)
        adj = [max(30.0, s) for s in secs]
        total = sum(adj)
        if total <= 0:
            return [1.0]*len(adj)
        # Normalize so average width ~1.0 (sum = n)
        return [a * len(adj) / total for a in adj]
    weighted = _compute_duration_widths(alb["tracks"])
    uniform = [1.0]*n
    return html.Div([
        dcc.Store(id={"type":"album-widths","album":album_id}, data={"mode":"weighted","weighted":weighted,"uniform":uniform}),
        dcc.Store(id={"type":"album-ratings","album":album_id}, data=ratings),
        dcc.Store(id={"type":"album-ignored","album":album_id}, data=ignored),
        dcc.Store(id={"type":"album-theme","album":album_id}, data=theme),
        EventListener(
            id={"type":"album-events","album":album_id},
            events=LISTEN_EVENTS,
            logging=False,
            children=Svg(
                id={"type":"album-svg","album":album_id},
                width=SVG_W, height=SVG_H,
                viewBox=f"0 0 {SVG_W} {SVG_H}",
                style={"background": DEFAULT_THEME["bg_dark"],"userSelect":"none","touchAction":"none","cursor":"crosshair"},
                children=_render_svg_children(album_id, ratings, ignored, weighted, [t["title"] for t in alb["tracks"]], theme)
            )
        )
    ])


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

def accent_color_bar(theme:Dict):
    """Return a thick horizontal bar composed of 5 cover-derived colors.
    Uses theme['accents_bar'] (raw clustered colors) falling back gracefully."""
    cols = (theme.get("accents_bar") or theme.get("accents") or DEFAULT_THEME["accents"])[:5]
    segs = []
    width_pct = 100/len(cols) if cols else 100
    for i,c in enumerate(cols):
        segs.append(html.Div(style={
            "flex":"1 1 auto",
            "background":c,
            "height":"28px"  # ~8x default hr thickness (approx 3-4px originally)
        }))
    return html.Div(segs, className="accent-bar", style={
        "display":"flex",
        "width":"100%",
        "borderRadius":"6px",
        "overflow":"hidden",
        "margin":"18px 0 14px 0"
    })

def layout_album(album_id:str):
    alb = ALBUMS[album_id]
    theme = theme_from_cover(Path(alb["cover_png"]))
    ensure_rating_struct(album_id)

    cover_big = image_to_data_uri(Path(alb["cover_png"]))
    # Compute total album duration in ms
    total_ms = sum(t.get("duration_ms") or 0 for t in alb["tracks"])
    secs = total_ms // 1000
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        album_duration_str = f"{h}:{m:02d}:{s:02d}"
    else:
        album_duration_str = f"{m}:{s:02d}"

    sel_mean, c = selected_album_mean(album_id)
    rank_info = compute_rank_context(album_id)
    rank_summary = render_rank_summary(rank_info)
    mode_label = "Mean (Duration)" if MEAN_WEIGHTED else "Mean (Uniform)"
    stat_cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.Div([
                    html.H6(mode_label),
                    html.H3(sel_mean if sel_mean is not None else "—", id={"type":"mean-selected","album":album_id})
                ], className="flex-grow-1"),
                html.Div(rank_summary, id={"type":"rank-display","album":album_id}, className="rank-display ms-auto")
            ], className="d-flex align-items-center justify-content-between flex-wrap gap-3")
        ])), md=6, lg=5),
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
                    html.Div(album_duration_str, className="muted"),
                    accent_color_bar(theme),
                    tracks_table,
                    # Delete album button relocated here to be less visually intrusive
                    html.Div([
                        dbc.Button(
                            "Delete Album",
                            id="delete-album-btn",
                            color="danger",
                            outline=True,
                            size="sm",
                            className="mt-2"
                        ),
                        dbc.Modal([
                            dbc.ModalHeader(dbc.ModalTitle("Confirm Deletion")),
                            dbc.ModalBody([
                                html.P("This will permanently remove this album, its ratings file, and cached images from the library. This action cannot be undone."),
                                html.P("Are you sure you want to continue?", className="fw-bold")
                            ]),
                            dbc.ModalFooter([
                                dbc.Button("Cancel", id="delete-cancel-btn", className="me-2"),
                                dbc.Button("Delete Album", id="delete-confirm-btn", color="danger")
                            ])
                        ], id="delete-modal", is_open=False, centered=True, backdrop="static"),
                    ]),
                    # Store current album id for deletion callbacks
                    dcc.Store(id="current-album-id", data=album_id)
                ], md=3),
                dbc.Col([
                    stat_cards, html.Br(),
                    # Controls row above graph
                    html.Div([
                        dbc.Button("Toggle Bar Widths", id={"type":"width-toggle","album":album_id}, color="secondary", size="sm", className="me-2"),
                        dbc.Button("Update Ranks", id={"type":"update-ranks","album":album_id}, color="secondary", size="sm", className="me-2"),
                        # Listen button tries native Spotify first (handled via assets/listen.js) with web fallback.
                        html.A(
                            dbc.Button(
                                "Listen",
                                id={"type":"listen-album-btn","album":album_id},
                                color="secondary",
                                size="sm",
                                className="me-2",
                                title="Open in Spotify"
                            ),
                            href=f"spotify:album:{album_id}",
                            className="direct-link",
                            style={"textDecoration":"none"},
                            **{"data-spotify-uri": f"spotify:album:{album_id}",
                               "data-spotify-web": f"https://open.spotify.com/album/{album_id}"}
                        ),
                        dbc.Button("Reset", id={"type":"reset-ratings","album":album_id}, color="warning", outline=True, size="sm", className="me-2"),
                    ], className="d-flex align-items-center flex-wrap gap-2 mb-2"),
                    svg_rating_component(album_id, theme),
                    dcc.Store(id={"type":"album-state","album":album_id}, data=RATINGS.get(album_id)),
                    dcc.Store(id={"type":"reset-snapshot","album":album_id})
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
    if pathname == "/albums":
        return [layout_albums()]
    if pathname == "/tierlist":
        return [layout_tierlist()]
    if pathname == "/settings":
        return [layout_settings()]
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


@app.callback(
    Output("albums-sort","data"),
    Output("albums-table-container","children"),
    Input({"type":"albums-header","col":ALL}, "n_clicks"),
    State("albums-sort","data"),
    prevent_initial_call=True
)
def sort_albums(header_clicks, sort_state):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    trig = ctx.triggered[0]["prop_id"].split(".")[0]
    try:
        obj = json.loads(trig)
    except Exception:
        return dash.no_update, dash.no_update
    col = obj.get("col")
    if col is None:
        return dash.no_update, dash.no_update
    # index column is not sortable (rely on natural order) -> map to album for stability
    if col == "index":
        col = "album"
    prev_col = (sort_state or {}).get("col","album")
    prev_asc = (sort_state or {}).get("asc", True)
    if col == prev_col:
        # toggle
        new_state = {"col": col, "asc": not prev_asc}
    else:
        # First click on a new column: set asc flag so that ratings show high->low, others low->high.
        if col in ("rating",):
            new_state = {"col": col, "asc": True}  # asc=True triggers reverse for ratings (high-first)
        else:
            new_state = {"col": col, "asc": True}  # standard ascending for text/duration
    table = render_albums_table(new_state)
    return new_state, [table]

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


###############################
# New Callbacks for SVG rating
###############################

# Settings callback: update global mean mode from switch
@app.callback(
    Output("settings-status", "children"),
    Input("mean-mode-switch", "value"),
    Input("tierlist-per-row-dropdown", "value"),
    Input("tierlist-thumb-switch", "value"),
    prevent_initial_call=False
)
def apply_settings(values, per_row, thumb_values):
    global MEAN_WEIGHTED, TIERLIST_ALBUMS_PER_ROW, TIERLIST_USE_LARGE_THUMBS
    # values is a list containing 'uniform' when switch is on
    want_uniform = isinstance(values, list) and ("uniform" in values)
    MEAN_WEIGHTED = not want_uniform

    if per_row is not None:
        try:
            per_row_int = int(per_row)
            if per_row_int in TIERLIST_ROW_OPTIONS:
                TIERLIST_ALBUMS_PER_ROW = per_row_int
        except (TypeError, ValueError):
            pass

    use_large = isinstance(thumb_values, list) and ("large" in thumb_values)
    TIERLIST_USE_LARGE_THUMBS = use_large

    return f"mode={'uniform' if want_uniform else 'weighted'}, per_row={TIERLIST_ALBUMS_PER_ROW}, thumb={'256' if use_large else '64'}"

@app.callback(
    Output({"type":"album-widths","album":MATCH}, "data"),
    Input({"type":"width-toggle","album":MATCH}, "n_clicks"),
    State({"type":"album-widths","album":MATCH}, "data"),
    prevent_initial_call=True
)
def toggle_width_mode(n_clicks, data):
    if not callback_context.triggered:
        return dash.no_update
    if not data:
        return dash.no_update
    mode = data.get("mode","weighted")
    data["mode"] = "uniform" if mode == "weighted" else "weighted"
    return data

@app.callback(
    Output({"type":"album-ignored","album":MATCH}, "data", allow_duplicate=True),
    Output({"type":"ignore-btn","album":MATCH,"ix":ALL}, "color"),
    Output({"type":"album-svg","album":MATCH}, "children", allow_duplicate=True),
    Input({"type":"ignore-btn","album":MATCH,"ix":ALL}, "n_clicks"),
    State({"type":"album-ratings","album":MATCH}, "data"),
    State({"type":"album-ignored","album":MATCH}, "data"),
    State({"type":"album-widths","album":MATCH}, "data"),
    State({"type":"album-theme","album":MATCH}, "data"),
    prevent_initial_call=True
)
def toggle_ignore(n_clicks_list, ratings_state, ignored_state, widths_state, theme_state):
    if not callback_context.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    trig = dash.callback_context.triggered_id
    album_id = trig.get("album") if isinstance(trig, dict) else None
    if album_id is None:
        return dash.no_update, dash.no_update, dash.no_update
    ensure_rating_struct(album_id)
    alb = ALBUMS[album_id]
    # Use authoritative RATINGS structure for ratings, merge with store state to be safe.
    current = RATINGS[album_id]
    R = list(current["ratings"]) if ratings_state is None else list(ratings_state)
    I = list(current["ignored"]) if ignored_state is None else list(ignored_state)
    if trig and "ix" in trig:
        i = trig["ix"]
        if 0 <= i < len(I):
            I[i] = not I[i]
    RATINGS[album_id]["ratings"] = R  # preserve existing ratings
    RATINGS[album_id]["ignored"] = I
    _write_back_album(album_id)
    colors = ["secondary" if not I[i] else "dark" for i in range(len(I))]
    mode = widths_state.get("mode","weighted") if widths_state else "weighted"
    widths = widths_state.get(mode, [1.0]*len(R)) if widths_state else [1.0]*len(R)
    theme = theme_state or theme_from_cover(Path(alb["cover_png"]))
    children = _render_svg_children(album_id, R, I, widths, [t["title"] for t in alb["tracks"]], theme)
    return I, colors, children

@app.callback(
    Output({"type":"album-ratings","album":MATCH}, "data"),
    Output({"type":"album-ignored","album":MATCH}, "data", allow_duplicate=True),
    Output({"type":"album-svg","album":MATCH}, "children", allow_duplicate=True),
    Input({"type":"album-events","album":MATCH}, "n_events"),
    State({"type":"album-events","album":MATCH}, "event"),
    State({"type":"album-ratings","album":MATCH}, "data"),
    State({"type":"album-ignored","album":MATCH}, "data"),
    State({"type":"album-widths","album":MATCH}, "data"),
    State({"type":"album-theme","album":MATCH}, "data"),
    prevent_initial_call=True
)
def handle_pointer(n_events, evt, ratings, ignored, widths_state, theme_state):
    if not evt:
        return ratings, ignored, no_update
    trig = dash.callback_context.triggered_id
    album_id = trig.get("album") if isinstance(trig, dict) else None
    if album_id is None:
        return ratings, ignored, no_update
    widths_mode = widths_state.get("mode","weighted") if widths_state else "weighted"
    widths = widths_state.get(widths_mode, [1.0]*len(ratings)) if widths_state else [1.0]*len(ratings)
    res = _point_from_offsets(evt.get("offsetX"), evt.get("offsetY"), widths)
    if evt.get("type") == "pointerdown" and res is not None:
        idx, val = res
        ensure_rating_struct(album_id)
        ratings = list(ratings)
        ignored = list(ignored)
        if val is None:
            # toggle ignore
            if 0 <= idx < len(ignored):
                ignored[idx] = not ignored[idx]
                RATINGS[album_id]["ignored"] = ignored
        else:
            ratings[idx] = round(val,1)
            RATINGS[album_id]["ratings"] = ratings
        _write_back_album(album_id)
        alb = ALBUMS[album_id]
        theme = theme_state or theme_from_cover(Path(alb["cover_png"]))
        children = _render_svg_children(album_id, ratings, ignored, widths, [t["title"] for t in alb["tracks"]], theme)
        return ratings, ignored, children
    return ratings, ignored, no_update

@app.callback(
    Output({"type":"album-svg","album":MATCH}, "children", allow_duplicate=True),
    Input({"type":"album-widths","album":MATCH}, "data"),
    State({"type":"album-ratings","album":MATCH}, "data"),
    State({"type":"album-ignored","album":MATCH}, "data"),
    State({"type":"album-theme","album":MATCH}, "data"),
    prevent_initial_call=True
)
def update_widths(width_state, ratings, ignored, theme_state):
    trig = dash.callback_context.triggered_id
    album_id = trig.get("album") if isinstance(trig, dict) else None
    if album_id is None:
        return no_update
    alb = ALBUMS[album_id]
    mode = width_state.get("mode","weighted") if width_state else "weighted"
    widths = width_state.get(mode, [1.0]*len(ratings)) if width_state else [1.0]*len(ratings)
    theme = theme_state or theme_from_cover(Path(alb["cover_png"]))
    children = _render_svg_children(album_id, ratings, ignored, widths, [t["title"] for t in alb["tracks"]], theme)
    return children

@app.callback(
    Output({"type":"rank-display","album":MATCH}, "children"),
    Output({"type":"mean-selected","album":MATCH}, "children"),
    Input({"type":"update-ranks","album":MATCH}, "n_clicks"),
    prevent_initial_call=True
)
def update_ranks(n_clicks):
    trig = dash.callback_context.triggered_id
    album_id = trig.get("album") if isinstance(trig, dict) else None
    if album_id is None:
        return dash.no_update, dash.no_update
    # Recompute stats
    sel_mean, c = selected_album_mean(album_id)
    sel_disp = sel_mean if sel_mean is not None else "—"
    rank_info = compute_rank_context(album_id)
    return render_rank_summary(rank_info), sel_disp

@app.callback(
    Output("delete-modal", "is_open"),
    Input("delete-album-btn", "n_clicks"),
    Input("delete-cancel-btn", "n_clicks"),
    State("delete-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_delete_modal_static(open_click, cancel_click, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    trig = ctx.triggered_id
    if trig == "delete-album-btn":
        return True
    if trig == "delete-cancel-btn":
        return False
    return dash.no_update

@app.callback(
    Output("url","pathname", allow_duplicate=True),
    Output("delete-modal", "is_open", allow_duplicate=True),
    Input("delete-confirm-btn", "n_clicks"),
    State("current-album-id", "data"),
    prevent_initial_call=True
)
def confirm_delete_static(n, album_id):
    if not n or not album_id:
        return dash.no_update, dash.no_update
    ok = delete_album_artifacts(album_id)
    if ok:
        return "/", False
    return dash.no_update, False


@app.callback(
    Output({"type":"reset-ratings","album":MATCH}, "children"),
    Output({"type":"reset-snapshot","album":MATCH}, "data"),
    Output({"type":"album-ratings","album":MATCH}, "data", allow_duplicate=True),
    Output({"type":"album-ignored","album":MATCH}, "data", allow_duplicate=True),
    Output({"type":"album-svg","album":MATCH}, "children", allow_duplicate=True),
    Input({"type":"reset-ratings","album":MATCH}, "n_clicks"),
    State({"type":"reset-ratings","album":MATCH}, "children"),
    State({"type":"album-ratings","album":MATCH}, "data"),
    State({"type":"album-ignored","album":MATCH}, "data"),
    State({"type":"album-widths","album":MATCH}, "data"),
    State({"type":"album-theme","album":MATCH}, "data"),
    prevent_initial_call=True
)
def reset_or_revert(n_clicks, current_label, ratings, ignored, widths_state, theme_state):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    trig = dash.callback_context.triggered_id
    album_id = trig.get("album") if isinstance(trig, dict) else None
    if album_id is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    ensure_rating_struct(album_id)
    # Acquire album metadata
    alb = ALBUMS[album_id]
    widths_mode = widths_state.get("mode","weighted") if widths_state else "weighted"
    widths = widths_state.get(widths_mode, [1.0]*len(ratings)) if widths_state else [1.0]*len(ratings)
    theme = theme_state or theme_from_cover(Path(alb["cover_png"]))

    if current_label == "Reset":
        # Snapshot original state PLUS the cleared state reference so we can distinguish initial cleared view from user edits.
        new_ratings = [None for _ in ratings]
        new_ignored = [False for _ in ignored]
        snapshot = {
            "original_ratings": list(ratings),
            "original_ignored": list(ignored),
            "cleared_ratings": list(new_ratings),
            "cleared_ignored": list(new_ignored),
            "ts": datetime.utcnow().isoformat()
        }
        # Persist cleared state
        RATINGS[album_id]["ratings"] = new_ratings
        RATINGS[album_id]["ignored"] = new_ignored
        _write_back_album(album_id)
        # Keep server-side snapshot for Revert action
        _RESET_SNAPSHOTS[album_id] = snapshot
        children = _render_svg_children(album_id, new_ratings, new_ignored, widths, [t["title"] for t in alb["tracks"]], theme)
        return "Revert", snapshot, new_ratings, new_ignored, children
    else:  # Revert
        snap = _RESET_SNAPSHOTS.get(album_id)
        if not snap:
            return "Reset", dash.no_update, ratings, ignored, dash.no_update
        # TTL check (5s)
        try:
            ts = datetime.fromisoformat(snap.get("ts"))
            if (datetime.utcnow() - ts).total_seconds() > 5:
                _RESET_SNAPSHOTS.pop(album_id, None)
                return "Reset", None, ratings, ignored, dash.no_update
        except Exception:
            _RESET_SNAPSHOTS.pop(album_id, None)
            return "Reset", None, ratings, ignored, dash.no_update
        restored_ratings = list(snap["original_ratings"])
        restored_ignored = list(snap["original_ignored"])
        RATINGS[album_id]["ratings"] = restored_ratings
        RATINGS[album_id]["ignored"] = restored_ignored
        _write_back_album(album_id)
        children = _render_svg_children(album_id, restored_ratings, restored_ignored, widths, [t["title"] for t in alb["tracks"]], theme)
        _RESET_SNAPSHOTS.pop(album_id, None)
        return "Reset", None, restored_ratings, restored_ignored, children

# Ephemeral map for snapshots keyed by album id
_RESET_SNAPSHOTS = {}

# Modify reset_or_revert to handle Revert path with global snapshot
def _reset_or_revert_impl(n_clicks, current_label, ratings, ignored, widths_state, theme_state):
    trig = dash.callback_context.triggered_id
    album_id = trig.get("album") if isinstance(trig, dict) else None
    if album_id is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    alb = ALBUMS[album_id]
    widths_mode = widths_state.get("mode","weighted") if widths_state else "weighted"
    widths = widths_state.get(widths_mode, [1.0]*len(ratings)) if widths_state else [1.0]*len(ratings)
    theme = theme_state or theme_from_cover(Path(alb["cover_png"]))
    ensure_rating_struct(album_id)
    if current_label == "Reset":
        snap = {"ratings": list(ratings), "ignored": list(ignored), "ts": datetime.utcnow().isoformat()}
        _RESET_SNAPSHOTS[album_id] = snap
        new_ratings = [None for _ in ratings]
        new_ignored = [False for _ in ignored]
        RATINGS[album_id]["ratings"] = new_ratings
        RATINGS[album_id]["ignored"] = new_ignored
        _write_back_album(album_id)
        children = _render_svg_children(album_id, new_ratings, new_ignored, widths, [t["title"] for t in alb["tracks"]], theme)
        return "Revert", snap, new_ratings, new_ignored, children
    else:  # Revert
        snap = _RESET_SNAPSHOTS.get(album_id)
        if not snap:
            return "Reset", dash.no_update, ratings, ignored, dash.no_update
        restored_ratings = list(snap["ratings"])
        restored_ignored = list(snap["ignored"])
        RATINGS[album_id]["ratings"] = restored_ratings
        RATINGS[album_id]["ignored"] = restored_ignored
        _write_back_album(album_id)
        children = _render_svg_children(album_id, restored_ratings, restored_ignored, widths, [t["title"] for t in alb["tracks"]], theme)
        # Clear snapshot after revert
        _RESET_SNAPSHOTS.pop(album_id, None)
        return "Reset", None, restored_ratings, restored_ignored, children

# Rebind the decorated function to wrapper to include revert logic
reset_or_revert.__wrapped__ = _reset_or_revert_impl  # type: ignore
reset_or_revert = _reset_or_revert_impl  # type: ignore

@app.callback(
    Output({"type":"reset-ratings","album":MATCH}, "children", allow_duplicate=True),
    Input({"type":"album-ratings","album":MATCH}, "data"),
    Input({"type":"album-ignored","album":MATCH}, "data"),
    State({"type":"reset-ratings","album":MATCH}, "children"),
    prevent_initial_call=True
)
def guard_label_after_external_change(current_ratings, current_ignored, label):
    """Keep 'Revert' label active ONLY while:
       - A snapshot exists
       - TTL (5s) not expired
       - User has not modified ratings/ignored beyond the initial cleared state
    Otherwise revert label to 'Reset'.
    """
    if label != "Revert":
        return dash.no_update
    trig = dash.callback_context.triggered_id
    album_id = trig.get("album") if isinstance(trig, dict) else None
    if album_id is None:
        return dash.no_update
    snap = _RESET_SNAPSHOTS.get(album_id)
    if not snap:
        return "Reset"
    # TTL check
    try:
        ts = datetime.fromisoformat(snap.get("ts"))
        if (datetime.utcnow() - ts).total_seconds() > 5:
            _RESET_SNAPSHOTS.pop(album_id, None)
            return "Reset"
    except Exception:
        _RESET_SNAPSHOTS.pop(album_id, None)
        return "Reset"
    # If user has begun editing (ratings differ from cleared_ratings OR ignored differ from cleared_ignored) invalidate
    if current_ratings != snap.get("cleared_ratings") or current_ignored != snap.get("cleared_ignored"):
        _RESET_SNAPSHOTS.pop(album_id, None)
        return "Reset"
    return dash.no_update

## Removed invalidate_snapshot_on_pointer: no drag/continuous pointer events remain.

if __name__ == "__main__":
    app.run_server(debug=True)
