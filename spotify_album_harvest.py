#!/usr/bin/env python3
"""
Harvest album metadata & covers from a Spotify playlist/album/track URL.

Outputs:
- images/<album_slug>.png
- albums.json  (with all fields below)

Notes:
- Spotify does not expose per-track "play counts" via API. We include
  'popularity' (0..100) instead. If real play counts are ever exposed,
  add them where indicated below.
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

import requests
from slugify import slugify
from PIL import Image
from io import BytesIO

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.cache_handler import MemoryCacheHandler
from dotenv import load_dotenv
# --------------- Helpers ----------------

def _load_existing_json(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def ms_to_mmss(ms: int) -> str:
    s = ms // 1000
    m, s = divmod(s, 60)
    return f"{m}:{s:02d}"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def download_image_to_png(url: str, out_path: Path) -> None:
    """Download image (usually JPEG) and save as PNG."""
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    img.save(out_path, format="PNG")

def pick_largest_image(images: List[Dict]) -> str:
    """Spotify returns multiple sizes; pick the largest (first is usually largest)."""
    if not images:
        return ""
    # images are typically sorted largest->smallest
    return images[0]["url"]

def pick_smallest_image(images: List[Dict]) -> str:
    if not images:
        return ""
    return images[-1]["url"]

def unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

# --------------- Core ----------------

class SpotifyHarvester:
    def __init__(self):
        # Use client credentials (no user auth)
        load_dotenv()
        cid = os.getenv("SPOTIFY_CLIENT_ID")
        secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        if not cid or not secret:
            raise RuntimeError("Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
        auth = SpotifyClientCredentials(client_id=cid, client_secret=secret, cache_handler=MemoryCacheHandler())
        self.sp = spotipy.Spotify(auth_manager=auth, requests_timeout=20, retries=3, status_forcelist=(429, 500, 502, 503, 504))

    # ---- URL parsing & expansion ----
    def parse_url(self, url_or_uri: str) -> Tuple[str, str]:
        """
        Returns (kind, id) where kind in {"album","playlist","track"}.
        Accepts full URLs like https://open.spotify.com/album/... or spotify:album:...
        """
        u = url_or_uri.strip()
        parts = u.split("/")
        if "open.spotify.com" in u:
            # e.g., https://open.spotify.com/album/{id}?si=...
            kind = parts[3].split("?")[0]
            sid = parts[4].split("?")[0]
        elif u.startswith("spotify:"):
            # e.g., spotify:album:{id}
            _, kind, sid = u.split(":")
        else:
            raise ValueError("Unrecognized Spotify URL/URI.")
        if kind not in {"album", "playlist", "track"}:
            raise ValueError(f"Unsupported link type: {kind}")
        return kind, sid

    def expand_tracks_from_link(self, kind: str, sid: str) -> List[Dict]:
        """
        Returns list of track objects (simplified) containing album + track info.
        For albums: returns its tracks.
        For playlists: returns all items' tracks.
        For tracks: returns single track.
        """
        sp = self.sp
        tracks = []

        if kind == "album":
            # Get album tracks (paginated)
            album = sp.album(sid)
            offset = 0
            while True:
                page = sp.album_tracks(sid, limit=50, offset=offset)
                for t in page["items"]:
                    # attach album info for ease
                    t["album"] = album
                    tracks.append(t)
                if page["next"]:
                    offset += 50
                else:
                    break

        elif kind == "playlist":
            # Walk playlist items
            offset = 0
            while True:
                page = sp.playlist_items(sid, additional_types=("track",), offset=offset, limit=100)
                for it in page["items"]:
                    t = it.get("track")
                    if t and t.get("id"):
                        tracks.append(t)
                if page["next"]:
                    offset += 100
                else:
                    break

        elif kind == "track":
            t = sp.track(sid)
            tracks.append(t)

        return tracks

    # ---- Album harvesting ----
    def harvest_albums(self, tracks: List[Dict], image_dir: Path) -> Dict[str, Dict]:
        """
        Given a bunch of tracks, collect unique album IDs and extract album-level info.
        Saves cover art PNGs.
        Returns dict keyed by album_id with structured metadata.
        """
        sp = self.sp
        ensure_dir(image_dir)

        album_ids: List[str] = []
        for t in tracks:
            album = t["album"]
            album_ids.append(album["id"])
        album_ids = unique(album_ids)

        albums_meta: Dict[str, Dict] = {}

        # Batch fetch albums in chunks of 20
        for i in range(0, len(album_ids), 20):
            batch_ids = album_ids[i:i+20]
            resp = sp.albums(batch_ids)
            for album in resp["albums"]:
                if not album or not album.get("id"):
                    continue

                album_id = album["id"]
                album_name = album["name"]
                artists = "; ".join([a["name"] for a in album.get("artists", [])]) or "Unknown"
                release_date = album.get("release_date", "")  # 'YYYY-MM-DD' or 'YYYY'
                year = release_date.split("-")[0] if release_date else ""

                # Cover art
                cover_url = pick_largest_image(album.get("images", []))
                thumb_url = pick_smallest_image(album.get("images", []))
                album_slug = slugify(f"{artists} - {album_name} ({year or 'unknown'})")
                png_path = image_dir / f"{album_slug}.png"
                if cover_url:
                    try:
                        download_image_to_png(cover_url, png_path)
                    except Exception as e:
                        print(f"[warn] Failed to download cover for {album_name}: {e}")

                # Fetch album tracks (full order, durations)
                album_tracks = []
                offset = 0
                while True:
                    at_page = sp.album_tracks(album_id, limit=50, offset=offset)
                    for t in at_page["items"]:
                        album_tracks.append({
                            "track_number": t.get("track_number"),
                            "disc_number": t.get("disc_number"),
                            "title": t.get("name"),
                            "duration_ms": t.get("duration_ms"),
                            "duration": ms_to_mmss(t.get("duration_ms") or 0),
                            "id": t.get("id"),
                            # popularity needs per-track fetch if desired; optional & slower:
                            # "popularity": sp.track(t["id"])["popularity"] if t.get("id") else None
                        })
                    if at_page["next"]:
                        offset += 50
                    else:
                        break

                # Sort by disc then track number
                album_tracks.sort(key=lambda x: (x["disc_number"] or 1, x["track_number"] or 0))

                albums_meta[album_id] = {
                    "album_id": album_id,
                    "album_name": album_name,
                    "artists": artists,
                    "year": year,
                    "release_date": release_date,
                    "cover_png": str(png_path),
                    "cover_url": cover_url,
                    "thumbnail_url": thumb_url,
                    "tracks": album_tracks,
                    # 'plays' not available; include aggregate popularity if you want:
                    # "album_popularity": album.get("popularity", None),
                    # "note": "Spotify API does not provide per-track play counts."
                }

                # Be gentle with rate limits
                time.sleep(0.05)

        return albums_meta

# --------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Harvest album metadata & covers from a Spotify link.")
    ap.add_argument("--url", help="Spotify playlist/album/track URL or URI",
                    default="https://open.spotify.com/playlist/0r3EAAP07Wc5JOSHRNhHmF?si=59e1f0d895d84f7e")
    ap.add_argument("--out", default="output_spotify", help="Output folder")
    ap.add_argument("--json_name", default="albums.json", help="Name of JSON file to write")
    ap.add_argument("--fresh", action="store_true",
                help="Ignore existing albums.json and rebuild from scratch.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    ensure_dir(images_dir)

    harvester = SpotifyHarvester()
    kind, sid = harvester.parse_url(args.url)
    print(f"[info] Link type: {kind}, id: {sid}")

    tracks = harvester.expand_tracks_from_link(kind, sid)
    print(f"[info] Collected {len(tracks)} track(s) from link.")

    albums_meta = harvester.harvest_albums(tracks, images_dir)

    # Map back which songs led us to which albums (optional trace)
    song_to_album = []
    for t in tracks:
        if not t or not t.get("id"):
            continue
        album = t.get("album", {})
        song_to_album.append({
            "track_id": t["id"],
            "track_title": t.get("name"),
            "album_id": album.get("id"),
            "album_name": album.get("name")
        })

    output = {
        "source_link": args.url,
        "harvested_albums_count": len(albums_meta),
        "albums": list(albums_meta.values()),
        "trace_tracks_to_albums": song_to_album
    }

    ensure_dir(out_dir)
    json_path = out_dir / args.json_name
    existing = {} if getattr(args, "fresh", False) else _load_existing_json(json_path)

    if existing:
        # Index existing albums by album_id
        existing_idx = {a["album_id"]: a for a in existing.get("albums", [])}

        # Overwrite/insert with latest scraped albums
        for alb in output["albums"]:
            existing_idx[alb["album_id"]] = alb

        merged_albums = list(existing_idx.values())

        # Merge trace (deduplicate by track_id+album_id)
        existing_trace = existing.get("trace_tracks_to_albums", [])
        seen = {(t.get("track_id"), t.get("album_id")) for t in existing_trace}
        for t in output["trace_tracks_to_albums"]:
            key = (t.get("track_id"), t.get("album_id"))
            if key not in seen:
                existing_trace.append(t)
                seen.add(key)

        # Keep the newest source link (or append if you prefer)
        output = {
            "source_link": output.get("source_link"),
            "harvested_albums_count": len(merged_albums),
            "albums": merged_albums,
            "trace_tracks_to_albums": existing_trace
        }

    # Write atomically (optional but safer)
    tmp_path = json_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    tmp_path.replace(json_path)


    print(f"[done] Wrote JSON to: {json_path}")
    print(f"[done] Saved cover PNGs in: {images_dir.resolve()}")

if __name__ == "__main__":
    main()
