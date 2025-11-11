# src/get_spotify_data.py
import os
import re
import time
from pathlib import Path
from typing import Iterable, Generator, List, Tuple, Optional, Dict, TypeVar

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

# ------------------------------------------------------------
# Env + Auth
# ------------------------------------------------------------
load_dotenv()

CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")

if not CLIENT_ID or not CLIENT_SECRET or not REDIRECT_URI:
    raise RuntimeError(
        "Missing Spotify credentials. Ensure SPOTIPY_CLIENT_ID, "
        "SPOTIPY_CLIENT_SECRET, and SPOTIPY_REDIRECT_URI are set in .env"
    )

auth_manager = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope="playlist-read-private",
    cache_path=".cache-soundscope",
    open_browser=False
)
sp = spotipy.Spotify(auth_manager=auth_manager)

# ------------------------------------------------------------
# Playlists loader (from playlists.txt)
#   Supported lines:
#     - URL or raw ID           -> auto-label from playlist name
#     - label, URL-or-ID        -> explicit label
#   Ignores blank lines and lines starting with '#'
# ------------------------------------------------------------
def extract_playlist_id(s: str) -> Optional[str]:
    s = s.strip()
    if not s:
        return None
    m = re.search(r"playlist/([A-Za-z0-9]+)", s)  # URL form
    if m:
        return m.group(1)
    # Raw ID fallback (be lenient on length)
    if re.fullmatch(r"[A-Za-z0-9]{10,}", s):
        return s
    return None

def slugify(name: str, maxlen: int = 24) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return (name or "playlist")[:maxlen]

def load_playlists_txt(path: Path) -> List[Tuple[str, str]]:
    """
    Returns list of (label, playlist_id).
    """
    if not path.exists():
        raise FileNotFoundError(f"playlists file not found: {path}")

    out: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if "," in line:
                label_part, id_part = line.split(",", 1)
                label = label_part.strip()
                pid = extract_playlist_id(id_part)
                if pid:
                    out.append((label, pid))
                else:
                    print(f"[WARN] Could not parse playlist ID from: {line}")
                continue

            pid = extract_playlist_id(line)
            if not pid:
                print(f"[WARN] Could not parse playlist ID from: {line}")
                continue

            # Fetch name for auto label
            try:
                meta = sp.playlist(pid, fields="name")
                auto_label = slugify(meta.get("name", "playlist"))
            except Exception as e:
                print(f"[WARN] Failed to fetch name for {pid}: {e}")
                auto_label = "playlist"
            out.append((auto_label, pid))

    # Dedup by playlist_id (keep first label encountered)
    seen = set()
    deduped: List[Tuple[str, str]] = []
    for lbl, pid in out:
        if pid not in seen:
            deduped.append((lbl, pid))
            seen.add(pid)
    return deduped

# Resolve playlists.txt at project root (…/SoundScope/playlists.txt)
PLAYLISTS_FILE = Path(__file__).resolve().parents[1] / "playlists.txt"
playlist_pairs = load_playlists_txt(PLAYLISTS_FILE)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
T = TypeVar("T")

def batch(iterable: Iterable[T], n: int) -> Generator[List[T], None, None]:
    """
    Yield lists of size up to n from an iterable.
    """
    buf: List[T] = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

def paginate_playlist_items(pid: str, market: Optional[str] = "US") -> List[dict]:
    """
    Return all items from a playlist (tracks only).
    """
    items: List[dict] = []
    try:
        results = sp.playlist_tracks(pid, market=market, additional_types=("track",))
    except Exception as e:
        print(f"[WARN] Failed to fetch playlist {pid}: {e}")
        return items

    items.extend(results.get("items", []))
    while results.get("next"):
        try:
            results = sp.next(results)
            items.extend(results.get("items", []))
        except Exception as e:
            print(f"[WARN] Pagination failed for playlist {pid}: {e}")
            break
    return items

# ------------------------------------------------------------
# Main collection
# ------------------------------------------------------------
all_rows: List[Dict[str, object]] = []
print("Fetching Spotify playlist data...")

for label, pid in playlist_pairs:
    print(f"  -> {label}: {pid}")
    items = paginate_playlist_items(pid, market="US")
    if not items:
        print(f"[INFO] No items returned for playlist {pid} ({label}). Skipping.")
        continue

    # Collect track IDs (skip local/None)
    track_objs = []
    for it in items:
        track = it.get("track")
        if not track or track.get("is_local") or not track.get("id"):
            continue
        track_objs.append(track)

    # Batch fetch audio features (max 100 per call)
    id_list = [t["id"] for t in track_objs]
    features_by_id: Dict[str, dict] = {}
    for chunk in batch(id_list, 100):
        try:
            feats = sp.audio_features(chunk)
        except Exception as e:
            print(f"[WARN] audio_features failed for chunk of {len(chunk)} IDs: {e}")
            feats = [None] * len(chunk)
        for tid, f in zip(chunk, feats):
            if f:
                features_by_id[tid] = f
        time.sleep(0.05)  # be polite

    # Build rows
    for t in track_objs:
        tid = t["id"]
        feats = features_by_id.get(tid)
        if not feats:
            continue

        row = {
            "track_id": tid,
            "track_name": t.get("name"),
            "artist": (t.get("artists") or [{}])[0].get("name"),
            "artist_id": (t.get("artists") or [{}])[0].get("id"),
            "album": (t.get("album") or {}).get("name"),
            "release_date": (t.get("album") or {}).get("release_date"),
            "popularity": t.get("popularity"),
            "genre": label,  # playlist bucket as baseline label
        }
        row.update(feats)
        all_rows.append(row)

    # Light pause between playlists
    time.sleep(0.2)

# ------------------------------------------------------------
# DataFrame + cleaning + save
# ------------------------------------------------------------
df = pd.DataFrame(all_rows)

before = len(df)
df = df.dropna(subset=["track_id"]).drop_duplicates(subset=["track_id"])
after = len(df)
print(f"Collected {before} rows → {after} unique tracks.")

preferred_cols = [
    "track_id","track_name","artist","artist_id","album","genre",
    "popularity","release_date",
    "danceability","energy","valence","tempo","loudness",
    "acousticness","instrumentalness","speechiness","liveness",
    "duration_ms","key","mode","time_signature"
]
keep = [c for c in preferred_cols if c in df.columns]
df = df[keep]

os.makedirs("data", exist_ok=True)
out_path = "data/spotify_tracks.csv"
df.to_csv(out_path, index=False, encoding="utf-8")
print(f"Saved dataset to {out_path}")
print(df.head(5).to_string(index=False))
