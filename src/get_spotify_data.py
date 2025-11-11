# src/get_spotify_data.py
import os
import re
import time
from pathlib import Path
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

# --- Credentials ---
client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")

auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# ---------- NEW: load playlists from playlists.txt ----------
def extract_playlist_id(s: str) -> str | None:
    """Return a Spotify playlist ID from a URL or raw ID."""
    s = s.strip()
    if not s:
        return None
    # URL like https://open.spotify.com/playlist/37i9dQZEVXbMDoHDwVN2tF?si=...
    m = re.search(r"playlist/([A-Za-z0-9]+)", s)
    if m:
        return m.group(1)
    # Raw ID (looks like 22 chars, alnum)
    if re.fullmatch(r"[A-Za-z0-9]{10,}", s):  # be lenient
        return s
    return None

def slugify(name: str) -> str:
    """Make a short, safe label."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    # keep it concise
    return name[:20] or "playlist"

def load_playlists_txt(path: Path) -> list[tuple[str, str]]:
    """
    Returns a list of (label, playlist_id).
    Supports lines like:
      - https://open.spotify.com/playlist/xxxxx
      - xxxxxxxxx (raw ID)
      - chill, https://open.spotify.com/playlist/xxxxx   (explicit label)
    Lines starting with '#' are comments.
    """
    pairs: list[tuple[str, str]] = []
    if not path.exists():
        raise FileNotFoundError(f"playlists file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Optional label comma format
            if "," in line:
                label_part, id_part = line.split(",", 1)
                label_part = label_part.strip()
                pid = extract_playlist_id(id_part)
                if pid:
                    pairs.append((label_part, pid))
                continue

            # No label provided; just an ID/URL
            pid = extract_playlist_id(line)
            if pid:
                # fetch name to make a friendly label
                try:
                    pl_meta = sp.playlist(pid, fields="name")
                    label = slugify(pl_meta.get("name", "playlist"))
                except Exception:
                    # fallback generic
                    label = "playlist"
                pairs.append((label, pid))

    # Deduplicate by playlist_id (keep first label)
    seen = set()
    deduped: list[tuple[str, str]] = []
    for lbl, pid in pairs:
        if pid not in seen:
            deduped.append((lbl, pid))
            seen.add(pid)
    return deduped

# Location of playlists.txt (project root by default)
PLAYLISTS_FILE = Path(__file__).resolve().parents[1] / "playlists.txt"
playlist_pairs = load_playlists_txt(PLAYLISTS_FILE)
# -----------------------------------------------------------

all_rows = []
print("Fetching Spotify playlist data...")

for label, pid in playlist_pairs:
    print(f"  -> {label}: {pid}")
    results = sp.playlist_tracks(pid, additional_types=("track",))
    items = results["items"]
    while results["next"]:
        results = sp.next(results)
        items.extend(results["items"])

    for it in items:
        track = it.get("track")
        if not track or track.get("is_local") or track.get("id") is None:
            continue

        tid = track["id"]
        feats_list = sp.audio_features([tid])  # still 1-by-1; simple & clear
        feats = feats_list[0] if feats_list else None
        if not feats:
            continue

        row = {
            "track_id": tid,
            "track_name": track.get("name"),
            "artist": (track.get("artists") or [{}])[0].get("name"),
            "artist_id": (track.get("artists") or [{}])[0].get("id"),
            "album": (track.get("album") or {}).get("name"),
            "release_date": (track.get("album") or {}).get("release_date"),
            "popularity": track.get("popularity"),
            "genre": label,  # using label as our bucket
        }
        row.update(feats)
        all_rows.append(row)
        time.sleep(0.03)  # be polite

# --- DataFrame + cleaning (unchanged) ---
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
