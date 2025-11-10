# get_spotify_data.py
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import os

# Load credentials from environment (or paste for quick test)
client_id = "f336c3c43bae45bdadee9df298376eb2"
client_secret = "3a0bdd907fec46cf90d3872dbb640368"

# Authenticate (no redirect needed for public data)
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Define playlists by genre/category
playlists = {
    "pop": "37i9dQZF1DX1H4LbvY4OJi",      # Dance Party
    "rock": "37i9dQZF1DWXRqgorJj26U",     # Rock Classics
    "rap": "37i9dQZF1DX0XUsuxWHRQd",      # RapCaviar
    "chill": "37i9dQZF1DX4WYpdgoIcn6",    # Chill Hits
    "piano": "37i9dQZF1DX4sWSpwq3LiO",    # Peaceful Piano
    "top50": "37i9dQZEVXbMDoHDwVN2tF"     # Top 50 Global
}

all_tracks = []

print("Fetching Spotify playlist data...")

for genre, pid in playlists.items():
    results = sp.playlist_tracks(pid)
    tracks = results["items"]

    # Paginate if there are more than 100 tracks
    while results["next"]:
        results = sp.next(results)
        tracks.extend(results["items"])

    for item in tracks:
        track = item["track"]
        if track is None:
            continue
        tid = track["id"]
        if tid is None:
            continue

        features = sp.audio_features(tid)[0]
        if not features:
            continue

        # Merge metadata with features
        track_data = {
            "track_name": track["name"],
            "artist": track["artists"][0]["name"],
            "genre": genre,
            "popularity": track["popularity"],
            "release_date": track["album"]["release_date"]
        }
        track_data.update(features)
        all_tracks.append(track_data)

        time.sleep(0.05)  # avoid hitting rate limits

# Create dataframe
df = pd.DataFrame(all_tracks)
print(f"Total songs collected: {len(df)}")

# Drop duplicates and missing values
df = df.drop_duplicates(subset=["track_name", "artist"]).dropna()

# Save dataset
os.makedirs("data", exist_ok=True)
df.to_csv("data/spotify_tracks.csv", index=False)

print("Saved dataset to data/spotify_tracks.csv")
print(df.head())
