import math
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import streamlit as st
import altair as alt

# ---------------------------
# Paths & config
# ---------------------------

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "spotify_tracks.csv"
MODELS_DIR = APP_DIR / "models"

st.set_page_config(
    page_title="SoundScope – Mood Explorer",
    page_icon="🎧",
    layout="wide",
)

# ---------------------------
# Loading data & artifacts
# ---------------------------


@st.cache_data
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Ensure optional spotify_url column exists
    if "spotify_url" not in df.columns:
        df["spotify_url"] = ""

    return df


@st.cache_resource
def load_artifacts():
    with open(MODELS_DIR / "mood_classifier_rf.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "mood_label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(MODELS_DIR / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, label_encoder, feature_cols


df = load_dataset()
model, label_encoder, feature_cols = load_artifacts()

numeric_features = feature_cols  # columns the model expects


# ---------------------------
# Helper functions
# ---------------------------


def predict_mood(row: pd.Series):
    """Run the trained model on a single row (Series) and return prediction + probs."""
    x = row[feature_cols].values.astype(float).reshape(1, -1)
    pred_idx = model.predict(x)[0]
    probs = model.predict_proba(x)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    class_labels = label_encoder.inverse_transform(np.arange(len(probs)))
    prob_dict = {label: float(p) for label, p in zip(class_labels, probs)}
    return pred_label, prob_dict


def mood_prob_chart(prob_dict: dict, height: int = 260):
    data = pd.DataFrame(
        [{"mood": mood, "probability": prob} for mood, prob in prob_dict.items()]
    )

    y_ticks = [i / 10 for i in range(0, 11)]

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("mood:N", sort="-y", title="Mood"),
            y=alt.Y(
                "probability:Q",
                axis=alt.Axis(format="%", values=y_ticks, title="Probability"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=["mood", alt.Tooltip("probability", format=".1%")],
            color="mood:N",
        )
        .properties(height=height)
    )
    return chart


def embed_spotify_player(spotify_url: str, height: int = 152):
    """Embed a compact Spotify player for a track."""
    if not isinstance(spotify_url, str) or "open.spotify.com/track" not in spotify_url:
        st.info("No Spotify preview available for this track in the dataset.")
        return

    embed_url = spotify_url.replace(
        "open.spotify.com/track", "open.spotify.com/embed/track"
    )

    # Make sure theme=0 is set (dark)
    if "theme=" not in embed_url:
        if "?" in embed_url:
            embed_url += "&theme=0"
        else:
            embed_url += "?theme=0"

    st.components.v1.iframe(embed_url, height=height)


# ---------------------------
# Page title
# ---------------------------

st.title("🎧 SoundScope - Mood Explorer")
st.write(
    "Explore a small curated set of songs, their audio-style features, and a mood "
    "classifier trained from scratch. Pick a track or play with the sliders to see "
    "how changes in tempo, energy, and other features affect the predicted mood."
)

# ---------------------------
# Sidebar filters
# ---------------------------

st.sidebar.header("Filters")

genre_options = ["All"] + sorted(df["genre"].dropna().unique().tolist())
mood_options = ["All"] + sorted(df["mood"].dropna().unique().tolist())
era_options = ["All"] + sorted(df["era"].dropna().unique().tolist())

selected_genre = st.sidebar.selectbox("Genre", genre_options, index=0)
selected_mood = st.sidebar.selectbox("Mood (label)", mood_options, index=0)
selected_era = st.sidebar.selectbox("Era/decade", era_options, index=0)

filtered = df.copy()
if selected_genre != "All":
    filtered = filtered[filtered["genre"] == selected_genre]
if selected_mood != "All":
    filtered = filtered[filtered["mood"] == selected_mood]
if selected_era != "All":
    filtered = filtered[filtered["era"] == selected_era]

st.sidebar.markdown("---")
st.sidebar.write(f"**Tracks visible:** {len(filtered)} / {len(df)} total")

# ---------------------------
# Tabs
# ---------------------------

tab1, tab2 = st.tabs(["Track explorer", "Slider playground"])

# -------------------------------------------------------------------
# TAB 1: Track explorer – pick a real song and see prediction + player
# -------------------------------------------------------------------
with tab1:
    col_left, col_right = st.columns([1.1, 1])

    selected_row = None
    spotify_url = ""

    # ---------------------------
    # Left: song selection + compact features
    # ---------------------------
    with col_left:
        st.subheader("Pick a track")

        if filtered.empty:
            st.warning("No tracks match the current filters. Try widening them.")
        else:
            options = (filtered["track_name"] + " – " + filtered["artist"]).tolist()
            default_index = 0
            selected_label = st.selectbox("Song list", options, index=default_index)

            name_part, artist_part = selected_label.split(" – ", maxsplit=1)
            selected_row = filtered[
                (filtered["track_name"] == name_part)
                & (filtered["artist"] == artist_part)
            ].iloc[0]

            spotify_url = selected_row.get("spotify_url", "")

            st.markdown(
                f"### {selected_row['track_name']} – {selected_row['artist']}"
            )
            st.caption(
                f"Genre: **{selected_row['genre']}** · Era: **{selected_row['era']}** · "
                f"Labeled mood: **{selected_row['mood']}**"
            )

            st.subheader("Audio-style features")

            feature_lines = [
                ("Tempo", f"{selected_row['tempo']:.0f} BPM"),
                ("Danceability", f"{selected_row['danceability']:.2f} (0–1)"),
                ("Energy", f"{selected_row['energy']:.2f} (0–1)"),
                ("Valence", f"{selected_row['valence']:.2f} (0–1)"),
                ("Loudness", f"{selected_row['loudness']:.1f} dBFS (0 = max)"),
                ("Acousticness", f"{selected_row['acousticness']:.2f} (0–1)"),
                ("Instrumentalness", f"{selected_row['instrumentalness']:.2f} (0–1)"),
                ("Speechiness", f"{selected_row['speechiness']:.2f} (0–1)"),
                ("Liveness", f"{selected_row['liveness']:.2f} (0–1)"),
                ("Popularity", f"{selected_row['popularity']:.0f} / 100"),
            ]

            for label, text in feature_lines:
                st.markdown(f"- **{label}:** {text}")

    # ---------------------------
    # Right: prediction + probs + small player
    # ---------------------------
    with col_right:
        st.subheader("Model mood prediction for this track")

        if selected_row is not None:
            if st.button("Predict mood"):
                pred_label, prob_dict = predict_mood(selected_row)

                # Compact badge-style label
                st.markdown(
                    f"**Predicted mood:** "
                    f"<span style='background-color:#16a34a; padding:0.2rem 0.6rem; "
                    f"border-radius:999px; color:white;'>{pred_label}</span>",
                    unsafe_allow_html=True,
                )

                chart = mood_prob_chart(prob_dict, height=260)
                st.altair_chart(chart, use_container_width=True)

                # Compact text list of probabilities
                st.markdown("")
                for mood, prob in sorted(
                    prob_dict.items(), key=lambda x: x[1], reverse=True
                ):
                    st.markdown(f"- **{mood}** – {prob*100:.1f}%")

                # Small embedded Spotify player
                if spotify_url and isinstance(spotify_url, str) and spotify_url.strip():
                    st.markdown("#### Play preview")
                    embed_spotify_player(spotify_url, height=152)
                else:
                    st.info("No Spotify preview available for this track in the dataset.")

# -------------------------------------------------------------------
# TAB 2: Slider playground. Experiment with custom feature vectors
# -------------------------------------------------------------------
with tab2:
    st.subheader("Playground: tweak audio features and see the mood")

    col_left_pg, col_right_pg = st.columns([1.1, 1])

    with col_left_pg:
        st.markdown("Adjust the sliders to create a hypothetical track:")

        tempo = st.slider("Tempo (BPM)", min_value=60, max_value=200, value=120, step=2)
        danceability = st.slider(
            "Danceability (0–1)", min_value=0.0, max_value=1.0, value=0.6, step=0.01
        )
        energy = st.slider(
            "Energy (0–1)", min_value=0.0, max_value=1.0, value=0.6, step=0.01
        )
        valence = st.slider(
            "Valence (0–1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01
        )
        loudness = st.slider(
            "Loudness (dBFS, 0=max)",
            min_value=-20.0,
            max_value=0.0,
            value=-6.0,
            step=0.5,
        )
        acousticness = st.slider(
            "Acousticness (0–1)", min_value=0.0, max_value=1.0, value=0.3, step=0.01
        )
        instrumentalness = st.slider(
            "Instrumentalness (0–1)", min_value=0.0, max_value=1.0, value=0.1, step=0.01
        )
        speechiness = st.slider(
            "Speechiness (0–1)", min_value=0.0, max_value=1.0, value=0.05, step=0.01
        )
        liveness = st.slider(
            "Liveness (0–1)", min_value=0.0, max_value=1.0, value=0.15, step=0.01
        )
        popularity = st.slider(
            "Popularity (0–100)", min_value=0, max_value=100, value=60, step=5
        )

        # Build a Series with the same feature names the model expects
        feature_values = {
            "tempo": tempo,
            "danceability": danceability,
            "energy": energy,
            "valence": valence,
            "loudness": loudness,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "speechiness": speechiness,
            "liveness": liveness,
            "popularity": popularity,
        }

        # In case feature_cols is a subset/superset, align
        series_for_model = pd.Series(
            {col: feature_values[col] for col in feature_cols}
        )

    with col_right_pg:
        st.subheader("Model prediction for slider settings")

        if st.button("Predict mood from sliders"):
            pred_label_pg, prob_dict_pg = predict_mood(series_for_model)

            st.markdown(
                f"**Predicted mood:** "
                f"<span style='background-color:#16a34a; padding:0.2rem 0.6rem; "
                f"border-radius:999px; color:white;'>{pred_label_pg}</span>",
                unsafe_allow_html=True,
            )

            chart_pg = mood_prob_chart(prob_dict_pg, height=300)
            st.altair_chart(chart_pg, use_container_width=True)

            st.markdown("")
            for mood, prob in sorted(
                prob_dict_pg.items(), key=lambda x: x[1], reverse=True
            ):
                st.markdown(f"- **{mood}** – {prob*100:.1f}%")
