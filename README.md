# SoundScope

SoundScope is an interactive music–analysis tool that blends Spotify-style audio features with machine-learning predictions inside a clean, intuitive Streamlit interface.  
It is designed as both a data-science project and a functional demo application that allows users to explore songs, visualize audio characteristics, and experiment with predictive models.

This repository contains:

- A curated static dataset of well-known tracks and their audio-feature approximations  
- A fully functional Streamlit web app (`app.py`) for interactive exploration  
- A training pipeline for a mood-classification model  
- Exploratory Jupyter notebooks for early milestone development  


------------------------------------------------------------
## 1. Project Background

SoundScope started as a milestone-based exploration notebook for analyzing audio-feature datasets.  
Originally, the plan was to collect real Spotify audio-feature data via the Spotify Web API.

However:

### Important Note About the API
Due to Spotify significantly tightening API permissions in 2024–2025 (including requiring approved user-token scopes for audio-feature bulk access), automated scraping of playlists and the audio-features endpoint now frequently returns **403 Forbidden**, even when using valid client credentials.  
This prevented real-time data extraction.

## My Solution
To keep the project functional and predictable, SoundScope now uses a **static CSV dataset** that includes:

- 100% real song titles and artists  
- Carefully hand-crafted audio features modeled after typical Spotify values  
- Spotify playback embed links for every track  
- Labels for genre, mood, and era  

This still allows us to:
- Build and test ML models  
- Explore audio-feature relationships  
- Provide an interactive UI  
- Compare and visualize songs  
- Train classifiers and run predictions  

…all without relying on unstable external APIs.

------------------------------------------------------------
## 2. Repository Structure
```
SoundScope/
│
├── app.py                     # Streamlit web application
│
├── data/
│   └── spotify_tracks.csv     # Static curated dataset with audio features + Spotify URLs
│
├── models/
│   ├── train_mood_model.py    # Retrains the ML model from the dataset
│   ├── mood_classifier_rf.pkl # Saved RandomForest classifier
│   ├── mood_label_encoder.pkl # LabelEncoder for the mood column
│   └── feature_cols.pkl       # List of numeric features used during training
│
├── notebooks/
│   └── milestone1_exploration.ipynb
│   └── soundscope_model_and_app.ipynb
│
├── src/
│   └── get_spotify_data.py    # Old file used to scrape API data (not in use)
│
├── requirements.txt
│
├── playlists.txt              # Old file used to store playlists to use (not in use)
│
├── .gitignore
│
└── README.md
```

------------------------------------------------------------
## 3. Milestone Notebook (Exploration)

`notebooks/milestone1_exploration.ipynb` was the starting point of the project. It includes:

- Loading the prototype dataset
- Descriptive statistics on numeric features
- Category analysis for genre/mood/era
- Missing-value analysis
- ASCII-based mini-visualizations
- Pearson correlation matrices
- A from-scratch softmax regression classifier
- Train/test split evaluation
- Next-step planning

This notebook represents **Milestone 1** and was kept fully self-contained with zero third-party visualization libraries.

------------------------------------------------------------
## 4. Streamlit Web Application

`app.py` is the modern, polished interface for SoundScope.

### Features Include:

### Track Browser  
- Filter by **genre, mood, era**
- Select a song to view its metadata  
- Display of numeric audio-style features in clean labeled format  

### Built-in Spotify Playback  
Every song includes an **embedded Spotify player** using: https://open.spotify.com/embed/track/<TRACK_ID>


This allows users to listen from directly inside the app.

### Mood Prediction  
- Uses a trained **RandomForestClassifier**
- Displays:
  - predicted mood
  - probability breakdown
  - a bar chart visualization
  - clean formatted probability text (e.g., “Happy — 63.2%”)

### Audio-Feature Sliders Page  
An experimental page where users can:
- manually change features like energy, danceability, valence, tempo  
- run predictions dynamically  
- view probability curves  
- visualize how mood output changes  

------------------------------------------------------------
## 5. Machine Learning Pipeline

The training script `train_mood_model.py` performs:

1. Loading and cleaning the CSV  
2. Label-encoding the target mood  
3. Selecting numeric audio-style features  
4. Train/test splitting  
5. Training a RandomForestClassifier  
6. Saving trained artifacts into the `models/` folder  

Artifacts saved:
- `mood_classifier_rf.pkl`
- `mood_label_encoder.pkl`
- `feature_cols.pkl`

These are automatically loaded by `app.py`.

------------------------------------------------------------
## 6. Dataset Details

The dataset (`spotify_tracks.csv`) contains:

- track_id  
- track_name  
- artist  
- tempo  
- danceability  
- energy  
- valence  
- loudness  
- acousticness  
- instrumentalness  
- speechiness  
- liveness  
- popularity  
- genre  
- mood  
- era  
- spotify_url (for embedded playback)

### Why numeric values are approximations  
Since API access was blocked, each numeric field (0–1 normalized values, tempo, loudness) was hand-crafted based on typical Spotify distributions.  
Examples:

- Loudness is negative dB (usually between –6 to –12)  
- Tempo generally ranges 80–160 BPM  
- Energy/danceability follow Spotify’s 0–1 scale  

These values preserve statistical realism without requiring live API pulls.

------------------------------------------------------------
## 7. Installation & Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

    streamlit run app.py


## 8. Usage

### Track Exploration
Choose a song from the dropdown → view its audio-style features → play the embedded Spotify preview → see the machine-learning mood prediction and probability breakdown.

### Slider Experimentation
Switch to the experimental sliders page to manually adjust key audio features such as tempo, energy, valence, and danceability.  
The model updates its mood prediction in real time and displays the probabilities in both graphical and text formats.


## 9. Limitations

- The app uses a **static CSV dataset** because recent Spotify API changes (2024–2025) block bulk audio-feature access.
- All numeric audio features (energy, loudness, danceability, etc.) are *approximate*, modeled after typical Spotify values.
- The mood classifier is intentionally lightweight and trained on a small dataset, making predictions more illustrative than authoritative.
- Embedded Spotify previews require valid track URLs; if a link is missing, playback is not available.
- Loudness and instrumentalness are best-effort approximations and may not match real audio engineering measurements.



## 10. Roadmap

### Upcoming Improvements
- Add a secondary classifier for genre prediction.
- Introduce a popularity-trend prediction model.
- Add richer visualizations for energy, valence, and tempo distributions.
- Improve the embedded playback layout and responsiveness.
- IMPLEMENTED * Add a “similar songs” recommendation panel based on feature distance.

### Medium-Term Features
- User-uploaded CSV support to run predictions on personal datasets.
- Spectrogram-style interactive visuals for tempo/energy patterns.
- More polished multi-track comparison system.

### Long-Term Goals
- Restore real Spotify API data once approval for private audio-feature scopes is granted.
- Replace the static dataset with live audio feature pulls.
- Add SHAP-based model explanations.
- Deploy the full app publicly via Streamlit Cloud, Render, or similar hosting.