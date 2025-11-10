# SoundScope

SoundScope is an exploratory project that combines Spotify-style audio features with machine learning models to predict song characteristics such as genre, mood, and popularity alignment. This repository tracks progress toward the final interactive web application.

## Milestone 1 Notebook

The `notebooks/milestone1_exploration.ipynb` notebook contains the first milestone deliverable:

- Loads a prototype dataset of 25 tracks with Spotify audio descriptors (tempo, danceability, energy, etc.).
- Performs descriptive statistics, missing-value checks, and exploratory visualizations.
- Trains a baseline logistic regression classifier to predict the genre using the numeric audio features.
- Summarizes next steps for data collection and modeling.

## Data

The `data/sample_tracks.csv` file is a hand-crafted dataset used to prototype the exploratory workflow. Future milestones will replace this with tracks collected from the Spotify Web API and chart datasets.

## Environment

The notebook assumes access to common data science libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`. Install them in your environment before running the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Project Roadmap

1. Expand the dataset by querying the Spotify Web API and merging with chart performance data.
2. Engineer features related to popularity trends and temporal context.
3. Train and compare advanced models (Random Forest, Gradient Boosting, LightGBM/XGBoost).
4. Build the SoundScope web interface in Streamlit or Flask with interactive visualizations.