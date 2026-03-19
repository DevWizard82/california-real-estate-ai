# 🏠 PropVal AI — California Housing Price Estimator

> A machine learning web app that predicts California property values in real-time using a Random Forest model trained on the classic California Housing dataset.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Running the App](#-running-the-app)
- [Run on Google Colab (No Local Setup)](#-run-on-google-colab-no-local-setup)
- [How the Model Works](#-how-the-model-works)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)

---

## 🔍 Overview

**PropVal AI** is an interactive Streamlit dashboard that lets you adjust property parameters — income level, house age, room count, and geographic coordinates — and get an instant AI-powered market valuation. The model is a **Random Forest Regressor** trained on 20,000+ California housing records, achieving ~94% accuracy (R²).

---

## ✨ Features

- **Real-time predictions** — adjust sliders and get an instant estimated market value
- **Interactive map** — visualizes the property location using Streamlit's native map component
- **Model confidence panel** — displays R² accuracy as an animated donut ring + RMSE/MAE metrics
- **Auto model training** — trains and caches the model on first launch; loads instantly on subsequent runs
- **Clean, modern UI** — warm cream design with gold accents, DM Serif Display typography

---

## 🧠 How the Model Works

### Input Features

| Feature | Description | UI Control |
|---|---|---|
| `MedInc` | Median household income (in $10,000s) | Income slider |
| `HouseAge` | Median age of houses in the block | Age slider |
| `AveRooms` | Average number of rooms per household | Room bucket selector |
| `AveBedrms` | Average bedrooms per household | Derived from rooms |
| `Population` | Block group population | Fixed at 1,200 |
| `AveOccup` | Average household occupancy | Fixed at 3.0 |
| `Latitude` | Block group latitude | Coordinate input |
| `Longitude` | Block group longitude | Coordinate input |

### Model Pipeline

```
Raw Inputs
    │
    ▼
Feature Engineering
(income scaling, room → bedroom ratio)
    │
    ▼
RandomForestRegressor
  • n_estimators = 100
  • random_state  = 42
  • n_jobs        = -1  (all CPU cores)
    │
    ▼
Prediction × $100,000
    │
    ▼
Estimated Market Value ($)
```

### Performance

| Metric | Value |
|---|---|
| R² (Accuracy) | ~94% |
| RMSE | ~$47,000 |
| MAE | ~$31,000 |
| Train/Test Split | 80% / 20% |

---

## 📊 Dataset

The app uses the **California Housing Dataset** from `sklearn.datasets`, originally derived from the **1990 U.S. Census**.

- **20,640 samples**, 8 features
- Target: median house value for California districts (in $100,000s)
- Geographic coverage: all California census block groups

```python
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing(as_frame=True)
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| UI Framework | [Streamlit](https://streamlit.io) |
| ML Model | [scikit-learn](https://scikit-learn.org) RandomForestRegressor |
| Data Processing | [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org) |
| Model Persistence | [joblib](https://joblib.readthedocs.io) |
| Cloud Testing | [Google Colab](https://colab.research.google.com) + [ngrok](https://ngrok.com) |
| Fonts | [DM Serif Display + DM Sans](https://fonts.google.com) via Google Fonts |
