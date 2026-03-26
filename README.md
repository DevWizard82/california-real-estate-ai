# 🏠 PropVal AI — California Housing Price Estimator

> A machine learning web app that predicts California property values in real-time using a Random Forest model trained on the classic California Housing dataset.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
[![Live Demo](https://img.shields.io/badge/Live_Demo-View_Site-deb887?style=for-the-badge)](https://california-real-estate-ai-6cpbj48p8azmhr4tdbhjmo.streamlit.app/)

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

## ⚠️ Known Limitations
 
### 1. Older homes predict higher prices
Intuitively, you'd expect older properties to be worth less due to depreciation. However, the model often predicts **higher prices for older homes** — and this is not a bug, it's a dataset artifact.
 
The California Housing dataset comes from the **1990 U.S. Census**, where older homes were frequently located in **established, high-demand neighborhoods** (historic SF districts, prime LA areas). The model learned this correlation:
 
```
older age → established neighborhood → higher price
```
 
This is a **correlation vs. causation** problem. The model is statistically correct on the training data, but the signal is confounded by location prestige, not age itself.
 
**What's missing to fix this properly:**
 
| Missing Feature | Why It Matters |
|---|---|
| Renovation history | A 1920s home fully renovated ≠ a deteriorated one |
| Neighborhood quality score | Age means different things in different areas |
| Proximity to amenities | Schools, transit, parks drive value independently |
| Crime rate | Strong negative price signal not captured here |
| Property condition | Raw age ignores maintenance and upgrades |
 
**Potential fixes for a v2:**
- Engineer an `age × income` interaction term to distinguish "old but wealthy area" from "old and declining area"
- Add a depreciation curve that only penalizes age in low-income census blocks
- Use a richer dataset (Zillow, Redfin, or ATTOM) with condition and renovation data
 
### 2. Dataset is from 1990
Prices, income levels, and neighborhood compositions have changed dramatically over 35 years. Predictions reflect **1990 California market dynamics**, not today's. This model is best used as a learning tool rather than a real valuation system.
 
### 3. Fixed population & occupancy
The app fixes `Population = 1,200` and `AveOccup = 3.0` for simplicity. In reality these vary significantly across census blocks and do influence the prediction.
 
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
