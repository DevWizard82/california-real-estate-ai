import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PropVal AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_or_train_model():
    if os.path.exists("rf_model.pkl"):
        model = joblib.load("rf_model.pkl")
        california = fetch_california_housing(as_frame=True)
        X, y = california.data, california.target
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        preds = model.predict(X_test)
    else:
        california = fetch_california_housing(as_frame=True)
        X, y = california.data, california.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        with st.spinner("Training model for the first time... (~20 sec)"):
            model.fit(X_train, y_train)
        joblib.dump(model, "rf_model.pkl")
        preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = model.score(X_test, y_test)
    return model, rmse, mae, r2

model, rmse_score, mae_score, r2_score = load_or_train_model()

ROOM_MAP = {"1-3": 2.0, "4-6": 5.0, "7-9": 8.0, "10+": 12.0}

# ── Session state ─────────────────────────────────────────────────────────────
if "rooms" not in st.session_state:
    st.session_state["rooms"] = "4-6"
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏠 PropVal AI")
st.caption("California property value estimator — powered by Random Forest")
st.divider()

# ── Layout ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ── LEFT: Inputs ──────────────────────────────────────────────────────────────
with col_left:
    st.subheader("Property Analysis")
    st.caption("Adjust parameters to generate a real-time AI valuation.")

    st.markdown("**Median Area Income**")
    med_income = st.slider(
        "Median Area Income ($)", min_value=10_000, max_value=150_000,
        value=85_000, step=1_000, format="$%d", label_visibility="collapsed"
    )
    st.caption(f"Selected: **${med_income:,}**")

    st.markdown("**Property Age (Years)**")
    house_age = st.slider(
        "Property Age", min_value=1, max_value=52,
        value=12, label_visibility="collapsed"
    )
    st.caption(f"Selected: **{house_age} yrs**")

    st.markdown("**Average Rooms**")
    r_cols = st.columns(4)
    for i, label in enumerate(ROOM_MAP):
        with r_cols[i]:
            is_selected = st.session_state["rooms"] == label
            if st.button(
                f"{'✓ ' if is_selected else ''}{label}",
                key=f"room_{label}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state["rooms"] = label
                st.rerun()

    st.markdown("**Location**")
    lat_col, lng_col = st.columns(2)
    with lat_col:
        st.caption("LATITUDE  •  32.5 → 42.0")
        latitude = st.number_input(
            "lat", value=34.0522, min_value=32.5, max_value=42.0,
            step=0.001, format="%.4f", label_visibility="collapsed"
        )
    with lng_col:
        st.caption("LONGITUDE  •  -124.5 → -114.1")
        longitude = st.number_input(
            "lng", value=-118.2437, min_value=-124.5, max_value=-114.1,
            step=0.001, format="%.4f", label_visibility="collapsed"
        )

    st.markdown("")
    predict_clicked = st.button(
        "🔍  Analyze Property Value",
        use_container_width=True,
        type="primary"
    )

# ── RIGHT: Outputs ────────────────────────────────────────────────────────────
with col_right:
    st.subheader("Estimated Market Value")

    if st.session_state["prediction"] is not None:
        pred_val = st.session_state["prediction"]
        st.metric(
            label="Predicted Price",
            value=f"${pred_val:,.0f}",
            delta=f"±${rmse_score * 100_000:,.0f} margin of error"
        )
    else:
        st.info("👈  Set your parameters and click **Analyze Property Value** to get a prediction.")

    st.divider()

    map_col, conf_col = st.columns([1.2, 0.8], gap="medium")

    with map_col:
        st.markdown("**📍 Location Context**")
        st.map(
            pd.DataFrame({"lat": [latitude], "lon": [longitude]}),
            zoom=9,
            use_container_width=True
        )

    with conf_col:
        st.markdown("**✅ Model Confidence**")
        confidence = round(r2_score * 100, 1)
        variance   = round((1 - r2_score) * 100, 1)

        st.metric("R² Accuracy", f"{confidence}%")
        st.metric("RMSE", f"${rmse_score * 100_000:,.0f}")
        st.metric("MAE",  f"${mae_score  * 100_000:,.0f}")
        st.caption(f"Predicted variance: ±{variance}%")

# ── Prediction logic ──────────────────────────────────────────────────────────
if predict_clicked:
    avg_rooms = ROOM_MAP[st.session_state["rooms"]]
    features  = np.array([[
        med_income / 10_000,
        float(house_age),
        avg_rooms,
        max(1.0, avg_rooms * 0.25),
        1200.0,
        3.0,
        latitude,
        longitude,
    ]])
    st.session_state["prediction"] = model.predict(features)[0] * 100_000
    st.rerun()
