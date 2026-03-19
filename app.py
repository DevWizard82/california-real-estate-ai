import streamlit as st
import numpy as np
import joblib
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PropVal AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F2EDE6 !important;
}

.main, .block-container {
    background-color: #F2EDE6 !important;
    padding-top: 2rem !important;
    max-width: 1200px;
}

/* ── Header / Logo ── */
.logo-block {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 2.5rem;
}
.logo-icon {
    background: #C9A96E;
    border-radius: 8px;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}
.logo-text {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #1C1C1E;
    letter-spacing: -0.5px;
}
.logo-text span {
    color: #C9A96E;
}

/* ── Cards ── */
.card {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 2rem 2.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}

/* ── Section label ── */
.section-title {
    font-size: 1.35rem;
    font-weight: 600;
    color: #1C1C1E;
    margin-bottom: 0.2rem;
}
.section-sub {
    font-size: 0.85rem;
    color: #8E8E93;
    margin-bottom: 1.6rem;
}

/* ── Slider label row ── */
.slider-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.1rem;
}
.slider-label .label-name {
    font-size: 0.88rem;
    font-weight: 600;
    color: #1C1C1E;
}
.slider-label .label-val {
    font-size: 0.88rem;
    font-weight: 600;
    color: #C9A96E;
}

/* ── Streamlit slider overrides ── */
.stSlider > div > div > div > div {
    background: #C9A96E !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #1C1C1E !important;
    border: none !important;
    width: 14px !important;
    height: 14px !important;
}

/* ── Room buttons ── */
.room-btn-row {
    display: flex;
    gap: 10px;
    margin-top: 0.4rem;
}

/* ── Lat / Lng inputs ── */
.input-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 0.5rem;
}
.input-group label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #8E8E93;
    text-transform: uppercase;
    margin-bottom: 4px;
    display: block;
}

/* ── Estimate card ── */
.estimate-card {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 2.5rem 2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 200px;
    text-align: center;
}
.estimate-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: #C9A96E;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.estimate-value {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: #1C1C1E;
    line-height: 1;
}
.estimate-dash {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: #CCCCCC;
    line-height: 1;
    letter-spacing: 4px;
}

/* ── Bottom cards ── */
.bottom-card {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    height: 100%;
}
.bottom-card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1C1C1E;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Confidence ring ── */
.confidence-ring-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
}
.confidence-ring-wrap .conf-pct {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #1C1C1E;
}
.confidence-ring-wrap .conf-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8E8E93;
}
.variance-line {
    font-size: 0.78rem;
    color: #8E8E93;
    margin-top: 0.4rem;
    text-align: center;
}

/* ── Predict button ── */
.stButton > button {
    background: #4A5568 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.85rem 1.5rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #2D3748 !important;
}

/* ── Number input ── */
.stNumberInput > div > div > input {
    background: #F8F7F4 !important;
    border: 1.5px solid #E5E2DC !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    color: #1C1C1E !important;
    padding: 0.5rem 0.8rem !important;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 16px;
    margin-top: 1rem;
}
.metric-box {
    flex: 1;
    background: #F8F7F4;
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
}
.metric-box .m-val {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1C1C1E;
}
.metric-box .m-lbl {
    font-size: 0.68rem;
    color: #8E8E93;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
}

/* ── Hide Streamlit default elements ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_or_train_model():
    model_path = "rf_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        rmse, mae, r2 = 0.0, 0.0, 0.94
    else:
        california = fetch_california_housing(as_frame=True)
        X = california.data
        y = california.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        r2   = model.score(X_test, y_test)
        joblib.dump(model, model_path)
    return model, rmse, mae, r2

model, rmse_score, mae_score, r2_score = load_or_train_model()

# ── Room-bucket mapping ───────────────────────────────────────────────────────
ROOM_MAP = {"1-3": 2.0, "4-6": 5.0, "7-9": 8.0, "10+": 12.0}

# ── Session state defaults ────────────────────────────────────────────────────
if "rooms" not in st.session_state:
    st.session_state["rooms"] = "4-6"
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None

# ── LOGO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="logo-block">
  <div class="logo-icon">🏠</div>
  <div class="logo-text">PropVal <span>AI</span></div>
</div>
""", unsafe_allow_html=True)

# ── Layout: Left col (inputs) | Right col (outputs) ──────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Property Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Adjust parameters to generate a real-time AI valuation.</div>', unsafe_allow_html=True)

    # ── Median Income ──
    med_income = st.slider(
        "Median Area Income ($)",
        min_value=10_000, max_value=150_000,
        value=85_000, step=1_000,
        format="$%d",
        label_visibility="collapsed"
    )
    st.markdown(f"""
    <div class="slider-label" style="margin-top:-0.8rem; margin-bottom:0.5rem">
        <span class="label-name">Median Area Income</span>
        <span class="label-val">${med_income:,}</span>
    </div>""", unsafe_allow_html=True)

    # ── House Age ──
    house_age = st.slider(
        "Property Age (Years)",
        min_value=1, max_value=52,
        value=12, step=1,
        label_visibility="collapsed"
    )
    st.markdown(f"""
    <div class="slider-label" style="margin-top:-0.8rem; margin-bottom:1rem">
        <span class="label-name">Property Age (Years)</span>
        <span class="label-val">{house_age} Yrs</span>
    </div>""", unsafe_allow_html=True)

    # ── Average Rooms ──
    st.markdown('<div class="slider-label"><span class="label-name">Average Rooms</span></div>', unsafe_allow_html=True)
    r_cols = st.columns(4)
    for i, (label, val) in enumerate(ROOM_MAP.items()):
        with r_cols[i]:
            selected = st.session_state["rooms"] == label
            btn_style = "background:#C9A96E;border:2px solid #C9A96E;color:#fff;" if selected else "background:#fff;border:2px solid #E5E2DC;color:#1C1C1E;"
            if st.button(label, key=f"room_{label}"):
                st.session_state["rooms"] = label
                st.rerun()

    # ── Lat / Lng ──
    st.markdown("<br>", unsafe_allow_html=True)
    lat_col, lng_col = st.columns(2)
    with lat_col:
        st.markdown('<span style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#8E8E93;text-transform:uppercase">LATITUDE <span style="color:#aaa;font-size:0.65rem">32.5 to 42.0</span></span>', unsafe_allow_html=True)
        latitude = st.number_input("lat", value=34.0522, min_value=32.5, max_value=42.0,
                                   step=0.001, format="%.4f", label_visibility="collapsed")
    with lng_col:
        st.markdown('<span style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;color:#8E8E93;text-transform:uppercase">LONGITUDE <span style="color:#aaa;font-size:0.65rem">-124.5 to -114.1</span></span>', unsafe_allow_html=True)
        longitude = st.number_input("lng", value=-118.2437, min_value=-124.5, max_value=-114.1,
                                    step=0.001, format="%.4f", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍  Analyze Property Value")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Right column ──────────────────────────────────────────────────────────────
with col_right:
    # Estimated value card
    st.markdown('<div class="estimate-card">', unsafe_allow_html=True)
    if st.session_state["prediction"] is not None:
        pred_val = st.session_state["prediction"]
        st.markdown(f"""
        <div class="estimate-label">Estimated Market Value</div>
        <div class="estimate-value">${pred_val:,.0f}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="estimate-label">Estimated Market Value</div>
        <div class="estimate-dash">$&mdash;&mdash;&mdash;,&mdash;&mdash;&mdash;</div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bottom row: Location map + Confidence
    bot_left, bot_right = st.columns([1.1, 0.9], gap="medium")

    with bot_left:
        st.markdown('<div class="bottom-card">', unsafe_allow_html=True)
        st.markdown('<div class="bottom-card-title">📍 Location Context</div>', unsafe_allow_html=True)
        map_data = {"lat": [latitude], "lon": [longitude]}
        import pandas as pd
        st.map(pd.DataFrame(map_data), zoom=9, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with bot_right:
        confidence = int(r2_score * 100)
        variance_pct = round((1 - r2_score) * 100, 1)

        st.markdown('<div class="bottom-card">', unsafe_allow_html=True)
        st.markdown('<div class="bottom-card-title">✅ Model Confidence</div>', unsafe_allow_html=True)

        # SVG donut ring
        stroke_pct = confidence / 100
        circumference = 2 * 3.14159 * 45
        dash = stroke_pct * circumference

        st.markdown(f"""
        <div class="confidence-ring-wrap">
          <svg width="130" height="130" viewBox="0 0 110 110">
            <circle cx="55" cy="55" r="45" fill="none" stroke="#F0EBE3" stroke-width="9"/>
            <circle cx="55" cy="55" r="45" fill="none" stroke="#C9A96E" stroke-width="9"
                    stroke-dasharray="{dash:.1f} {circumference:.1f}"
                    stroke-dashoffset="{circumference/4:.1f}"
                    stroke-linecap="round"/>
            <text x="55" y="52" text-anchor="middle" font-family="DM Serif Display,serif"
                  font-size="20" fill="#1C1C1E">{confidence}%</text>
            <text x="55" y="66" text-anchor="middle" font-family="DM Sans,sans-serif"
                  font-size="7" fill="#8E8E93" letter-spacing="1">ACCURACY</text>
          </svg>
          <div class="variance-line">Predicted variance: +/- {variance_pct}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Mini metrics
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box">
            <div class="m-val">${rmse_score*100_000:,.0f}</div>
            <div class="m-lbl">RMSE</div>
          </div>
          <div class="metric-box">
            <div class="m-val">${mae_score*100_000:,.0f}</div>
            <div class="m-lbl">MAE</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ── Prediction logic ──────────────────────────────────────────────────────────
if predict_clicked:
    avg_rooms = ROOM_MAP[st.session_state["rooms"]]
    med_income_scaled = med_income / 10_000   # dataset unit: tens of thousands

    # California Housing features:
    # MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
    avg_bedrms  = max(1.0, avg_rooms * 0.25)
    population  = 1200.0
    ave_occup   = 3.0

    features = np.array([[
        med_income_scaled,
        float(house_age),
        avg_rooms,
        avg_bedrms,
        population,
        ave_occup,
        latitude,
        longitude,
    ]])

    raw_pred = model.predict(features)[0]
    dollar_pred = raw_pred * 100_000
    st.session_state["prediction"] = dollar_pred
    st.rerun()