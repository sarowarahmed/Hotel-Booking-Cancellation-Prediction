import shap
import streamlit as st
import pandas as pd
import joblib
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# ======================
# 🔽 CONFIG
# ======================
st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

# ======================
# 🔽 LOAD MODEL (Hugging Face)
# ======================
MODEL_URL = "https://huggingface.co/sarowarahmed/hotel-booking-cancellation-model/resolve/main/model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        with open("model.pkl", "wb") as f:
            f.write(requests.get(MODEL_URL).content)
    return joblib.load("model.pkl")

model = load_model()

# ======================
# 🔽 UI HEADER
# ======================
st.title("🏨 Hotel Booking Cancellation Predictor")

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("Predict whether a booking will be canceled with AI")

st.divider()

# ======================
# 🔽 INPUT SECTION
# ======================
st.subheader("📥 Enter Booking Details")

col1, col2, col3 = st.columns(3)

with col1:
    adults = st.slider("Adults", 1, 5, 2)
    children = st.slider("Children", 0, 5, 0)
    weekends = st.slider("Weekend Nights", 0, 10, 1)
    weekdays = st.slider("Weekday Nights", 0, 20, 2)
    lead_time = st.slider("Lead Time", 0, 500, 50)

with col2:
    meal_type = st.selectbox("Meal Type", ["BB", "HB", "FB", "SC"])
    room_type = st.selectbox("Room Type", ["A", "B", "C", "D", "E", "F", "G"])
    segment = st.selectbox("Market Segment", ["Online", "Offline", "Corporate", "Direct"])
    deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Refundable", "Non Refund"])
    repeat = st.selectbox("Repeated Guest", [0, 1])

with col3:
    price = st.number_input("Price (ADR)", 0.0, 500.0, 100.0)
    requests = st.slider("Special Requests", 0, 5, 0)
    arrival_month = st.selectbox("Arrival Month", list(range(1, 13)))
    arrival_day = st.slider("Arrival Day", 1, 31, 15)

# ======================
# 🔽 FEATURE ENGINEERING (MUST MATCH NOTEBOOK)
# ======================
arrival = arrival_month * 30 + arrival_day  # approximate (same logic as notebook)

price_outlier = 1 if price > 300 else 0
lead_time_outlier = 1 if lead_time > 200 else 0

# ======================
# 🔽 CREATE INPUT DATA
# ======================
input_data = pd.DataFrame({
    "adults": [adults],
    "children": [children],
    "weekends": [weekends],
    "weekdays": [weekdays],
    "meal_type": [meal_type],
    "room_type": [room_type],
    "arrival": [arrival],
    "lead_time": [lead_time],
    "segment": [segment],
    "repeat": [repeat],
    "price": [price],
    "requests": [requests],
    "price_outlier": [price_outlier],
    "lead_time_outlier": [lead_time_outlier],
    "arrival_month": [arrival_month],
    "arrival_day": [arrival_day]
})

# ======================
# 📊 QUICK INSIGHTS
# ======================
st.subheader("📊 Booking Insights")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig = px.bar(
        x=["Lead Time", "ADR", "Requests"],
        y=[lead_time, price, requests],
        title="Booking Feature Overview"
    )
    st.plotly_chart(fig, width='stretch')

with chart_col2:
    fig = px.pie(
        names=["Adults", "Children"],
        values=[adults, children],
        title="Guest Composition"
    )
    st.plotly_chart(fig, width='stretch')

# ======================
# 🔽 PREDICTION
# ======================
st.divider()

if st.button("🔮 Predict Cancellation"):

    prediction = model.predict(input_data)[0]

    # Probability (if supported)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0][1]
    else:
        proba = None

    # ======================
    # 🎯 RESULT DISPLAY
    # ======================
    if prediction == 1:
        st.error("❌ High Risk: Booking will likely be CANCELED")
    else:
        st.success("✅ Low Risk: Booking will likely NOT be canceled")

    # ======================
    # 🔍 SHAP EXPLANATION
    # ======================
    st.subheader("🧠 Why this prediction?")

    try:
        preprocessor = model.named_steps['preprocessor']
        model_step = model.named_steps['clf']

        # Transform input
        X_transformed = preprocessor.transform(input_data)

        explainer = shap.TreeExplainer(model_step)
        shap_values = explainer.shap_values(X_transformed)

        fig, ax = plt.subplots()
        shap.plots.waterfall(explainer(X_transformed)[0])
        st.pyplot(fig)

    except Exception as e:
        st.error(f"SHAP error: {e}")

    # ======================
    # 💡 BUSINESS INSIGHT
    # ======================
    
    st.subheader("💡 Business Insight")

    if prediction == 1:
        st.write("""
        🔴 This booking has high cancellation risk.
    
        Recommended Actions:
        - Require partial prepayment
        - Send confirmation reminders
        - Offer discounts for commitment
        """)
    else:
        st.write("""
        🟢 This booking is reliable.
    
        Recommended Actions:
        - Upsell premium services
        - Offer loyalty benefits
        """)

    # ======================
    # 📊 PROBABILITY METER
    # ======================
    if proba is not None:
        st.subheader("📊 Cancellation Probability")

        st.progress(int(proba * 100))
        st.write(f"**Risk Score:** {proba:.2%}")

        # Interpretation
        if proba > 0.7:
            st.warning("⚠️ Very High Risk")
        elif proba > 0.4:
            st.info("⚠️ Moderate Risk")
        else:
            st.success("✅ Low Risk")

    # ======================
    # 📈 FEATURE IMPORTANCE
    # ======================
    st.subheader("📈 Feature Importance")

    try:
        preprocessor = model.named_steps['preprocessor']
        model_step = model.named_steps['clf']
        
        importances = model_step.feature_importances_
        feature_names = preprocessor.get_feature_names_out()

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)

        st.bar_chart(importance_df.set_index("Feature"))

    except Exception as e:
        st.error(f"Feature importance error: {e}")

# ======================
# 🔽 FOOTER
# ======================
st.divider()
st.caption("Built by Sarowar Ahmed | Machine Learning Project")
