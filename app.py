import streamlit as st
import pandas as pd
import joblib
import requests
import os
import numpy as np
import matplotlib.pyplot as plt

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
st.markdown("Predict whether a booking will be canceled with AI")

st.divider()

# ======================
# 🔽 INPUT SECTION
# ======================
col1, col2 = st.columns(2)

with col1:
    lead_time = st.slider("Lead Time (days)", 0, 500, 50)
    adr = st.number_input("Average Daily Rate (ADR)", 0.0, 500.0, 100.0)
    previous_cancellations = st.slider("Previous Cancellations", 0, 10, 0)

with col2:
    deposit_type = st.selectbox(
        "Deposit Type",
        ["No Deposit", "Refundable", "Non Refund"]
    )
    total_guests = st.slider("Total Guests", 1, 10, 2)
    total_stay = st.slider("Total Nights", 1, 30, 3)

# ======================
# 🔽 CREATE INPUT DATA
# ======================
input_data = pd.DataFrame({
    "lead_time": [lead_time],
    "adr": [adr],
    "previous_cancellations": [previous_cancellations],
    "deposit_type": [deposit_type],
    "total_guests": [total_guests],
    "total_stay": [total_stay]
})

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
        model_step = model.named_steps['model']
        importances = model_step.feature_importances_

        features = model.named_steps['preprocessor'].get_feature_names_out()

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)

        fig, ax = plt.subplots()
        ax.barh(importance_df["Feature"], importance_df["Importance"])
        ax.invert_yaxis()

        st.pyplot(fig)

    except:
        st.info("Feature importance not available for this model.")

# ======================
# 🔽 FOOTER
# ======================
st.divider()
st.caption("Built by Sarowar Ahmed | Machine Learning Project")
