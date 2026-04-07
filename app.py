import streamlit as st
import requests
import joblib
import os

# Load Model
MODEL_URL = "https://huggingface.co/sarowarahmed/hotel-booking-cancellation-model/resolve/main/model.pkl"

if not os.path.exists("model.pkl"):
    with open("model.pkl", "wb") as f:
        f.write(requests.get(MODEL_URL).content)

model = joblib.load("model.pkl")

st.title("🏨 Hotel Booking Cancellation Predictor")

st.write("Predict whether a booking will be canceled")

# User Inputs
lead_time = st.slider("Lead Time", 0, 500, 50)
adr = st.number_input("Average Daily Rate", 0.0, 500.0, 100.0)
previous_cancellations = st.slider("Previous Cancellations", 0, 10, 0)

deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Refundable", "Non Refund"])

# Convert input to dataframe
input_data = pd.DataFrame({
    "lead_time": [lead_time],
    "adr": [adr],
    "previous_cancellations": [previous_cancellations],
    "deposit_type": [deposit_type]
})

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("❌ Booking will likely be CANCELED")
    else:
        st.success("✅ Booking will likely NOT be canceled")
