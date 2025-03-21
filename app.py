
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model_path = "F:/Confirm Ticket3/models/prediction_of_ticket_model.pkl"
model = joblib.load(model_path)

# Streamlit UI
st.set_page_config(page_title="Railway Ticket Confirmation Predictor", layout="wide")

st.markdown("# 🚆 Railway Ticket Confirmation Predictor")

st.write("### Enter details below to check the probability of ticket confirmation.")

booking_status = st.selectbox("**Booking Status** (0: Waiting List, 1: RAC, 2: Confirmed)", [0, 1, 2])
quota = st.selectbox("**Quota** (1: General, 2: Tatkal, 3: Ladies, etc.)", [1, 2, 3, 4, 5])
train_type = st.selectbox("**Train Type** (1: Express, 2: Superfast, 3: Rajdhani, etc.)", [1, 2, 3, 4])
passenger_age = st.number_input("**Passenger Age** (Enter Age in Years)", min_value=1, max_value=100, value=25)
gender = st.selectbox("**Gender** (0: Female, 1: Male)", [0, 1])
days_before_journey = st.number_input("**Days Before Journey** (Days left before travel)", min_value=0, max_value=120, value=30)

# New: Add Waitlist Number Input
waitlist_number = st.number_input("**Waitlist Number** (Enter WL number if on waiting list)", min_value=0, max_value=500, value=0)
wl_display = f"WL {int(waitlist_number)}"

# Display Waitlist Number only if booking status is 'Waiting List'
if booking_status == 0:
    st.write(f"🚦 **Your Waitlist Number:** {wl_display}")

# Prediction Button
if st.button("Predict Confirmation Probability"):
    input_data = np.array([[booking_status, quota, train_type, passenger_age, gender, days_before_journey]])
    prediction = model.predict_proba(input_data)[0][1]  # Probability of confirmation

    st.success(f"✅ Probability of ticket confirmation: **{prediction*100:.2f}%**")
