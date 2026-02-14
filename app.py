
import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="House Rent Prediction", layout="centered")

st.title("🏠 House Rent Prediction App")
st.write("Enter house details to predict rent price")

# Load saved model & scaler
model = joblib.load("house_rent_linear_model.pkl")
scaler = joblib.load("scaler.pkl")

# User Inputs
area = st.number_input("Area (in sqft)", min_value=100, max_value=10000, value=1000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=2)
washrooms = st.number_input("Number of Washrooms", min_value=1, max_value=10, value=2)

# Predict Button
if st.button("Predict Rent 💰"):

    new_house = pd.DataFrame({
        'Area': [area],
        'Bedrooms': [bedrooms],
        'Washrooms': [washrooms]
    })

    # Match training columns
    new_house = pd.get_dummies(new_house)
    
    # IMPORTANT: training time lo use chesina columns match cheyyali
    x_columns = joblib.load("scaler.pkl")  # dummy load just for safety
    
    new_house = new_house.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Scale input
    new_scaled = scaler.transform(new_house)

    # Predict
    prediction = model.predict(new_scaled)

    st.success(f"🏷️ Predicted House Rent: ₹ {int(prediction[0])}")
