import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Models
lr_model = joblib.load("linear_regression_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

# Title
st.title("🌾 Crop Yield Prediction App")

# User Input Form
st.header("Enter Crop and Soil Details")

state = st.selectbox("State", ["Andhra Pradesh", "Karnataka", "Maharashtra", "Tamil Nadu"])  # Update with your actual categories
crop = st.selectbox("Crop", ["Rice", "Wheat", "Maize", "Sugarcane"])
fertilizer = st.selectbox("Fertilizer Type", ["Urea", "DAP", "MOP", "Complex"])
area = st.number_input("Area (hectares)", min_value=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
ph = st.number_input("Soil pH", min_value=0.0)

# Prediction
if st.button("Predict"):
    input_dict = {
        "Area": area,
        "Rainfall": rainfall,
        "Temperature": temperature,
        "Soil_pH": ph,
        f"State_{state}": 1,
        f"Crop_{crop}": 1,
        f"Fertilizer_Type_{fertilizer}": 1
    }

    # Create DataFrame with all encoded columns (dummy row with zeros)
    base_df = pd.DataFrame([0]*len(lr_model.coef_), index=lr_model.feature_names_in_).T
    for key in input_dict:
        if key in base_df.columns:
            base_df[key] = input_dict[key]

    # Make prediction
    lr_pred = lr_model.predict(base_df)[0]
    rf_pred = rf_model.predict(base_df)[0]

    st.success(f"📈 Linear Regression Prediction: **{lr_pred:.2f} tons/ha**")
    st.success(f"🌲 Random Forest Prediction: **{rf_pred:.2f} tons/ha**")
