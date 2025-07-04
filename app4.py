import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config with a custom title and layout
st.set_page_config(page_title="Crop Yield Predictor", layout="wide")

# Apply custom background with CSS
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({image_url});
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image (use a public link or raw GitHub image link)
set_background("https://images.unsplash.com/photo-1582281298051-01009a066b2e")  # Replace with your image

# Display logo
st.image("logo.png", width=150)  # Make sure 'logo.png' is in the same folder

# Title and Description
st.markdown("<h1 style='text-align: center; color: green;'>🌾 Crop Yield Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Predict crop yield using weather, soil, and fertilizer input</h4>", unsafe_allow_html=True)
st.write("")

# Load models
lr_model = joblib.load("linear_regression_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

# Sidebar input
st.sidebar.header("🧮 Input Features")
area = st.sidebar.number_input("Area (ha)", min_value=0.1, value=1.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, value=100.0)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, value=25.0)
ph = st.sidebar.number_input("Soil pH", min_value=0.0, value=6.5)

state = st.sidebar.selectbox("State", ["State_Karnataka", "State_Tamil Nadu", "State_Maharashtra"])
crop = st.sidebar.selectbox("Crop", ["Crop_Rice", "Crop_Wheat", "Crop_Maize"])
fertilizer = st.sidebar.selectbox("Fertilizer", ["Fertilizer_Type_Urea", "Fertilizer_Type_DAP"])

# Prepare input
input_data = {
    "Area": area,
    "Rainfall": rainfall,
    "Temperature": temperature,
    "Soil_pH": ph,
    state: 1,
    crop: 1,
    fertilizer: 1
}

all_features = list(lr_model.feature_names_in_)
input_df = pd.DataFrame([0] * len(all_features), index=all_features).T

for col in input_data:
    if col in input_df.columns:
        input_df[col] = input_data[col]

# Predict
if st.button("Predict Crop Yield"):
    lr_pred = lr_model.predict(input_df)[0]
    rf_pred = rf_model.predict(input_df)[0]

    st.success(f"📈 Linear Regression Prediction: **{lr_pred:.2f} tons/ha**")
    st.success(f"🌲 Random Forest Prediction: **{rf_pred:.2f} tons/ha**")

