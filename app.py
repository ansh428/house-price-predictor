import streamlit as st
import numpy as np
import joblib

# Load the trained Gradient Boosting model
gbr_model = joblib.load("gradient_boosting_model.pkl")

# Function to preprocess input data
def preprocess_input(features):
    """Ensure input data is properly formatted."""
    return np.array([features])

# Set page layout and style
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton button {
            background-color: #ff4b4b !important;
            color: white !important;
            font-size: 16px !important;
            border-radius: 10px !important;
        }
        .stTextInput, .stNumberInput {
            border-radius: 5px !important;
        }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.title("🏠 California House Price Predictor")
st.markdown("### Enter the house details below to get a **predicted price**!")

# User input fields (with two columns for better layout)
col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("📍 Longitude", value=-121.5, format="%.2f")
    latitude = st.number_input("📍 Latitude", value=37.7, format="%.2f")
    housing_median_age = st.number_input("🏡 Housing Median Age", value=30)
    total_rooms = st.number_input("🛏️ Total Rooms", value=6000)
    total_bedrooms = st.number_input("🛏️ Total Bedrooms", value=1200)

with col2:
    population = st.number_input("👨‍👩‍👧 Population", value=2000)
    households = st.number_input("🏠 Households", value=800)
    median_income = st.number_input("💰 Median Income ($1000s)", value=5.0)

# Ocean proximity selection (one-hot encoding)
st.markdown("### 🌊 Select Ocean Proximity:")
ocean_options = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
ocean_proximity = st.radio("🏖️ Ocean Proximity", ocean_options, horizontal=True)

# Convert to one-hot encoding
ocean_encoding = [1 if ocean_proximity == option else 0 for option in ocean_options]

# Predict button
if st.button("🔍 Predict House Price"):
    # Combine all input features
    input_features = [longitude, latitude, housing_median_age, total_rooms, 
                      total_bedrooms, population, households, median_income] + ocean_encoding

    # Preprocess and predict
    processed_features = preprocess_input(input_features)
    predicted_price = gbr_model.predict(processed_features)

    # Display result with better formatting
    st.success(f"🏡 **Estimated House Price: ${predicted_price[0]:,.2f}**")
