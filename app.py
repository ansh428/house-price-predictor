import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained Gradient Boosting model
gbr_model = joblib.load("gradient_boosting_model.pkl")

# Load past predictions if available
try:
    history_df = pd.read_csv("predictions.csv")
except FileNotFoundError:
    history_df = pd.DataFrame(columns=["Longitude", "Latitude", "Households", "Median Income", "Prediction"])

# Function to preprocess input data
def preprocess_input(features):
    return np.array([features])

# Streamlit UI
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")

# Custom styling
st.markdown("""
    <style>
        .main { background-color: #f5f5f5; }
        .stButton button { background-color: #ff4b4b !important; color: white !important; font-size: 16px !important; border-radius: 10px !important; }
        .stTextInput, .stNumberInput { border-radius: 5px !important; }
    </style>
    """, unsafe_allow_html=True)

# App title
st.title("🏠 California House Price Predictor")
st.markdown("### Enter the house details below to get a **predicted price**!")

# User input fields
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
    # Combine input features
    input_features = [longitude, latitude, housing_median_age, total_rooms, total_bedrooms, 
                      population, households, median_income] + ocean_encoding

    # Preprocess and predict
    processed_features = preprocess_input(input_features)
    predicted_price = gbr_model.predict(processed_features)

    # Display result
    st.success(f"🏡 **Estimated House Price: ${predicted_price[0]:,.2f}**")

    # Save prediction to history
    new_entry = pd.DataFrame([[longitude, latitude, households, median_income, predicted_price[0]]], 
                             columns=["Longitude", "Latitude", "Households", "Median Income", "Prediction"])
    history_df = pd.concat([history_df, new_entry], ignore_index=True)
    history_df.to_csv("predictions.csv", index=False)

# Feature importance visualization
st.markdown("### 📊 Feature Importance")
st.markdown("Which factors impact house prices the most?")

# Load feature importances
feature_importances = gbr_model.feature_importances_
feature_names = ["Longitude", "Latitude", "Housing Age", "Total Rooms", "Total Bedrooms", 
                 "Population", "Households", "Median Income", "<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

# Create DataFrame
feat_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False)

# Plot feature importance
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=feat_df, ax=ax)
st.pyplot(fig)

# Display past predictions
if not history_df.empty:
    st.markdown("### 📈 Past Predictions")
    st.dataframe(history_df.tail(10))  # Show last 10 predictions

    # Plot predictions over time
    st.markdown("### 📊 House Price Predictions Over Time")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df.index, history_df["Prediction"], marker="o", linestyle="-", color="blue")
    ax.set_xlabel("Prediction Number")
    ax.set_ylabel("House Price ($)")
    ax.set_title("Trend of Predicted House Prices")
    st.pyplot(fig)
