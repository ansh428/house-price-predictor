import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static

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

# Set up Streamlit UI
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")

# 🎨 Dark Mode & Styling
st.markdown("""
    <style>
        .main { background-color: #1E1E1E; color: white; }
        .stButton button { background-color: #FF4B4B !important; color: white !important; font-size: 16px !important; border-radius: 10px !important; }
        .stTextInput, .stNumberInput { border-radius: 5px !important; background-color: #292929 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# 🏡 App Title
st.title("🏠 California House Price Predictor")
st.markdown("### Enter the house details below to get a **predicted price**!")

# 🗺️ Interactive Map for Selecting Location
st.markdown("### 📍 Select a Location on the Map")

# Default location (California)
default_location = {"lat": 37.7, "lon": -121.5}
m = folium.Map(location=[default_location["lat"], default_location["lon"]], zoom_start=6)
marker = folium.Marker([default_location["lat"], default_location["lon"]], draggable=True)
m.add_child(marker)
folium_static(m)

# 🏡 Other Input Features
col1, col2 = st.columns(2)
with col1:
    housing_median_age = st.number_input("🏡 Housing Median Age", value=30)
    total_rooms = st.number_input("🛏️ Total Rooms", value=6000)
    total_bedrooms = st.number_input("🛏️ Total Bedrooms", value=1200)

with col2:
    population = st.number_input("👨‍👩‍👧 Population", value=2000)
    households = st.number_input("🏠 Households", value=800)
    median_income = st.number_input("💰 Median Income ($1000s)", value=5.0)

# 🌊 Ocean proximity selection (one-hot encoding)
st.markdown("### 🌊 Select Ocean Proximity:")
ocean_options = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
ocean_proximity = st.radio("🏖️ Ocean Proximity", ocean_options, horizontal=True)
ocean_encoding = [1 if ocean_proximity == option else 0 for option in ocean_options]

# 🔍 Predict Button
if st.button("🔍 Predict House Price"):
    # Get marker location from the interactive map
    latitude, longitude = marker.location

    # Combine input features
    input_features = [longitude, latitude, housing_median_age, total_rooms, total_bedrooms, 
                      population, households, median_income] + ocean_encoding

    # Preprocess & Predict
    processed_features = preprocess_input(input_features)
    predicted_price = gbr_model.predict(processed_features)

    # Display result
    st.success(f"🏡 **Estimated House Price: ${predicted_price[0]:,.2f}**")

    # Save prediction to history
    new_entry = pd.DataFrame([[longitude, latitude, households, median_income, predicted_price[0]]], 
                             columns=["Longitude", "Latitude", "Households", "Median Income", "Prediction"])
    history_df = pd.concat([history_df, new_entry], ignore_index=True)
    history_df.to_csv("predictions.csv", index=False)

# 🌎 House Price Heatmap
st.markdown("### 🌎 California House Price Heatmap")

# Create map centered in California
m = folium.Map(location=[37.5, -119.5], zoom_start=6)

# Example house price data points
house_data = [
    {"lat": 34.05, "lon": -118.25, "price": 500000},  # Los Angeles
    {"lat": 37.77, "lon": -122.41, "price": 750000},  # San Francisco
    {"lat": 32.72, "lon": -117.16, "price": 650000},  # San Diego
]

# Add price markers
for house in house_data:
    folium.Marker(
        location=[house["lat"], house["lon"]],
        popup=f"Price: ${house['price']:,.0f}",
        icon=folium.Icon(color="blue"),
    ).add_to(m)

# Display heatmap
folium_static(m)

# 📊 Feature Importance Visualization
st.markdown("### 📊 Feature Importance")
st.markdown("Which factors impact house prices the most?")

feature_importances = gbr_model.feature_importances_
feature_names = ["Longitude", "Latitude", "Housing Age", "Total Rooms", "Total Bedrooms", 
                 "Population", "Households", "Median Income", "<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

feat_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=feat_df, ax=ax)
st.pyplot(fig)

# 📈 Past Predictions
if not history_df.empty:
    st.markdown("### 📈 Past Predictions")
    st.dataframe(history_df.tail(10))  # Show last 10 predictions

    # 🗑️ Reset Button
    if st.button("🗑️ Reset Past Predictions"):
        history_df = pd.DataFrame(columns=["Longitude", "Latitude", "Households", "Median Income", "Prediction"])
        history_df.to_csv("predictions.csv", index=False)
        st.warning("Past predictions have been reset!")
        st.rerun() # Refresh app

    # 📊 House Price Trends Over Time
    st.markdown("### 📊 House Price Predictions Over Time")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df.index, history_df["Prediction"], marker="o", linestyle="-", color="blue")
    ax.set_xlabel("Prediction Number")
    ax.set_ylabel("House Price ($)")
    ax.set_title("Trend of Predicted House Prices")
    st.pyplot(fig)
