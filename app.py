import streamlit as st
import requests
import pandas as pd

# ----------------------------
# Streamlit App Config
# ----------------------------
st.set_page_config(page_title="Real Estate AI App", page_icon="🏡", layout="wide")
st.title("🏡 Real Estate AI Platform")

# Backend API URL
API_URL = "http://127.0.0.1:8000"

menu = ["Real Estate Price Prediction", "House Price Time Series Forecasting"]
choice = st.sidebar.radio("Select Task", menu)

# ----------------------------
# PRICE PREDICTION
# ----------------------------
if choice == "Real Estate Price Prediction":
    st.header("🔹 House Price Prediction")

    # User Inputs
    col1, col2 = st.columns(2)
    with col1:
        location = st.text_input("Location", "")
        bhk = st.number_input("BHK", min_value=1, step=1)
        city = st.text_input("City", "DefaultCity")
    with col2:
        total_area = st.number_input("Total Area (sqft)", min_value=100.0, step=10.0)
        price_per_sqft = st.number_input("Price per sqft", min_value=500.0, step=50.0)
        bathrooms = st.number_input("Bathrooms", min_value=1, step=1)
        balcony = st.selectbox("Balcony", ["Yes", "No"])

    if st.button("Predict Price"):
        try:
            payload = {
                "Location": location,
                "City": city,
                "BHK": bhk,
                "Total_Area": total_area,
                "Price_per_SQFT": price_per_sqft,
                "Baths": bathrooms,
                "Balcony": balcony
            }

            response = requests.post(f"{API_URL}/predict_price", json=payload)

            if response.status_code == 200:
                predicted_price = response.json()["predicted_price"]
                st.success(f"🏠 Predicted House Price: ₹ {predicted_price:,.2f} Lakhs")
            else:
                st.error(f"API Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

# ----------------------------
# TIME SERIES FORECASTING
# ----------------------------
elif choice == "House Price Time Series Forecasting":
    st.header("🔹 House Price Time Series Forecasting")

    # Fetch available regions from backend
    try:
        regions_response = requests.get(f"{API_URL}/available_regions")

        if regions_response.status_code == 200:
            available_regions = regions_response.json()["regions"]
        else:
            st.error("Failed to fetch available regions from backend.")
            available_regions = []
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
        available_regions = []

    if available_regions:
        region = st.selectbox("Select Region", available_regions)
        horizon = st.number_input("Forecast Horizon (months)", min_value=1, max_value=60, step=1)

        if st.button("Forecast"):
            try:
                payload = {"region": region, "horizon": horizon}
                response = requests.post(f"{API_URL}/forecast", json=payload)

                if response.status_code == 200:
                    forecast_data = response.json()["forecast"]
                    forecast_df = pd.DataFrame(forecast_data)
                    st.dataframe(forecast_df)
                    st.line_chart(forecast_df.set_index("Month"))
                else:
                    st.error(f"API Error: {response.json()['detail']}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
    else:
        st.warning(" No available regions found. Please check if backend is running correctly.")
