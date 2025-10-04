import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys, os

# ----------------------------
# Add src folder to Python path to fix imports
# ----------------------------
sys.path.append(os.path.join(os.getcwd(), "Full_Pipeline_House_Price_Prediction", "src"))
from inference import RealEstatePredictor  # feature_engineering import works inside inference.py

# ----------------------------
# Model Paths
# ----------------------------
PRICE_MODEL_PATH = "Full_Pipeline_House_Price_Prediction/models/real_estate_pipeline_adaboost.joblib"
TS_MODELS_PATH = "House_Price_Prediction_Time_Series_Forecasting/models/all_states_models.pkl"

# ----------------------------
# Streamlit App Config
# ----------------------------
st.set_page_config(page_title="Real Estate AI App", page_icon="🏡", layout="wide")
st.title("🏡 Real Estate AI Platform")

menu = ["Real Estate Price Prediction", "House Price Time Series Forecasting"]
choice = st.sidebar.radio("Select Task", menu)

# ----------------------------
# Cache the predictor for efficiency
# ----------------------------
@st.cache_resource
def load_price_predictor():
    return RealEstatePredictor(model_path=PRICE_MODEL_PATH)

@st.cache_resource
def load_ts_models():
    return joblib.load(TS_MODELS_PATH)

# ----------------------------
# PRICE PREDICTION
# ----------------------------
if choice == "Real Estate Price Prediction":
    st.header("🔹 House Price Prediction")

    predictor = load_price_predictor()

    # Collect user inputs
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
            # Build property dictionary matching pipeline features
            property_data = {
                "Location": location,
                "Property Title": f"{bhk} BHK Apartment",
                "Total_Area": total_area,
                "Price_per_SQFT": price_per_sqft,
                "Baths": bathrooms,
                "Balcony": balcony,
                "City": city
            }

            # Predict price
            prediction = predictor.predict(property_data)
            st.success(f"🏠 Predicted House Price: ₹ {prediction:,.2f}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ----------------------------
# TIME SERIES FORECASTING
# ----------------------------
elif choice == "House Price Time Series Forecasting":
    st.header("🔹 House Price Time Series Forecasting")

    try:
        ts_models = load_ts_models()
        available_regions = list(ts_models.keys()) if isinstance(ts_models, dict) else []
        st.write("🔍 DEBUG: Available regions:", available_regions)

        if available_regions:
            region = st.selectbox("Select Region", available_regions)
            horizon = st.number_input("Forecast Horizon (months)", min_value=1, max_value=60, step=1)

            if st.button("Forecast"):
                model = ts_models[region]

                # Handle dict inside dict case
                if isinstance(model, dict):
                    st.warning("⚠ Selected region model is a dictionary, using first entry.")
                    model = list(model.values())[0]

                # Create future DataFrame for Prophet
                future = pd.DataFrame({
                    "ds": pd.date_range(start=pd.Timestamp.today(), periods=horizon, freq="M")
                })
                forecast = model.predict(future)

                forecast_df = forecast[["ds", "yhat"]].rename(
                    columns={"ds": "Month", "yhat": "Forecasted Price"}
                )

                st.write(forecast_df)
                st.line_chart(forecast_df.set_index("Month"))
        else:
            st.warning("⚠ No regions found in time series models.")

    except Exception as e:
        st.error(f"Error loading time series models: {e}")
