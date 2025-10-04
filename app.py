import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import os

# Add folder with feature_engineering.py to path
sys.path.append(os.path.join(os.getcwd(), "Full_Pipeline_House_Price_Prediction", "src"))
from feature_engineering import RealEstateFeatureEngineer

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(page_title="Real Estate AI App", page_icon="🏡", layout="wide")
st.title("🏡 Real Estate AI Platform")

# ----------------------------
# Load Saved Pipeline & Models
# ----------------------------
PRICE_MODEL_PATH = "Full_Pipeline_House_Price_Prediction/models/real_estate_pipeline_adaboost.joblib"
TS_MODELS_PATH = "House_Price_Prediction_Time_Series_Forecasting/models/all_states_models.pkl"

saved = joblib.load(PRICE_MODEL_PATH)
pipeline = saved['pipeline']        # Full pipeline (encoder + scaler + model)
features = saved['features']        # Features used in training pipeline

ts_models = joblib.load(TS_MODELS_PATH)

# ----------------------------
# Sidebar Menu
# ----------------------------
menu = ["Real Estate Price Prediction", "House Price Time Series Forecasting"]
choice = st.sidebar.radio("Select Task", menu)

# ----------------------------
# PRICE PREDICTION
# ----------------------------
# ----------------------------
# PRICE PREDICTION
# ----------------------------
if choice == "Real Estate Price Prediction":
    st.header("🔹 House Price Prediction")

    # Collect raw user inputs
    col1, col2 = st.columns(2)
    with col1:
        location = st.text_input("Location", "")
        city = st.text_input("City", "")
        bhk = st.number_input("BHK", min_value=1, step=1)
    with col2:
        total_area = st.number_input("Total Area (sqft)", min_value=100.0, step=10.0)
        price_per_sqft = st.number_input("Price per sqft", min_value=500.0, step=50.0)
        bathrooms = st.number_input("Bathrooms", min_value=1, step=1)
        balcony = st.selectbox("Balcony", ["Yes", "No"])

    if st.button("Predict Price"):
        try:
            # 1️⃣ Initialize feature engineer (no outlier removal in inference)
            fe = RealEstateFeatureEngineer(remove_outliers=False)

            # 2️⃣ Build raw input DataFrame
            input_df = pd.DataFrame([{
                "Location": f"{location}, {city}" if city else location,
                "Property Title": f"{bhk} BHK Apartment",
                "Total_Area": total_area,
                "Price_per_SQFT": price_per_sqft,
                "Baths": bathrooms,
                "Balcony": balcony
            }])

            # 3️⃣ Apply feature engineering to generate all derived features
            input_df = fe.transform(input_df)

            # 4️⃣ Keep only features used in training pipeline
            input_df = input_df[features]

            # 5️⃣ Predict using the saved pipeline
            prediction = pipeline.predict(input_df)[0]
            st.success(f"🏠 Predicted House Price: ₹ {prediction:,.2f} Lakhs")

            # Optional: show debug info
            st.write("🔍 DEBUG: Input shape:", input_df.shape)
            st.write("🔍 DEBUG: Features used:", list(input_df.columns))

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ----------------------------
# TIME SERIES FORECASTING
# ----------------------------
elif choice == "House Price Time Series Forecasting":
    st.header("🔹 House Price Time Series Forecasting")
    try:
        available_regions = list(ts_models.keys()) if isinstance(ts_models, dict) else []
        st.write("🔍 Available regions:", available_regions)

        if available_regions:
            region = st.selectbox("Select Region", available_regions)
            horizon = st.number_input("Forecast Horizon (months)", min_value=1, max_value=60, step=1)

            if st.button("Forecast"):
                model = ts_models[region]
                if isinstance(model, dict):
                    st.warning("⚠ Selected region model is a dictionary, using first entry.")
                    model = list(model.values())[0]

                # Prophet forecast
                future = pd.DataFrame({"ds": pd.date_range(start=pd.Timestamp.today(), periods=horizon, freq="M")})
                forecast = model.predict(future)
                forecast_df = forecast[["ds", "yhat"]].rename(columns={"ds":"Month","yhat":"Forecasted Price"})

                st.write(forecast_df)
                st.line_chart(forecast_df.set_index("Month"))
        else:
            st.warning("⚠ No regions found in time series models.")
    except Exception as e:
        st.error(f"Error loading time series models: {e}")
