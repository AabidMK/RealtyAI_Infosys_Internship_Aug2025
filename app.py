import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import json

# ---------------- LOAD MODELS ---------------- #
@st.cache_resource
def load_price_model():
    return joblib.load("lasso_pipeline.pkl")

@st.cache_data
def load_forecasts():
    df = pd.read_csv("all_region_forecasts.csv")
    if df['ds'].dtype == 'object':
        df['ds'] = pd.to_datetime(df['ds'])
    return df

lasso_pipeline = load_price_model()
forecasts_df = load_forecasts()

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="🏠 Real Estate Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align: center; color: white;'>🏠 Real Estate Analytics Dashboard</h1>", unsafe_allow_html=True)
st.write("---")

# ---------------- SIDEBAR ---------------- #
# ---------------- SIDEBAR BUTTONS ---------------- #
if 'active_model' not in st.session_state:
    st.session_state.active_model = "price"  # default model

st.sidebar.markdown("### Select Model")
if st.sidebar.button("🏡 House Price Prediction"):
    st.session_state.active_model = "price"

if st.sidebar.button("📈 Time Series Forecasting"):
    st.session_state.active_model = "forecast"

# Load JSON with allowed cities and localities
with open("city_locality.json", "r") as f:
    city_locality_dict = json.load(f)

# ---------------- HOUSE PRICE PREDICTION ---------------- #
if st.session_state.active_model == "price":
    st.subheader("🏡 House Price Prediction")
    st.write("Enter the details below to predict the estimated house price.")

    # Using columns for better layout
    col1, col2, col3 = st.columns(3)
    with col1:
        Baths = st.number_input("Number of Baths 🛁", min_value=1, max_value=10, value=2)
        BHK = st.number_input("BHK 🏢", min_value=1, max_value=10, value=2)
    
    with col2:
        Balcony = st.selectbox("Balcony 🌅", ["Yes", "No"])
    
    with col3:
        city_selected = st.selectbox("City 🌆", options=list(city_locality_dict.keys()))
        locality_selected = st.selectbox("Locality 🏘", options=city_locality_dict[city_selected])

    DEFAULT_TOTAL_AREA = 1000
    DEFAULT_PRICE_PER_SQFT = 5000

    if st.button("Predict Price 💰", type="primary"):
        Balcony_Cleaned = 1 if Balcony.lower() == "yes" else 0
        Total_Area_Log = np.log(DEFAULT_TOTAL_AREA)
        Price_per_SQFT_Log = np.log(DEFAULT_PRICE_PER_SQFT)

        input_df = pd.DataFrame({
            "Total_Area_Log": [Total_Area_Log],
            "Price_per_SQFT_Log": [Price_per_SQFT_Log],
            "Baths": [Baths],
            "Balcony_Cleaned": [Balcony_Cleaned],
            "BHK": [BHK],
            "City": [city_selected],
            "Locality": [locality_selected]
        })

        prediction = lasso_pipeline.predict(input_df)[0]

        st.markdown("### Estimated House Price")
        st.metric(label="💸 Price (in Lakhs ₹)", value=f"{prediction:.2f}")

# ---------------- TIME SERIES FORECASTING ---------------- #
elif st.session_state.active_model == "forecast":
    st.subheader("📈 Housing Price Forecast (ZHVI_AllHomes)")
    regions = forecasts_df['RegionName'].unique().tolist()
    selected_region = st.selectbox("Select Region", regions)

    if st.button("Show Forecast 📊", key="forecast_show"):
        region_forecast = forecasts_df[forecasts_df['RegionName'] == selected_region]

        if not region_forecast.empty:
            plt.figure(figsize=(14,6))
            plt.plot(region_forecast['ds'], region_forecast['yhat'], label="Forecast", color="#FF5733", linewidth=2)
            plt.fill_between(region_forecast['ds'],
                             region_forecast['yhat_lower'],
                             region_forecast['yhat_upper'],
                             color="#FFC300",
                             alpha=0.3,
                             label="Confidence Interval")
            plt.title(f"{selected_region} Housing Price Forecast", fontsize=16)
            plt.xlabel("Date")
            plt.ylabel("ZHVI_AllHomes")
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(plt)
        else:
            st.warning("No forecast data available for this region.")
