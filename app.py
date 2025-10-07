import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------------------------------
# ✅ Streamlit Page Config (MUST BE FIRST COMMAND)
# -------------------------------------------------
st.set_page_config(page_title="Real Estate Analytics", layout="wide")

# -------------------------------------------------
# Load saved models and data
# -------------------------------------------------
@st.cache_resource
def load_price_pipeline():
    """Load trained Random Forest pipeline and top localities"""
    pipeline = joblib.load("real_estate_pipeline (1).pkl")
    info = joblib.load("real_estate_pipeline_info.pkl")
    return pipeline, info["top_localities"]


@st.cache_data
def load_forecast_data():
    """Load region-level forecast data and saved Prophet plots"""
    forecast_df = pd.read_csv("all_regions_forecast (1).csv")
    forecast_plots = joblib.load("all_forecasts_plots.joblib")
    return forecast_df, forecast_plots


# -------------------------------------------------
# Load Models
# -------------------------------------------------
pipeline, top_localities = load_price_pipeline()
all_regions, forecast_plots = load_forecast_data()

# -------------------------------------------------
# Streamlit Layout
# -------------------------------------------------
st.title("🏙️ Real Estate Analytics Dashboard")
tab1, tab2 = st.tabs(["📊 Price Prediction", "📈 Forecasting"])

# =================================================
# TAB 1: PRICE PREDICTION
# =================================================
with tab1:
    st.header("🏠 Price Prediction")

    col1, col2 = st.columns(2)
    with col1:
        total_area = st.number_input("Total Area (in sqft)", min_value=100, value=1200)
        price_per_sqft = st.number_input("Price per Sqft (₹)", min_value=500, value=5500)
        baths = st.number_input("Number of Bathrooms", min_value=1, value=2)
    with col2:
        city = st.text_input("City", "Bangalore")
        locality = st.text_input("Locality", "Whitefield")

    if st.button("🔍 Predict Price"):
        # Prepare DataFrame for model
        df_input = pd.DataFrame([{
            "Total_Area": total_area,
            "Price_per_SQFT": price_per_sqft,
            "Baths": baths,
            "City": city,
            "Locality": locality
        }])

        # Feature Engineering (same as training)
        df_input["log_area"] = np.log1p(df_input["Total_Area"])
        df_input["Area_per_Room"] = df_input["Total_Area"] / np.maximum(df_input["Baths"], 1)
        df_input["Locality"] = df_input["Locality"].apply(
            lambda x: x if x in top_localities else "Other"
        )

        # Prediction
        pred_price = pipeline.predict(df_input)[0]
        st.success(f"💰 Predicted Property Price: ₹{pred_price:.2f} Lakhs")

# =================================================
# TAB 2: REGIONAL FORECASTING
# =================================================
with tab2:
    st.header("📈 Regional Price Forecasting")

    # Ensure correct column name
    region_col = "RegionName" if "RegionName" in all_regions.columns else "Region"
    regions = sorted(all_regions[region_col].unique())

    selected_region = st.selectbox("🏘️ Select Region", regions)
    months_ahead = st.slider("⏳ Forecast Period (months)", 1, 24, 6)

    if st.button("🔮 Generate Forecast"):
        # Filter forecast data for selected region
        df_region_forecast = all_regions[all_regions[region_col] == selected_region].copy()

        if df_region_forecast.empty:
            st.error("❌ Forecast data not available for the selected region.")
        else:
            df_region_forecast = df_region_forecast.head(months_ahead)
            st.subheader(f"Forecast for {selected_region} (next {months_ahead} months)")
            st.dataframe(df_region_forecast)

            # Display saved Prophet forecast plots (side-by-side)
            if selected_region in forecast_plots:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(forecast_plots[selected_region]["forecast_plot"], caption="Forecast Plot")
                with col2:
                    st.image(forecast_plots[selected_region]["components_plot"], caption="Trend Components")
            else:
                st.warning("⚠️ No plot available for this region.")
