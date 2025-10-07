import streamlit as st
import joblib
import pandas as pd
from prophet import Prophet
import sklearn.compose._column_transformer
import os

# -----------------------------
# 🧩 Compatibility Patch for Older sklearn Pipelines
# -----------------------------
if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
    sklearn.compose._column_transformer._RemainderColsList = list

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="🏡 Real Estate Prediction Dashboard", layout="wide")
st.title("🏡 Real Estate Price Prediction & Forecast Dashboard")

# -----------------------------
# File Paths
# -----------------------------
# -----------------------------
# File Paths
# -----------------------------
lr_model_path = r"C:\Users\vthhr\Desktop\real_estate_pipeline.pkl"
prophet_models_path = r"C:\Users\vthhr\Desktop\all_regions_prophet_models.pkl"

# -----------------------------
# ✅ Load Models (Safe, Cached)
# -----------------------------# -----------------------------
# ✅ Load Models (No Cache)
# -----------------------------
def load_models():
    """Load both Linear Regression and Prophet models safely once."""
    lr_model = None
    all_prophet_models = {}
    regions = []

    # --- Load LR model (NO CACHE) ---
    if os.path.exists(lr_model_path):
        print("🧩 Loading Linear Regression model...")
        try:
            lr_model = joblib.load(lr_model_path)
            print("✅ Linear Regression model loaded successfully.")
        except Exception as e:
            st.error(f"❌ Error loading Linear Regression model: {e}")
    else:
        st.warning("⚠ Linear Regression model not found.")

    # --- Load Prophet models (still cached safely) ---
    if os.path.exists(prophet_models_path):
        print("🧩 Loading Prophet models...")
        try:
            loaded_models = joblib.load(prophet_models_path)
            if isinstance(loaded_models, dict):
                all_prophet_models = {
                    r: m for r, m in loaded_models.items() if isinstance(m, Prophet)
                }
            else:
                st.warning(f"⚠ Prophet file is not a dict (got {type(loaded_models)}).")
        except Exception as e:
            st.error(f"❌ Error loading Prophet models: {e}")
    else:
        st.warning("⚠ Prophet models file not found.")

    regions = list(all_prophet_models.keys())
    return lr_model, all_prophet_models, regions


# ✅ Load models once (no cache for list issues)
lr_model, all_prophet_models, regions = load_models()


# -----------------------------
# 🧮 Linear Regression Price Prediction
# -----------------------------
st.header("📊 Linear Regression Price Prediction")

features_input = st.text_input(
    "Enter property features (comma-separated, e.g. 1200, 3, 2, 1):"
)
if st.button("Predict Price"):
    if not lr_model:
        st.error("❌ Linear Regression model not loaded.")
    elif features_input.strip():
        try:
            parts = [x.strip() for x in features_input.split(",")]
            if len(parts) != 9:
                st.error("⚠ Please enter exactly 9 comma-separated values.")
            else:
                # Prepare DataFrame without forcing numeric conversion
                input_df = pd.DataFrame([{
                    "Total_Area": float(parts[0]),
                    "Price_per_SQFT": float(parts[1]),
                    "Baths": float(parts[2]),
                    "Bedrooms": float(parts[3]),
                    "Name": parts[4],
                    "Property Title": parts[5],
                    "Balcony": parts[6],
                    "State": parts[7],   # ✅ Capitalized
                    "City": parts[8],    # ✅ Capitalized
                }])


                # Pass full DataFrame directly to pipeline (it handles encoding)
                prediction = lr_model.predict(input_df)[0]

                st.success(f"🏠 Predicted Price: ₹{prediction:,.2f} Lakhs")

        except Exception as e:
            st.error(f"Error predicting: {e}")
    else:
        st.warning("Please enter the property features.")

# -----------------------------
# 🔮 Prophet Forecast Section
# -----------------------------
st.header("📈 Prophet Forecast for Regional Trends")

if regions:
    selected_region = st.selectbox("Select a region:", regions)
    months = st.number_input("Months to forecast:", min_value=1, max_value=60, value=12)

    if st.button("Generate Forecast"):
        try:
            model = all_prophet_models[selected_region]

            if not isinstance(model, Prophet):
                st.error(f"❌ {selected_region} model is invalid. Please retrain it.")
            else:
                future = model.make_future_dataframe(periods=months, freq="M")
                forecast = model.predict(future)

                forecast_display = forecast[['ds', 'yhat']].tail(months)
                forecast_display['ds'] = pd.to_datetime(forecast_display['ds'])
                forecast_display['Month'] = forecast_display['ds'].dt.strftime('%Y-%m')
                forecast_display = forecast_display[['Month', 'yhat']].rename(
                    columns={"yhat": "Predicted Price"}
                )

                st.subheader(f"{selected_region} – Forecast for Next {months} Months")
                st.dataframe(forecast_display)

                st.line_chart(
                    forecast_display.set_index("Month"),
                    height=350,
                )
        except Exception as e:
            st.error(f"Error forecasting for {selected_region}: {e}")
else:
    st.warning("⚠ Prophet models could not be loaded. Forecasting unavailable.")