import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# --- CONSTANTS ---
# Using a fixed exchange rate for demonstration
USD_TO_INR_RATE = 80.0
# -----------------

# -----------------------------
# Page Config 
# -----------------------------
st.set_page_config(
    page_title=" Home Value Analytics",
    layout="wide",
    page_icon="🏡"
)

# -----------------------------
# Custom CSS for Streamlit Cards and Styling (UPDATED)
# -----------------------------
st.markdown("""
    <style>
    /* Global Styles */
    body {background-color: #f8f9fa;}
    .main-title {text-align: center; color: #2c3e50; font-size: 38px; font-weight: 700;}
    .sub-title {text-align: center; color: gray; font-size: 16px;}
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        height: 3em; width: 100%;
    }
    .stButton>button:hover {
        background-color: #2980b9; color: white;
    }
    
    /* Card Styles (Metric boxes for Tab 1) */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 100%;
    }

    /* Specific style for Tab 1 prediction cards (Light Green) */
    #tab-container-1 div[data-testid="stMetric"] {
        background-color: #e6ffe6; /* Light green */
        border: 1px solid #c8e6c9; 
    }
    
    /* Center the metric value and label */
    div[data-testid="stMetric"] > div:nth-child(1) {
        text-align: center;
    }
    div[data-testid="stMetric"] label {
        color: #555555 !important;
        font-weight: 500;
        font-size: 14px;
    }
    /* Value color and size */
    div[data-testid="stMetricValue"] {
        color: #2c3e50 !important;
        font-size: 22px !important; /* Smaller size for cards */
        font-weight: 700 !important;
    }
    /* Secondary text (used for USD or Rupee comparison) */
    div[data-testid="stMetricDelta"] {
        font-size: 14px;
        color: #2c3e50 !important;
        margin-top: 5px;
    }
    
    /* === New Dashboard Styles for Tab 2 === */
    .page-title-tab2 {
        text-align: center; color: #2c3e50; font-size: 42px; font-weight: bold; margin-bottom: 5px;
    }
    .page-subtitle-tab2 {
        text-align: center; color: gray; font-size: 18px; margin-bottom: 30px;
    }

    .section {
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
    }
    .settings-section {
        background-color: #ecf0f1;
    }
    .plot-section {
        background-color: #f9f9f9;
    }
    .table-section {
        background-color: #ecf0f1;
    }
    .kpi-section {
        background-color: #f9f9f9;
    }

    /* KPI Cards for Tab 2 */
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .card:hover {
        transform: scale(1.03);
    }
    .card h3 {
        color: #2c3e50;
        font-size: 18px;
        margin-bottom: 5px;
    }
    .card h2 {
        color: #3498db;
        margin-top: 5px;
        font-size: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Title 
# -----------------------------
st.markdown("<h1 class='main-title'>🏡 Realty AI-Smart Real Estate Insight Platform</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict or forecast housing prices using advanced ML models.</p>", unsafe_allow_html=True)

# -----------------------------
# Define City-Locality Mapping 
# -----------------------------
CITY_LOCALITY_MAP = {
    'Bangalore': ['Bannerughatta', 'Bommanahalli', 'Dodsworth Layout', 'Horamavu Agara', 
                  'Kereguddadahalli', 'Kithaganur Colony', 'Lingadheeranahalli', 
                  'Mattanahalli', 'Meenakshi Layout', 'Nelamangala', 'Sarjapur', 
                  'Srinivasa Nagar', 'Veer Sandra', 'Yelahanka', 'Other'],
    'Chennai': ['Avadi', 'Veppampattu', 'Other'],
    'Hyderabad': ['Terracon Doctors Enclave', 'Other'],
    'Kolkata': ['Other'], 
    'Mumbai': ['Naigaon East', 'Other'],
    'New Delhi': ['Devli', 'Sector 12 Dwarka', 'Tukaram Nagar', 'Uttam nagar west', 'Other'],
    'Pune': ['Alandi', 'Ambegaon', 'Chakan', 'Hindustan Antibiotics Colony', 
             'Hinjewadi Phase 2', 'Narhe', 'Wagholi', 'Other'],
    'Thane': ['Other'], 
    'Other': ['Apartments', 'Other'] 
}
ALL_CITIES = sorted(CITY_LOCALITY_MAP.keys())


# -----------------------------
# Currency Formatting Functions 
# -----------------------------
def format_indian_currency(amount: float) -> str:
    """Converts a raw rupee amount into Lakhs or Crores format."""
    CRORE = 10000000
    LAKH = 100000
    
    if abs(amount) >= CRORE:
        value = amount / CRORE
        return f"₹{value:,.2f} Cr"
    elif abs(amount) >= LAKH:
        value = amount / LAKH
        return f"₹{value:,.2f} L"
    else:
        return f"₹{amount:,.0f}"

def format_usd_currency(amount_inr: float) -> str:
    """Converts a raw rupee amount to USD and formats it."""
    amount_usd = amount_inr / USD_TO_INR_RATE
    
    # Use M/Bn for Millions/Billions for USD
    MILLION = 1000000
    BILLION = 1000000000
    
    if abs(amount_usd) >= BILLION:
        value = amount_usd / BILLION
        return f"${value:,.2f} Bn"
    elif abs(amount_usd) >= MILLION:
        value = amount_usd / MILLION
        return f"${value:,.2f} M"
    else:
        return f"${amount_usd:,.0f}"

def format_usd_per_sqft(amount_inr: float) -> str:
    """Formats price per sqft in USD."""
    amount_usd = amount_inr / USD_TO_INR_RATE
    return f"${amount_usd:,.0f}/sqft"


# -----------------------------
# Load Models 
# -----------------------------
@st.cache_resource
def load_price_model():
    return joblib.load("real_estate_pipeline_v20250924_122040.joblib")

@st.cache_resource
def load_forecast_models():
    # Model variable name remains 'forecast_models' for consistency with main code structure
    return joblib.load("all_states_prophet_models.pkl")

try:
    price_model = load_price_model()
    # Using 'forecast_models' as the variable name as per original code
    forecast_models = load_forecast_models()

    # -----------------------------
    # Tabs
    # -----------------------------
    tab1, tab2 = st.tabs(["🏠 House Price Prediction", "📈 Price Forecasting"])

    # ===========================================================================================
    # 1️⃣ HOUSE PRICE PREDICTION
    # ===========================================================================================
    with tab1:
        # Container with ID to apply green card CSS specifically
        st.markdown("<div id='tab-container-1'>", unsafe_allow_html=True) 
        
        st.markdown("### 🏗️ Enter Property Details")

        col1, col2, col3 = st.columns(3)
        with col1:
            city = st.selectbox("City", ALL_CITIES)
            locality_options = CITY_LOCALITY_MAP.get(city, ['Other'])
            locality = st.selectbox("Locality", locality_options)
            
        with col2:
            bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=3)
            baths = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
            
        with col3:
            total_area = st.number_input("Total Area (sqft)", min_value=200.0, max_value=10000.0, value=1200.0)
            balcony_option = st.selectbox("Has Balcony?", ['Yes', 'No', 'Not Specified'], index=0)
            
        placeholder = st.empty()

        # Predict button
        if st.button("🚀 Predict House Price"):
            
            # 1. Prepare Input Data
            input_data = pd.DataFrame({
                'Total_Area': [total_area], 'BHK': [bhk], 'Bath': [baths],
                'Balcony': [balcony_option], 'City': [city],
                'Locality': [locality], 'Price_per_SQFT': [0] 
            })
            
            try:
                # 2. Prediction (raw_predicted_price is in Rupees)
                predicted_price_in_lakhs = price_model.predict(input_data)[0]
                raw_predicted_price = predicted_price_in_lakhs * 100000 
                
                # 3. Output Calculations (IN & USD)
                formatted_inr = format_indian_currency(raw_predicted_price)
                formatted_usd = format_usd_currency(raw_predicted_price)
                
                if total_area > 0:
                    price_per_sqft_inr = raw_predicted_price / total_area
                    formatted_inr_sqft = f"₹{price_per_sqft_inr:,.0f}/sqft"
                    formatted_usd_sqft = format_usd_per_sqft(price_per_sqft_inr)
                else:
                    formatted_inr_sqft = "N/A"
                    formatted_usd_sqft = "N/A"

                # 4. Display as CARDS 
                with placeholder.container():
                    st.markdown("---")
                    st.markdown("### Prediction Results")
                    
                    # Use weighted columns for smaller, left-aligned cards
                    col_pred_1, col_pred_2, col_spacer = st.columns([1, 1, 3]) 
                    
                    with col_pred_1:
                        st.metric(
                            label="🏡 Estimated Total Price (INR)", 
                            value=formatted_inr,
                            delta=f"Equivalent: {formatted_usd} (USD)", 
                            delta_color="off"
                        )
                    
                    with col_pred_2:
                        st.metric(
                            label="💰 Estimated Price per Sqft (INR)", 
                            value=formatted_inr_sqft,
                            delta=f"Equivalent: {formatted_usd_sqft} (USD)", 
                            delta_color="off"
                        )
                
            except Exception as e:
                placeholder.empty()
                st.error(f"⚠️ **Prediction error:** The model prediction failed. Error details: `{e}`")

        st.markdown("</div>", unsafe_allow_html=True)
        
    # ===========================================================================================
    # 2️⃣ PRICE FORECASTING (UPDATED AS REQUESTED)
    # ===========================================================================================
    with tab2:
        
        # -----------------------------
        # Header
        # -----------------------------
        st.markdown("<div class='page-title-tab2'>🏠 Home Value Forecasting Dashboard</div>", unsafe_allow_html=True)
        st.markdown("<div class='page-subtitle-tab2'>Predict housing price trends across all states with Meta Prophet</div>", unsafe_allow_html=True)

        # -----------------------------
        # Forecast Settings
        # -----------------------------
        st.markdown('<div class="section settings-section">', unsafe_allow_html=True)
        st.markdown("<h2 style='color:#2c3e50;'>⚙️ Forecast Settings</h2>", unsafe_allow_html=True)

        # Using the loaded 'forecast_models' variable (which is equivalent to 'all_state_models' in the requested snippet)
        state = st.selectbox("Select State", list(forecast_models.keys())) 
        col1, col2, col3 = st.columns(3)
        with col1:
            year = st.number_input("Starting Year", min_value=2000, max_value=2030, value=datetime.now().year, step=1)
        with col2:
            month = st.selectbox("Starting Month", list(range(1, 13)), index=datetime.now().month-1)
        with col3:
            periods = st.slider("Forecast Horizon (months)", 1, 36, 12)
        st.markdown('</div>', unsafe_allow_html=True)

        # -----------------------------
        # Generate Forecast
        # -----------------------------
        if st.button("🚀 Generate Forecast", use_container_width=True):
            model = forecast_models[state]

            # Future dataframe starting from user-selected date
            start_date = datetime(year, month, 1)
            future = pd.DataFrame({'ds': pd.date_range(start=start_date, periods=periods, freq='ME')})
            forecast = model.predict(future)

            # --- Forecast Plot (Model output 'yhat' assumed to be in USD as per requested snippet) ---
            st.markdown('<div class="section plot-section">', unsafe_allow_html=True)
            st.markdown("<h2 style='color:#2c3e50;'>📈 Forecast Plot</h2>", unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode='lines', name='Predicted Value', line=dict(color='#3498db', width=3)))
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
                y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
                fill='toself',
                fillcolor='rgba(52, 152, 219, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='Confidence Interval'
            ))
            fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Predicted Home Value ($)",
                              title=f"{state} Home Value Forecast", title_x=0.5, height=600,
                              legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
            st.plotly_chart(fig, config={'responsive': True})
            st.markdown('</div>', unsafe_allow_html=True)

            # -----------------------------
            # Forecast Table
            # -----------------------------
            st.markdown('<div class="section table-section">', unsafe_allow_html=True)
            st.markdown("<h2 style='color:#2c3e50;'>📋 Forecasted Values</h2>", unsafe_allow_html=True)

            latest_forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            latest_forecast.loc[:, "Date"] = pd.to_datetime(latest_forecast["ds"]).dt.strftime("%b %Y")
            latest_forecast = latest_forecast.drop(columns="ds")
            latest_forecast.columns = ["Predicted ($)", "Lower Bound ($)", "Upper Bound ($)", "Date"]
            latest_forecast = latest_forecast[["Date", "Predicted ($)", "Lower Bound ($)", "Upper Bound ($)"]]
            latest_forecast.index = range(1, len(latest_forecast) + 1)
            latest_forecast.index.name = "S.No"

            st.dataframe(latest_forecast.style.format({
                "Predicted ($)": "{:,.2f}",
                "Lower Bound ($)": "{:,.2f}",
                "Upper Bound ($)": "{:,.2f}"
            }), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # -----------------------------
            # KPI Cards
            # -----------------------------
            st.markdown('<div class="section kpi-section">', unsafe_allow_html=True)
            st.markdown("<h2 style='color:#2c3e50;'>🧮 Key Metrics</h2>", unsafe_allow_html=True)
            col_kpi_1, col_kpi_2, col_kpi_3 = st.columns(3)
            latest_value = latest_forecast["Predicted ($)"].iloc[-1]
            growth = ((latest_value - latest_forecast["Predicted ($)"].iloc[0]) / latest_forecast["Predicted ($)"].iloc[0]) * 100

            col_kpi_1.markdown(f"<div class='card'><h3>Latest Predicted Value</h3><h2>${latest_value:,.2f}</h2></div>", unsafe_allow_html=True)
            col_kpi_2.markdown(f"<div class='card'><h3>Growth over Period</h3><h2>{growth:.2f}%</h2></div>", unsafe_allow_html=True)
            col_kpi_3.markdown(f"<div class='card'><h3>Forecast Range</h3><h2>${latest_forecast['Lower Bound ($)'].min():,.0f} - ${latest_forecast['Upper Bound ($)'].max():,.0f}</h2></div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info("👈 Select forecast parameters and click **'Generate Forecast'** to view predictions.")


except Exception as e:
    st.error(f"❌ **Critical Error:** Could not load one or both models. Please ensure the model files are accessible. Error details: `{e}`")
    st.stop()


# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>
© 2025 Home Value Analytics | Developed by Vishnu Vardhan Reddy
</p>
""", unsafe_allow_html=True)