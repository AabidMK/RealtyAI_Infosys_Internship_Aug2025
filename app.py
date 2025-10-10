import streamlit as st
import joblib
import pandas as pd
import os

# Path to Downloads folder
DOWNLOADS_PATH = os.path.join(os.path.expanduser("~"), "Downloads")
# DOWNLOADS_PATH = "models"

# Patch for sklearn version mismatch
import sklearn.compose._column_transformer as ct
if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    ct._RemainderColsList = _RemainderColsList

# Page Config
st.set_page_config(page_title="Real Estate Price Predictor", page_icon="🏠", layout="wide")

# Sidebar Navigation
st.sidebar.title("Go to")
page = st.sidebar.selectbox("Navigation", ["🏡 Home", "💰 Price Prediction", "📈 Forecast Future Prices"])

# Styling
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #e0f7fa, #fce4ec);
            font-family: 'Poppins', sans-serif;
        }
        .main-content {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            max-width: 900px;
            margin: 2rem auto;
        }
        .title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            color: #2E86C1;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #5D6D7E;
            margin-bottom: 2rem;
        }
        .stButton>button {
            background-color: #2E86C1;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            border: none;
        }
        .stButton>button:hover {
            background-color: #2874A6;
        }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    bagging_model = joblib.load(os.path.join(DOWNLOADS_PATH,"bagging_regressor_model.pkl"))
    pipeline = joblib.load(os.path.join(DOWNLOADS_PATH,"real_estate_pipeline.pkl"))
    prophet_models = joblib.load(os.path.join(DOWNLOADS_PATH,"all_prophet_models.pkl"))
    return bagging_model, pipeline, prophet_models

bagging_model, pipeline, prophet_models = load_models()

# Main content
st.markdown('<div class="main-content">', unsafe_allow_html=True)

st.markdown('<h1 class="title">🏠 Real Estate Prediction Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict property prices and visualize future trends using ML models.</p>', unsafe_allow_html=True)

# =============== Home Page =============== #
if page == "🏡 Home":
    st.markdown("### Welcome!")
    st.write("""
    Use this platform to:
    - 🔮 Predict current property prices  
    - 📊 Forecast future real estate trends  
    Select an option from the sidebar to get started!
    """)

# =============== Price Prediction Page =============== #
elif page == "💰 Price Prediction":
    st.header("🧩 Enter Property Details")
    col1, col2 = st.columns(2)

    with col1:
        sqft = st.number_input("📏 Square Feet", min_value=200, max_value=10000, value=1200)
        bedrooms = st.number_input("🛏 Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("🚿 Bathrooms", min_value=1, max_value=10, value=2)

    with col2:
        location = st.text_input("📍 Location", "Hyderabad")
        year_built = st.number_input("🏗 Year Built", min_value=1900, max_value=2025, value=2015)

    if st.button("🔮 Predict Price"):
        input_data = pd.DataFrame([{
            "sqft": sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "location": location,
            "year_built": year_built,
            "Balcony_Encoded": 0,
            "Description": "",
            "Name": "",
            "City": location,
            "Baths": bathrooms,
            "BHK": bedrooms,
            "Total_Area": sqft,
            "Price_per_SQFT": 0
        }])

        try:
            prediction = pipeline.predict(input_data)[0]
            st.success(f"🏡 **Estimated Property Price:** ₹ {prediction:,.2f}")
        except Exception as e:
            st.error(f"❌ Error in prediction: {e}")

# =============== Forecast Page =============== #
elif page == "📈 Forecast Future Prices":
    st.header("📊 Forecast Future Prices")

    # List of available states
    states = [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
        'Delaware', 'DistrictofColumbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois',
        'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts',
        'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
        'NewHampshire', 'NewJersey', 'NewMexico', 'NewYork', 'NorthCarolina', 'NorthDakota',
        'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'RhodeIsland', 'SouthCarolina', 'SouthDakota',
        'Tennessee', 'Texas', 'UnitedStates', 'Utah', 'Vermont', 'Virginia', 'Washington',
        'WestVirginia', 'Wisconsin', 'Wyoming'
    ]

    # Dropdown for state selection
    selected_location = st.selectbox("🏙 Select State for Forecast", options=states, index=4)

    # Forecast Button
    if st.button("📈 Show Forecast"):
        if selected_location in prophet_models:
            model = prophet_models[selected_location]
            future = model.make_future_dataframe(periods=12, freq="M")
            forecast = model.predict(future)
            st.line_chart(forecast.set_index("ds")[["yhat"]])
            st.caption(f"🕒 Predicted future trend for **{selected_location}** (next 12 months).")
        else:
            st.warning(f"⚠️ No Prophet model found for **{selected_location}**.")