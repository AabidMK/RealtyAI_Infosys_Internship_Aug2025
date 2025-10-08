import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from prophet import Prophet
from prophet.plot import plot_plotly

# --- Custom Transformer for Price Prediction Pipeline ---
# This class MUST be defined for joblib to load the price prediction pipeline
class RealEstateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.top_neighborhoods = None

    def fit(self, X, y=None):
        # Learn the top neighborhoods from the training data
        temp_df = X.copy()
        temp_df['Location_Details'] = temp_df['Location'].str.split(',')
        temp_df['Neighborhood'] = temp_df['Location_Details'].apply(lambda loc: loc[0].strip() if loc and len(loc) > 0 else 'Unknown')
        neighborhood_counts = temp_df['Neighborhood'].value_counts()
        self.top_neighborhoods = neighborhood_counts.nlargest(30).index.tolist()
        return self

    def transform(self, X, y=None):
        df = X.copy()
        def get_bhk_from_title(title):
            if pd.isna(title): return 0
            match = re.search(r'(\d+)\s*BHK', str(title), re.IGNORECASE)
            return int(match.group(1)) if match else 0
        df['BHK'] = df['Property Title'].apply(get_bhk_from_title)
        df['Has_Balcony'] = df['Balcony'].map({'Yes': 1, 'Y': 1, 'No': 0, 'N': 0}).fillna(0)
        df['Total_Area'] = pd.to_numeric(df['Total_Area'], errors='coerce')
        df['Price_per_SQFT'] = pd.to_numeric(df['Price_per_SQFT'], errors='coerce')
        df['Baths'] = pd.to_numeric(df['Baths'], errors='coerce').fillna(1)
        if 'BHK' in df.columns:
            df['Total_Area'] = df['Total_Area'].fillna(df.groupby('BHK')['Total_Area'].transform('median'))

        df['Location_Details'] = df['Location'].str.split(',')
        df['Neighborhood'] = df['Location_Details'].apply(lambda loc: loc[0].strip() if loc and len(loc) > 0 else 'Unknown')
        df['City_Name'] = df['Location_Details'].apply(lambda loc: loc[-1].strip() if loc and len(loc) > 0 else 'Unknown')
        df.loc[~df['Neighborhood'].isin(self.top_neighborhoods), 'Neighborhood'] = 'Other'
        
        # --- Feature Engineering ---
        df['log_area'] = np.log1p(df['Total_Area'])
        df['Area_per_Room'] = df['Total_Area'] / np.maximum(df['BHK'], 1)
        df['log_area_per_room'] = np.log1p(df['Area_per_Room'])
        df['Bath_to_BHK_ratio'] = df['Baths'] / np.maximum(df['BHK'], 1)
        df['Total_Rooms'] = df['BHK'] + df['Baths']
        df['Area_Efficiency'] = df['Total_Area'] / np.maximum(df['Total_Rooms'], 1)
        df['Area_x_BHK'] = df['Total_Area'] * df['BHK']
        df['Area_x_Baths'] = df['Total_Area'] * df['Baths']
        df['log_Area_x_BHK'] = np.log1p(df['Area_x_BHK'])

        def classify_property_size(area):
            if area < 500: return 'Compact'
            elif area < 1000: return 'Medium'
            elif area < 2000: return 'Large'
            else: return 'Luxury'
        df['Property_Size_Category'] = df['Total_Area'].apply(classify_property_size)

        def classify_bhk_count(bhk):
            if bhk <= 1: return '1BHK'
            elif bhk <= 2: return '2BHK'
            elif bhk <= 3: return '3BHK'
            else: return '4+BHK'
        df['BHK_Category'] = df['BHK'].apply(classify_bhk_count)
        
        df['Price_per_Room'] = df['Price_per_SQFT'] * df['Area_per_Room']
        df['Is_Premium_Size'] = (df['Total_Area'] > 1600).astype(int)
        df['Has_Multiple_Baths'] = (df['Baths'] >= 2).astype(int)

        def create_luxury_rating(row):
            rating = 0
            if row['Total_Area'] > 1500: rating += 2
            elif row['Total_Area'] > 1000: rating += 1
            if row['BHK'] >= 4: rating += 2
            elif row['BHK'] >= 3: rating += 1
            if row['Baths'] >= 3: rating += 1
            if row['Has_Balcony']: rating += 1
            return rating
        df['Luxury_Score'] = df.apply(create_luxury_rating, axis=1)
        
        return df

# --- Model Loading Functions ---

@st.cache_resource
def load_prediction_pipeline():
    """Loads the property price prediction pipeline."""
    pipeline = joblib.load('property_price_pipeline.joblib')
    return pipeline

@st.cache_resource
def load_forecasting_models():
    """Loads the dictionary of Prophet forecasting models."""
    models = joblib.load('prophet_models.joblib')
    return models

# Load all models at the start
prediction_pipeline = load_prediction_pipeline()
forecasting_models = load_forecasting_models()

# --- Streamlit UI ---

st.set_page_config(page_title="Real Estate Analytics Platform", layout="wide")
st.title('🏠 Real Estate Analytics Platform')

# Create tabs
tab1, tab2 = st.tabs(["Forecasting", "Price Prediction"])

# Get the list of states from the loaded models for use in both tabs
state_list = [state for state in forecasting_models.keys() if state != 'UnitedStates']

# --- Forecasting Tab ---
with tab1:
    st.header("📈 Home Value Forecasting")
    st.markdown("Select a region and a time period to forecast future home values.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_state = st.selectbox("Select Region", state_list, index=state_list.index("California"))
        forecast_months = st.slider("Months to Forecast", min_value=1, max_value=60, value=24, help="Drag the slider to choose how many months to predict.")

    if st.button("Generate Forecast", key='forecast_button'):
        with st.spinner(f"Forecasting home values for {selected_state}..."):
            model = forecasting_models[selected_state]
            
            future_df = model.make_future_dataframe(periods=forecast_months, freq='M')
            forecast = model.predict(future_df)
            
            # Create a Plotly figure
            fig = plot_plotly(model, forecast)
            fig.update_layout(
                title=f"Forecast for {selected_state}",
                xaxis_title="Date",
                yaxis_title="Home Value",
                width=800,
                height=500
            )
            
            col2.plotly_chart(fig)
            
            st.write("### Forecast Data (Last 12 Months)")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12).rename(columns={
                'ds': 'Date', 'yhat': 'Predicted Value', 'yhat_lower': 'Lower Estimate', 'yhat_upper': 'Upper Estimate'
            }).set_index('Date'))


# --- Price Prediction Tab (UPDATED) ---
with tab2:
    st.header("💰 Individual Property Price Prediction")
    st.markdown("Enter the details of a property to get an estimated price in Lakhs.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Core Details")
        # UPDATED: Changed from a dropdown to a manual text input for location
        location = st.text_input('Location', placeholder="Enter the Location:")
        bhk = st.number_input('Number of Bedrooms (BHK)', min_value=1, max_value=10, value=2)
        
    with col2:
        st.subheader("Size & Area")
        total_area = st.number_input('Total Area (sq. ft.)', min_value=100, max_value=10000, value=1150)
        price_per_sqft = st.number_input('Price per (Sq. Ft.)', min_value=1000, max_value=50000, value=8260)
        
    with col3:
        st.subheader("Amenities")
        baths = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
        balcony = st.selectbox('Has Balcony?', ('Yes', 'No'))

    # Check if the user has entered a location before showing the predict button
    if location and st.button("Predict Price", key='predict_button', type="primary"):
        # Construct a dummy 'Property Title' because the pipeline's transformer needs it to extract BHK
        dummy_prop_title = f"{bhk} BHK Apartment"

        # Create a DataFrame from the inputs
        unseen_data = pd.DataFrame({
            'Property Title': [dummy_prop_title],
            'Price': [None],  # Not used for prediction
            'Location': [location], # Using the manually entered location
            'Total_Area': [total_area],
            'Price_per_SQFT': [price_per_sqft],
            'Baths': [baths],
            'Balcony': [balcony]
        })

        with st.spinner('Calculating the estimated price...'):
            predicted_price = prediction_pipeline.predict(unseen_data)
        
        st.metric(label="Predicted Property Value", value=f"₹ {predicted_price[0]:,.2f} Lakhs")