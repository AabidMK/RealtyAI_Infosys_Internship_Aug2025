import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from forecasting_utils import HousePriceForecaster, load_forecasting_data, get_region_statistics
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="House Price Prediction & Forecasting",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    """Load the trained model and location encoding"""
    try:
        pipeline, location_target = joblib.load("pipeline.pkl")
        return pipeline, location_target
    except FileNotFoundError:
        st.error("Model file 'pipeline.pkl' not found. Please run train.py first to train the model.")
        return None, None

@st.cache_data
def load_data():
    """Load the real estate data"""
    try:
        df = pd.read_csv("Data/Real Estate Data V21.csv")
        return df
    except FileNotFoundError:
        st.error("Data file 'Real Estate Data V21.csv' not found.")
        return None

@st.cache_resource
def load_forecaster():
    """Load the Prophet forecaster"""
    try:
        forecaster = HousePriceForecaster("prophet_models_all_regions.joblib")
        return forecaster
    except FileNotFoundError:
        st.warning("Prophet models not found. Please ensure 'prophet_models_all_regions.joblib' exists.")
        return None

def preprocess_input_data(data_dict):
    """Preprocess input data for prediction"""
    df = pd.DataFrame([data_dict])
    
    # Extract BHK from Property Title
    df['BHK'] = df['Property Title'].apply(
        lambda t: int(re.search(r'(\d+)\s*BHK', str(t), re.IGNORECASE).group(1)) 
        if re.search(r'(\d+)\s*BHK', str(t), re.IGNORECASE) else 0
    )
    
    # Convert Balcony to numeric
    df['Balcony'] = df['Balcony'].map({'Yes': 1, 'No': 0}).fillna(0)
    
    # Extract Property Type
    def prop_type(t):
        t = str(t).lower()
        if "independent house" in t: 
            return "Independent House"
        elif "flat" in t: 
            return "Flat"
        elif "villa" in t: 
            return "Villa"
        else: 
            return "Other"
    
    df['Property_Type'] = df['Property Title'].apply(prop_type)
    
    # Extract City
    df['City'] = df['Location'].apply(lambda x: str(x).split(",")[-1].strip())
    
    return df

def preprocess_input_data_with_bhk(data_dict, extracted_bhk):
    """Preprocess input data for prediction with extracted BHK"""
    df = pd.DataFrame([data_dict])
    
    # Use extracted BHK if available, otherwise extract from title
    if extracted_bhk is not None:
        df['BHK'] = extracted_bhk
    else:
        df['BHK'] = df['Property Title'].apply(
            lambda t: int(re.search(r'(\d+)\s*BHK', str(t), re.IGNORECASE).group(1)) 
            if re.search(r'(\d+)\s*BHK', str(t), re.IGNORECASE) else 0
        )
    
    # Convert Balcony to numeric
    df['Balcony'] = df['Balcony'].map({'Yes': 1, 'No': 0}).fillna(0)
    
    # Extract Property Type
    def prop_type(t):
        t = str(t).lower()
        if "independent house" in t: 
            return "Independent House"
        elif "flat" in t: 
            return "Flat"
        elif "villa" in t: 
            return "Villa"
        else: 
            return "Other"
    
    df['Property_Type'] = df['Property Title'].apply(prop_type)
    
    # Extract City
    df['City'] = df['Location'].apply(lambda x: str(x).split(",")[-1].strip())
    
    return df

def create_features(df, location_target):
    """Create engineered features for prediction"""
    # Location encoding
    df['Location_Encoded'] = df['Location'].map(location_target)
    df['Location_Encoded'] = df['Location_Encoded'].fillna(np.mean(list(location_target.values())))
    
    # Log transformation
    df['Log_Total_Area'] = np.log(df['Total_Area'] + 1)
    
    # Interaction features
    df['BHK_x_Total_Area'] = df['BHK'] * df['Total_Area']
    df['BHK_x_Baths'] = df['BHK'] * df['Baths']
    df['Balcony_x_BHK'] = df['Balcony'] * df['BHK']
    df['Log_Total_Area_x_BHK'] = df['Log_Total_Area'] * df['BHK']
    df['Log_Total_Area_squared'] = df['Log_Total_Area'] ** 2
    
    return df

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 House Price Prediction & Forecasting</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["🏠 Price Prediction",  "🔮 Price Forecasting"])
    
    # Load data and model
    pipeline, location_target = load_model()
    df = load_data()
    forecaster = load_forecaster()
    df_ts = load_forecasting_data()
    
    if page == "🏠 Price Prediction":
        price_prediction_page(pipeline, location_target, df)
    elif page == "🔮 Price Forecasting":
        forecasting_page(forecaster, df_ts)

def price_prediction_page(pipeline, location_target, df):
    """House price prediction page"""
    st.header("🏠 House Price Prediction")
    
    if pipeline is None or location_target is None:
        st.error("Model not available. Please ensure pipeline.pkl exists.")
        return
    
    # Create two columns for input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Property Details")
        
        # Property Title
        property_title = st.text_input(
            "Property Title",
            placeholder="e.g., 2 BHK Flat in Andheri",
            help="Enter the property title including BHK information"
        )
        
        # Location
        location = st.text_input(
            "Location",
            placeholder="e.g., Andheri, Mumbai",
            help="Enter location in format: Area, City"
        )
        
        # Total Area
        total_area = st.number_input(
            "Total Area (sq ft)",
            min_value=100,
            max_value=10000,
            value=1200,
            step=50
        )
        
        # Number of Bathrooms
        baths = st.number_input(
            "Number of Bathrooms",
            min_value=1,
            max_value=10,
            value=2
        )
    
    with col2:
        st.subheader("Additional Features")
        
        # Balcony
        balcony = st.selectbox(
            "Balcony",
            ["Yes", "No"],
            help="Does the property have a balcony?"
        )
        
        # City selection (for reference)
        if df is not None:
            cities = df['Location'].str.split(',').str[-1].str.strip().unique()
            selected_city = st.selectbox(
                "City (for reference)",
                sorted(cities),
                help="Select the city for better location encoding"
            )
        
        # Property Type and BHK (auto-detected from title)
        if property_title:
            # Extract Property Type
            prop_type = "Flat" if "flat" in property_title.lower() else \
                       "Villa" if "villa" in property_title.lower() else \
                       "Independent House" if "independent house" in property_title.lower() else "Other"
            
            # Extract BHK from Property Title
            bhk_match = re.search(r'(\d+)\s*BHK', property_title, re.IGNORECASE)
            extracted_bhk = int(bhk_match.group(1)) if bhk_match else None
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Property Type**: {prop_type}")
            with col2:
                if extracted_bhk:
                    st.info(f"**BHK**: {extracted_bhk}")
                else:
                    st.warning("**BHK**: Not found in title")
    
    # Prediction button
    if st.button("🔮 Predict Price", type="primary"):
        if not property_title or not location:
            st.error("Please fill in all required fields (Property Title and Location)")
            return
        
        # Extract BHK from Property Title for validation
        bhk_match = re.search(r'(\d+)\s*BHK', property_title, re.IGNORECASE)
        extracted_bhk = int(bhk_match.group(1)) if bhk_match else None
        
        if extracted_bhk is None:
            st.warning("⚠️ BHK not found in Property Title. Please include BHK in the title (e.g., '2 BHK Flat in Andheri')")
            st.info("The prediction will proceed with BHK=0, which may affect accuracy.")
        
        try:
            # Prepare input data
            input_data = {
                'Property Title': property_title,
                'Location': location,
                'Total_Area': total_area,
                'Baths': baths,
                'Balcony': balcony
            }
            
            # Preprocess data with extracted BHK
            df_input = preprocess_input_data_with_bhk(input_data, extracted_bhk)
            df_features = create_features(df_input, location_target)
            
            # Feature columns
            feature_cols = [
                'Log_Total_Area', 'Baths', 'Balcony', 'BHK', 'Location_Encoded',
                'BHK_x_Total_Area', 'BHK_x_Baths', 'Balcony_x_BHK', 
                'Log_Total_Area_x_BHK', 'Log_Total_Area_squared',
                'City', 'Property_Type'
            ]
            
            X_pred = df_features[feature_cols]
            
            # Make prediction
            log_price_pred = pipeline.predict(X_pred)
            predicted_price = np.exp(log_price_pred) - 1
            
            # Display results
            st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Predicted Price")
            st.markdown(f"# ₹{predicted_price[0]:.2f} Lakhs")
            st.markdown(f"### (₹{predicted_price[0]*100000:,.0f})")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional insights
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Price per sq ft", f"₹{predicted_price[0]*100000/total_area:.0f}")
            with col2:
                st.metric("BHK", f"{df_features['BHK'].iloc[0]}")
            with col3:
                st.metric("Bathrooms", f"{baths}")
            with col4:
                prop_type = df_features['Property_Type'].iloc[0]
                st.metric("Property Type", prop_type)
            
            # Show extracted information
            st.subheader("📋 Extracted Information")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Property Type**: {prop_type}")
            with col2:
                if extracted_bhk:
                    st.success(f"**BHK**: {extracted_bhk} (extracted from title)")
                else:
                    st.warning("**BHK**: Not found in title")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    



def forecasting_page(forecaster, df_ts):
    """Advanced time series forecasting page using Prophet models"""
 
    
    if forecaster is None:
        st.error("Forecasting models not available. Please ensure Prophet models are trained and saved.")
        st.info("To train the models, run the forecasting notebook and save the models using joblib.")
        return
    
    # Get available regions
    available_regions = forecaster.get_available_regions()
    
    if not available_regions:
        st.warning("No forecasting models available.")
        return
    
   
    
    # Sidebar controls
    st.sidebar.subheader("Forecasting Controls")
    
    # Region selection
    selected_regions = st.sidebar.multiselect(
        "Select Regions to Forecast",
        available_regions,
        default=available_regions[:3] if len(available_regions) >= 3 else available_regions
    )
    
    # Forecasting parameters
    forecast_days = st.sidebar.slider(
        "Forecast Period (Days)",
        min_value=30,
        max_value=1095,  # 3 years
        value=365,
        step=30
    )
    
    include_history = st.sidebar.checkbox("Include Historical Data", value=True)
    
    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Single Region Forecast", "Multi-Region Comparison", "Region Statistics"]
    )
    
    if not selected_regions:
        st.warning("Please select at least one region to proceed.")
        return
    
    # Main content based on analysis type
    if analysis_type == "Single Region Forecast":
        single_region_forecast(forecaster, selected_regions[0], forecast_days, include_history)
    
    elif analysis_type == "Multi-Region Comparison":
        multi_region_comparison(forecaster, selected_regions, forecast_days)
    
    elif analysis_type == "Region Statistics":
        region_statistics_page(forecaster, selected_regions, df_ts)

def single_region_forecast(forecaster, region, forecast_days, include_history):
    """Display single region forecast with detailed analysis"""
    st.subheader(f"📊 Forecast for {region}")
    
    try:
        # Generate forecast
        with st.spinner(f"Generating forecast for {region}..."):
            forecast_data = forecaster.forecast_region(region, forecast_days, include_history)
        
        # Display forecast plot
        fig_forecast = forecaster.create_forecast_plot(forecast_data, f"({forecast_days} days)")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Display components plot
        with st.expander("📈 Trend and Seasonality Components"):
            fig_components = forecaster.create_components_plot(forecast_data)
            st.plotly_chart(fig_components, use_container_width=True)
        
        # Forecast summary
        summary = forecaster.get_forecast_summary(forecast_data)
        
        if summary:
            st.subheader("📋 Forecast Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current ZHVI",
                    f"${summary['current_value']:,.0f}",
                    help="Current home value index"
                )
            
            with col2:
                st.metric(
                    "Forecast End Value",
                    f"${summary['forecast_end_value']:,.0f}",
                    help="Predicted value at end of forecast period"
                )
            
            with col3:
                change_color = "normal" if summary['percent_change'] >= 0 else "inverse"
                st.metric(
                    "Expected Change",
                    f"{summary['percent_change']:.1f}%",
                    delta=f"${summary['total_change']:,.0f}",
                    delta_color=change_color
                )
            
            with col4:
                st.metric(
                    "Forecast Period",
                    f"{summary['forecast_periods']} days",
                    help="Number of days forecasted"
                )
            
            # Additional insights
            st.subheader("💡 Key Insights")
            
            if summary['percent_change'] > 5:
                st.success(f"📈 **Positive Trend**: {region} shows strong growth potential with {summary['percent_change']:.1f}% expected increase.")
            elif summary['percent_change'] < -5:
                st.warning(f"📉 **Declining Trend**: {region} shows declining values with {summary['percent_change']:.1f}% expected decrease.")
            else:
                st.info(f"📊 **Stable Market**: {region} shows relatively stable values with {summary['percent_change']:.1f}% expected change.")
            
            # Forecast range
            st.info(f"**Forecast Range**: ${summary['min_forecast']:,.0f} - ${summary['max_forecast']:,.0f}")
    
    except Exception as e:
        st.error(f"Error generating forecast for {region}: {str(e)}")

def multi_region_comparison(forecaster, regions, forecast_days):
    """Display multi-region forecast comparison"""
    st.subheader("🔄 Multi-Region Forecast Comparison")
    
    if len(regions) < 2:
        st.warning("Please select at least 2 regions for comparison.")
        return
    
    try:
        with st.spinner("Generating multi-region forecast comparison..."):
            fig_comparison = forecaster.compare_regions_forecast(regions, forecast_days)
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Summary table
        st.subheader("📊 Forecast Summary by Region")
        
        summary_data = []
        for region in regions:
            try:
                forecast_data = forecaster.forecast_region(region, forecast_days, include_history=False)
                summary = forecaster.get_forecast_summary(forecast_data)
                
                if summary:
                    summary_data.append({
                        'Region': region,
                        'Current ZHVI': f"${summary['current_value']:,.0f}",
                        'Forecast End': f"${summary['forecast_end_value']:,.0f}",
                        'Change %': f"{summary['percent_change']:.1f}%",
                        'Change $': f"${summary['total_change']:,.0f}"
                    })
            except Exception as e:
                st.warning(f"Could not generate summary for {region}: {str(e)}")
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
            
            # Download option
            csv = df_summary.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast Summary",
                data=csv,
                file_name=f"forecast_summary_{forecast_days}days.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error generating comparison: {str(e)}")

def region_statistics_page(forecaster, regions, df_ts):
    """Display detailed statistics for selected regions"""
    st.subheader("📈 Region Statistics & Analysis")
    
    if df_ts is None:
        st.warning("Historical data not available for detailed statistics.")
        return
    
    # Region selection for detailed stats
    selected_region = st.selectbox("Select Region for Detailed Analysis", regions)
    
    if selected_region:
        # Get region statistics
        stats = get_region_statistics(df_ts, selected_region)
        
        if stats:
            st.subheader(f"📊 Statistics for {selected_region}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", f"{stats['total_records']:,}")
            
            with col2:
                st.metric("Latest ZHVI", f"${stats['latest_zhvi']:,.0f}" if stats['latest_zhvi'] else "N/A")
            
            with col3:
                st.metric("Average ZHVI", f"${stats['zhvi_mean']:,.0f}")
            
            with col4:
                st.metric("ZHVI Range", f"${stats['zhvi_min']:,.0f} - ${stats['zhvi_max']:,.0f}")
            
            st.info(f"**Data Period**: {stats['date_range']}")
            
            # Historical trend plot
            region_data = df_ts[df_ts['RegionName'] == selected_region]
            
            if not region_data.empty and 'ZHVI_AllHomes' in region_data.columns:
                fig_historical = px.line(
                    region_data,
                    x='Date',
                    y='ZHVI_AllHomes',
                    title=f'Historical ZHVI Trend for {selected_region}',
                    labels={'ZHVI_AllHomes': 'ZHVI (Home Value Index)'}
                )
                st.plotly_chart(fig_historical, use_container_width=True)
        
        # Quick comparison table
        if len(regions) > 1:
            st.subheader("🔄 Quick Region Comparison")
            
            comparison_data = []
            for region in regions:
                stats = get_region_statistics(df_ts, region)
                if stats and stats['latest_zhvi']:
                    comparison_data.append({
                        'Region': region,
                        'Latest ZHVI': f"${stats['latest_zhvi']:,.0f}",
                        'Average ZHVI': f"${stats['zhvi_mean']:,.0f}",
                        'Records': stats['total_records']
                    })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)

if __name__ == "__main__":
    main()
