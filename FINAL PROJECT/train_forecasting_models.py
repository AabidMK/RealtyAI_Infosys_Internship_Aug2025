"""
Script to train Prophet models for house price forecasting
Based on the FORECASTING notebook implementation
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the time series data"""
    print("Loading time series data...")
    df = pd.read_csv("Data/State_time_series.csv")
    
    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).reset_index(drop=True)
    
    # Create multivariate dataframe
    data = df[['Date', 'RegionName', 'ZHVI_AllHomes']].copy()
    data = data.sort_values(['RegionName', 'Date']).reset_index(drop=True)
    
    # Fill missing values
    data['ZHVI_AllHomes'] = (
        data.groupby('RegionName')['ZHVI_AllHomes']
            .transform(lambda x: x.interpolate(method='linear').ffill().bfill())
    )
    
    # Remove outliers using rolling median
    def remove_outliers_rolling_median(series, window=5, mad_multiplier=3.0):
        """Replace outliers in a series using rolling median + MAD."""
        rolling_med = series.rolling(window=window, center=True, min_periods=1).median()
        abs_dev = (series - rolling_med).abs()
        rolling_mad = abs_dev.rolling(window=window, center=True, min_periods=1).median()

        # Avoid zero MAD (flat regions)
        eps = series.std() * 0.1 if series.std() > 0 else 1e-8
        rolling_mad = rolling_mad.replace(0, eps)

        threshold = rolling_mad * mad_multiplier
        outliers = abs_dev > threshold

        # Replace outliers with rolling median
        series_cleaned = series.copy()
        series_cleaned[outliers] = rolling_med[outliers]
        return series_cleaned

    # Apply outlier removal per region
    data['ZHVI_AllHomes'] = data.groupby('RegionName')['ZHVI_AllHomes'].transform(remove_outliers_rolling_median)
    
    print(f"Data loaded and preprocessed. Shape: {data.shape}")
    print(f"Number of unique regions: {data['RegionName'].nunique()}")
    
    return data

def train_prophet_models(data):
    """Train Prophet models for each region"""
    print("Training Prophet models...")
    
    prophet_models = {}
    regions_with_insufficient_data = []
    
    for region in data['RegionName'].unique():
        region_df = data[data['RegionName'] == region][['Date', 'ZHVI_AllHomes']].copy()
        region_df = region_df.rename(columns={'Date':'ds', 'ZHVI_AllHomes':'y'})
        region_df = region_df.sort_values('ds').reset_index(drop=True)
        
        # Skip regions with less than 2 non-null data points
        if region_df['y'].count() < 2:
            print(f"Skipping region {region}: not enough non-null data points ({region_df['y'].count()})")
            regions_with_insufficient_data.append(region)
            continue
        
        try:
            # Initialize and train Prophet model
            model = Prophet(
                yearly_seasonality=True, 
                weekly_seasonality=False, 
                daily_seasonality=False
            )
            model.fit(region_df)
            
            # Store the trained model
            prophet_models[region] = model
            print(f"✅ Trained Prophet model for region: {region}")
            
        except Exception as e:
            print(f"❌ Error training model for {region}: {str(e)}")
            regions_with_insufficient_data.append(region)
    
    print(f"\n🎉 Successfully trained {len(prophet_models)} Prophet models.")
    if regions_with_insufficient_data:
        print(f"⚠️  Skipped {len(regions_with_insufficient_data)} regions due to insufficient data.")
    
    return prophet_models

def save_models(prophet_models):
    """Save the trained models"""
    print("Saving trained models...")
    
    # Save all models to a single joblib file
    dump(prophet_models, 'prophet_models_all_regions.joblib')
    print(f"✅ Saved {len(prophet_models)} Prophet models to 'prophet_models_all_regions.joblib'")
    
    # Also save a list of available regions
    available_regions = list(prophet_models.keys())
    dump(available_regions, 'available_regions.joblib')
    print(f"✅ Saved list of {len(available_regions)} available regions")

def main():
    """Main function to train and save Prophet models"""
    print("🚀 Starting Prophet model training process...")
    print("=" * 50)
    
    try:
        # Load and preprocess data
        data = load_and_preprocess_data()
        
        # Train models
        prophet_models = train_prophet_models(data)
        
        if not prophet_models:
            print("❌ No models were trained successfully. Please check your data.")
            return
        
        # Save models
        save_models(prophet_models)
        
        print("=" * 50)
        print("🎉 Prophet model training completed successfully!")
        print(f"📊 Trained models for {len(prophet_models)} regions")
      
    except FileNotFoundError as e:
        print(f"❌ Error: {str(e)}")
        print("Please ensure 'Data/State_time_series.csv' exists in the project directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print("Please check your data and try again.")

if __name__ == "__main__":
    main()
