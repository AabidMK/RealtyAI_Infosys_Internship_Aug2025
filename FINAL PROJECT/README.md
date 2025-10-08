# 🏠 House Price Prediction & Forecasting System

A machine learning system for predicting real estate prices and forecasting housing market trends.

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the price prediction model:
```bash
python train.py
```

3. Train the forecasting models:
```bash
python train_forecasting_models.py
```

4. Run the application:
```bash
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
├── Data/
│   ├── Real Estate Data V21.csv      # Property data for price prediction
│   └── State_time_series.csv         # Time series data for forecasting
├── streamlit_app.py                  # Main application
├── train.py                          # Train price prediction model
├── train_forecasting_models.py       # Train forecasting models
├── forecasting_utils.py              # Forecasting utility functions
├── pipeline.pkl                      # Trained price prediction model
├── prophet_models_all_regions.joblib # Trained forecasting models
├── available_regions.joblib          # List of available regions for forecasting
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🎯 Key Features

### Price Prediction
- Predict house prices based on property features
- Interactive form with property details (area, BHK, bathrooms, balcony, location)
- Automatic feature extraction from property title
- Price visualization with metrics (price per sq ft, etc.)

### Market Forecasting
- Forecast housing market trends using time series analysis
- Single region forecasting with confidence intervals
- Multi-region comparison charts
- Trend and seasonality component analysis
- Historical data visualization
- Forecast summaries with key metrics

## 🧠 How It Works

### Price Prediction
Uses Gradient Boosting Regressor with features like:
- Property area, BHK, bathrooms, balcony
- Location encoding and property type
- Engineered features (interactions, logarithmic transformations)

### Market Forecasting
Uses Facebook Prophet to analyze historical data and predict trends:
- Time series forecasting with seasonal components
- Confidence intervals for predictions
- Trend analysis and seasonal decomposition

## 📄 License

This project is licensed under the MIT License.