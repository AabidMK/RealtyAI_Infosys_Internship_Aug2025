### REALTYAI – INTELLIGENT REAL ESTATE ANALYTICS PLATFORM

### PROJECT SUMMARY

RealtyAI is an AI-powered platform designed to predict residential property prices and provide insights into market trends across major Indian cities. By combining machine learning models and time series forecasting, the system enables property buyers, investors, and real estate professionals to make informed decisions efficiently.

### MAIN CAPABILITIES:

1. Predictive Property Valuation

Estimates property prices based on location, area (sq.ft), number of bedrooms (BHK), and bathrooms.

Powered by a pre-trained machine learning pipeline (real_estate_price_pipeline.pkl).

Delivers real-time predictions for informed decision-making.

2. City-Level Market Forecasting

Uses Prophet models (city_forecast_models.pkl) to forecast future trends for multiple Indian cities.

Users can select a city and forecast duration to visualize projected market behavior.

Supports investment planning and strategic property decisions.


### TECHNOLOGY STACK:

Programming & Libraries: Python, Pandas, NumPy, Scikit-learn, XGBoost, Prophet, Matplotlib, Seaborn

Platforms & Tools: Jupyter Notebook, Streamlit, Google Colab

Models: Regression-based pipelines, Random Forest, XGBoost, Prophet


### PROJECT ORGANIZATION:

RealtyAI/

├── app.py
├── README.md 
├── requirements.txt                 
│

├── data/
│   └── Indian_Real_Estate_Data.csv  
│

├── models/
│   ├── real_estate_price_pipeline.pkl  
│   └── city_forecast_models.pkl 

├── notebooks/
│   ├── Data_Preprocessing.ipynb      
│   └── Time_Series_Analysis.ipynb      

└── assets/
    └── ui_screenshot.png               
    

### SETUP & USAGE:

### Environment Setup:

Install necessary Python libraries:

pip install pandas numpy scikit-learn xgboost prophet matplotlib seaborn streamlit

### Running the Application:

1.Launch the Streamlit app:

streamlit run app.py

2.Open the URL provided in the terminal (typically http://localhost:8501) to interact with the platform.


### DATA INSIGHTS:

1.Dataset: Indian_Real_Estate_Data.csv

2.Key Columns: City, Total Area, Bedrooms (BHK), Bathrooms, Price (in Lakhs), Listing Date

Used for both machine learning prediction and time series analysis.


### MODELS INCLUDED:

1.Property Price Predictor (real_estate_price_pipeline.pkl)

Regression-based pipeline for predicting market prices.

2.City Forecast Models (city_forecast_models.pkl)

Prophet-based models for forecasting future city-wise price trends.
