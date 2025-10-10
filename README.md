SmartPropAI – Intelligent Real Estate Analytics System
Project Overview:

SmartPropAI is an advanced AI-driven analytics solution that predicts residential property values and analyzes real estate market trends across major Indian cities. The platform integrates machine learning and time series forecasting techniques to assist property buyers, investors, and real estate consultants in making smarter, data-backed decisions.

Core Features:
AI-Based Property Price Estimation

Generates estimated property prices using key features such as city, built-up area (sq.ft), number of bedrooms (BHK), and bathrooms.

Employs a pre-trained ML pipeline (real_estate_price_pipeline.pkl) for accurate valuation.

Provides instant, data-driven predictions to enhance user decision-making.

City-Wise Market Forecasting

Utilizes Prophet forecasting models (city_forecast_models.pkl) to predict future real estate price trends for multiple cities.

Enables users to select a city and forecast duration for visualizing upcoming price movements.

Assists with strategic investments and long-term market planning.

Technology Stack

Languages & Libraries: Python, Pandas, NumPy, Scikit-learn, XGBoost, Prophet, Matplotlib, Seaborn
Tools & Platforms: Jupyter Notebook, Streamlit, Google Colab
Models Used: Regression Pipelines, Random Forest, XGBoost, Prophet

Setup & Execution
1. Environment Configuration

Install all required dependencies:

pip install pandas numpy scikit-learn xgboost prophet matplotlib seaborn streamlit

2. Launching the App

Run the Streamlit interface:

streamlit run app.py

Then, open the URL displayed in the terminal (usually http://localhost:8501
) to start interacting with SmartPropAI.

Data & Insights

Dataset: Indian_Real_Estate_Data.csv

Important Columns: City, Total Area, Bedrooms (BHK), Bathrooms, Price (in Lakhs), Listing Date

This dataset fuels both the price prediction model and the forecasting engine for comprehensive analysis.

Integrated Models

Property Price Prediction Model (real_estate_price_pipeline.pkl)

Regression-based ML pipeline for estimating current market values.

City Forecasting Model (city_forecast_models.pkl)

Prophet-driven model to project future price trends across cities.
