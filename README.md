🏡 RealtyAI – Smart Real Estate Insight Platform

An AI-powered real estate insight platform built with Streamlit that predicts property prices and forecasts housing trends using Machine Learning and Prophet models.

🚀 Features
💰 Price Prediction

Predict property prices accurately based on user inputs.

Takes input features like:

📍 Location / City

🏠 BHK (Bedrooms)

🚿 Bathrooms

📏 Total Area (in Sq. Ft.)

🏗 Year Built

Utilizes a Bagging Regressor model trained on real estate datasets.

Generates instant, easy-to-understand price estimates.

Displays prediction results with a clean and interactive interface.

📈 Forecast Future Prices

Forecast future housing price trends using time series models.

Select any U.S. state from a dropdown list.

Uses Prophet models trained for each region.

Provides:

⏳ Forecast for the next 12 months

📊 Line chart of predicted values over time

📉 Confidence interval visualization (trend clarity)

Helps users, investors, and analysts identify long-term property trends.

🎨 Visualization & User Interface

A modern and responsive interface built with Streamlit.

Sidebar-based navigation for easy access between pages.

Styled layout using custom CSS for:

Gradient backgrounds

Rounded cards

Custom buttons with hover effects

Interactive charts for real-time forecast visualization.

Works seamlessly on both desktop and mobile browsers.

🧰 Technology Stack

Framework: Streamlit

Machine Learning: scikit-learn, Prophet

Data Handling: pandas, numpy, joblib

Visualization: Streamlit Line Charts

⚙️ Setup Instructions

Clone the Repository

git clone <your_repo_url>
cd RealtyAI


Install Dependencies

pip install streamlit pandas numpy scikit-learn prophet joblib


Place Model Files
Ensure the following files are available in your Downloads folder:

bagging_regressor_model.pkl
real_estate_pipeline.pkl
all_prophet_models.pkl


Run the App

streamlit run App.py

🧠 Model Details

Price Prediction Model: Bagging Regressor (Decision Tree-based)

Forecast Model: Facebook Prophet (Region-wise forecasting)

Forecast Duration: 12 months (extendable)

🪪 License

This project is licensed under the MIT License.
