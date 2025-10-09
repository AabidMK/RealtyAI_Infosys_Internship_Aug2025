###  RealtyAI  Smart Real Estate Insight Platform

Develop an AI platform that evaluates property conditions, predicts price trends, and 
segments satellite images of real estate regions. This system is useful for property 
buyers, investors, and urban planners. 

### Table of Contents

- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Models](#-models)
- [License](#-license)
- [Contributors](#-contributors)

## Features

### Price Prediction
- Predict real estate prices based on property attributes.  
- **Input Parameters:** Location, City, BHK, Total Area, Price per SQFT, Bathrooms, Balcony.  
- Uses **Adaboost** ML pipeline for accurate predictions.  

### Time Series Forecasting
- **Single Region Forecast:** Generate forecasts with confidence intervals.     
- **Forecast Horizon:** 1–36 months ahead.  
- Built with **Prophet** for reliable forecasting.

## Technology Stack

### Backend
- **Framework:** FastAPI  
- **Core Libraries:**  
  - scikit-learn `1.7.1`  
  - Prophet `1.1.0+`  
  - pandas `2.0.0+`  
  - numpy `1.24.0+`  
  - joblib `1.3.0+` (for model serialization)  
- **Purpose:** Exposes REST API endpoints for price prediction and forecasting.

### Frontend
- **Framework:** Streamlit  
- **Charts & Visualization:** Plotly / Matplotlib  
- **Purpose:** Provides an interactive user interface to input property details and view analytics.  




