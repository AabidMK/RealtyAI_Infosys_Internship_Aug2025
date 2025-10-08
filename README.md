# Reality AI: Smart Real Estate Insight Platform

Reality AI is an advanced platform aiming to deliver smart insights for the real estate sector. It provides robust predictive analytics, including house price prediction and future trend forecasting, integrating the power of modern machine learning and time series analysis.

## Features

- **House Price Prediction Model**
  - End-to-end pipeline for predicting real estate prices.
  - Data preprocessing: EDA, feature engineering, log transformation, encoding, and scaling.
  - Multiple regression models evaluated (R², RMSE), with Lasso Regression finalized.
  - Model persistence enabled via `joblib`.
- **Time Series Forecasting Model**
  - Uses Facebook's Prophet for forecasting real estate trends.
  - Accurate forecast visualization and analysis.
- **Modern Full Stack Architecture**
  - Frontend: React
  - Backend: FastAPI
  - Model deployment and integration between components for seamless predictions.


### Model Artifacts

- Lasso regularized regression model: `lasso_pipeline.pkl`
- Prophet forecasting model: `all_region_forecasts.csv`

### Data Requirements

- Placed the required datasets.
- Ensuring column names and format match training requirements.

## Usage

- Use the platform to generate house price predictions and view time series forecasts.
- Backend endpoints deployed via FastAPI for model inference.

## Model Development Overview

### House Price Prediction

- Preprocessing steps: EDA, feature engineering, categorical encoding, numerical transformations, scaling.
- Model selection: Baseline and advanced models evaluated (linear regression, ridge, lasso, etc.).
- Final model: **Lasso Regression** selected based on highest R² and lowest RMSE.
- Model saved using `joblib` for efficient serving.

### Time Series Forecasting

- EDA and stationarity checks conducted.
- Prophet model chosen due to strong performance with real estate temporal data.
- Time series predictions saved and visualized via frontend dashboard.

## Tech Stack

- **Frontend:** React.js
- **Backend:** FastAPI (Python)
- **ML Libraries:** scikit-learn, pandas, numpy, Prophet, joblib
- **Deployment:** Modular for local development; can be containerized for production
