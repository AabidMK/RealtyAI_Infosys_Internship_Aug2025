# RealtyAI - Smart Real Estate Insight Platform

A comprehensive AI-powered real estate analytics platform that provides price predictions and time series forecasting for various regions using Machine Learning and Prophet models.

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Models](#models)
- [License](#license)

## Features

### Price Prediction
- Predict real estate prices based on property features
- Input parameters: Location, City, BHK, Total Area, Price per SQFT, Bathrooms, Balcony
- Uses Adaboost ML pipeline for accurate predictions

### Time Series Forecasting
- Single Region Forecast: Detailed forecast with confidence intervals for one region
- Multi-Region Comparison: Compare forecasts across multiple regions
- Region Statistics: Historical data analysis and market insights
- Forecast horizon: 1-36 months
- Prophet-based forecasting models

### Visualization
- Interactive charts with Recharts
- Historical data vs. forecast comparison
- Confidence interval visualization
- Responsive design for all devices

## Technology Stack

### Backend
- Framework: FastAPI 0.104.0+
- ML Libraries: 
  - scikit-learn 1.7.1
  - Prophet 1.1.0+
  - pandas 2.0.0+
  - numpy 1.24.0+
- Model Serialization: joblib 1.3.0+

### Frontend
- Framework: Stremlit


## Project Structure

```
RealtyAI_Infosys_Internship_Aug2025/
│
├── backend/
│   ├── main.py                    # FastAPI application
│
├── Full_Pipeline_House_Price_Prediction/
│   ├── data/
│   │   └── Real Estate Data V21.csv
│   ├── models/
│   │   └── real_estate_pipeline_adaboost.joblib
│   └── src/
│       ├── feature_enginering.py
│       ├── train_pipeline.py
│       └── inference.py
│  
│
├── /House_Price_Prediction_Time_Series_Forecasting
│   ├── model/
│   │   └── all_region_models.pkl
│   ├── State_time_series.csv
│   └── Time_Series_Forecasting_Price_Prediction.ipynb              
│
├── app.py
├── AI Project_ RealtyAI Smart Real Estate Insight Platform.pdf
├── LICENSE
└── README.md                      # This file
```

## Prerequisites

Before you begin, ensure you have the following installed:

### System Requirements
- Operating System: Windows 10/11, macOS, or Linux
- Python: 3.10 or higher (required for scikit-learn 1.7.1)
- Node.js: 16.x or higher
- npm: 8.x or higher (comes with Node.js)

### Package Managers
- uv: Fast Python package installer ([Installation Guide](https://github.com/astral-sh/uv))


## Installation & Setup

### Step 1: Clone the Repository

```bash
cd d:\dev\test\internship
git clone https://github.com/AabidMK/RealtyAI_Infosys_Internship_Aug2025.git
cd RealtyAI_Infosys_Internship_Aug2025
git checkout aditi_nagave
```

### Step 2: Backend Setup

#### 2.1 Install FastAPI and Dependencies

**For Windows:**
```powershell
pip install fastapi uvicorn scikit-learn==1.7.1 prophet pandas numpy joblib pydantic
```
2.1.1 Verify Python and pip

**Check Python version:**
```
python --version
```
or
```
python3 --version
```

Should be 3.10 or higher.

**Check pip version:**

```pip --version```

2.1.2 Verify FastAPI and Uvicorn

Check that FastAPI and Uvicorn are installed:
```
pip show fastapi uvicorn
```
This will display version info for both packages.

Alternatively, you can verify Uvicorn directly:
```
uvicorn --version
```
#### 2.2 Verify Model Files

Ensure the following model files exist in the `Full_Pipeline_House_Price_Prediction` directory:
```
models
├── real_estate_pipeline_adaboost.joblib
```
Ensure the following model files exist in the `House_Price_Prediction_Time_Series_Forecasting` directory:
```
models
├── all_region_models.pkl
```

If missing, download from the project repository or train new models using the provided notebooks.

### Step 3: Frontend Setup

#### 3.1 Install Stremlit

```bash
pip install streamlit
```

## Running the Application

### Start Backend Server

Open a terminal and run:

```bash
uvicorn backend.main:app --reload
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [9680] using StatReload
INFO:     Started server process [11836]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Backend will be available at: **http://127.0.0.1:8000**

### Start Frontend Development Server

Open a **new terminal** and run:

```bash
streamlit run app.py
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://10.25.14.34:8501
```

Frontend will be available at: **http://localhost:8501**

## Models

### Price Prediction Model
- Algorithm: Adaboost with Decision Tree base estimators
- Features: Location, City, BHK, Total_Area, Price_per_SQFT, Bathroom, Balcony
- Training Data: Real estate listings from multiple Indian cities
- File: `real_estate_pipeline_adaboost.joblib`

### Time Series Forecasting Models
- Algorithm: Facebook Prophet
- Regions: 50+ US states/regions
- Training Period: 1996-2018 (historical ZHVI data)
- Forecast Capability: Up to 36 months ahead
- File: `all_region_models.pkl`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
