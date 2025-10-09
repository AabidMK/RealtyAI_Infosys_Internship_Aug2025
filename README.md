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

## Project Structure

```bash
Real_Estate_Price_Prediction_Model/
│
├── backend/
│   ├── main.py
│   ├── testData.txt
│   ├── requirements.txt
│   └── __pycache__/
│
├── Full_Pipeline_House_Price_Prediction/
│   ├── data/
│   │   └── Real Estate Data V21.csv
│   ├── models/
│   │   └── real_estate_pipeline_adaboost.joblib
│   └── src/
│       ├── feature_engineering.py
│       ├── inference.py
│       ├── train_pipeline.py
│
├── House_Price_Prediction_Time_Series_Forecasting/
│   ├── models/
│   │   └── all_states_models.pkl
│   ├── State_time_series.csv
│   └── Time_Series_Forecasting_Price_Prediction.ipynb
│
├── App.py
├── README.md
├── .gitignore
└── LICENSE

```

## Prerequisites  

### System Requirements  
- **OS:** Windows 10/11, macOS, or Linux  
- **Python:** 3.10+  
- **Node.js:** 16.x+  
- **npm:** 8.x+  

### Package Managers  
- **npm**: For frontend dependencies  

---

## Installation & Setup  

### Step 1: Clone the Repository  
```bash
cd d:\dev\test\internship
git clone https://github.com/AabidMK/RealtyAI_Infosys_Internship_Aug2025.git
cd RealtyAI_Infosys_Internship_Aug2025
git checkout aditi_nagave
```

### Step 2: Create & activate a virtual environment
```
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```
2.2 Install Backend Dependencies

Make sure you are inside your backend folder:

cd backend


Then install dependencies via pip:

pip install -r requirements.txt


⚠️ Ensure requirements.txt includes packages like:

fastapi
uvicorn
pandas
scikit-learn
joblib

2.3 Verify Model Files

Ensure your trained model files exist in:

/backend/Models
├── real_estate_pipeline_v20250915_182141.joblib
└── all_region_models.joblib


If missing:

Train models using the provided Jupyter notebooks.

Or download pre-trained .joblib files.

2.4 Run the FastAPI Backend
uvicorn main:app --reload


Replace main with your FastAPI file name if different.

Access API docs at 👉 http://127.0.0.1:8000/docs

🎨 Step 3: Frontend Setup (Streamlit)
3.1 Navigate to frontend
cd ../frontend

3.2 Install Frontend Dependencies
pip install -r requirements.txt


⚠️ Ensure requirements.txt includes:

streamlit
pandas
matplotlib
seaborn
joblib

3.3 Run Streamlit App
streamlit run app.py


Replace app.py with your Streamlit file if different.

Frontend will open in browser at http://localhost:8501


