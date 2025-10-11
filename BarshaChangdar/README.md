# RealityAI — Real Estate Forecasting Project

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Built%20with-Python-3776AB?logo=python&logoColor=white" alt="Python Badge" />
  </a>
  <a href="https://colab.research.google.com/">
    <img src="https://img.shields.io/badge/Made%20in-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white" alt="Google Colab Badge" />
  </a>
  <a href="https://facebook.github.io/prophet/">
    <img src="https://img.shields.io/badge/Forecasting-Prophet-00BFFF?logo=meta&logoColor=white" alt="Prophet Badge" />
  </a>
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit Badge" />
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT" />
  </a>
</p>

---

## Table of Contents
- [About the Project](#about-the-project)
- [Objectives](#objectives)
- [Tools and Technologies](#tools-and-technologies)
- [Repository Structure](#repository-structure)
- [Features](#features)
- [Installation](#installation)
- [Models](#models)
- [Results](#results)
- [Author](#author)
- [Acknowledgement](#acknowledgement)
- [License](#license)

---

## About the Project
**RealityAI** is a real estate forecasting project developed as part of my **Infosys Internship (Aug 2025)**.  
It uses machine learning and time series forecasting with Facebook Prophet to analyze and predict real estate market trends across various U.S. regions.

Key goals include:  
- Developing price prediction models based on property features  
- Creating state-wise time series forecasting models  
- Providing an interactive visualization dashboard via Streamlit  

---

## Objectives
- Clean and preprocess real estate datasets  
- Build machine learning models for price prediction  
- Apply Prophet for time series forecasting  
- Visualize and compare historical vs. forecasted data  
- Deliver an interactive app for data exploration  

---

## Tools and Technologies

| Category        | Tools / Libraries               |
|-----------------|-------------------------------|
| Programming     | Python 3.10+, Google Colab      |
| Machine Learning| scikit-learn, BaggingRegressor |
| Forecasting     | Facebook Prophet               |
| Data Analysis   | pandas, numpy                  |
| Visualization   | matplotlib, seaborn, streamlit |
| Version Control | Git, GitHub                   |

---

## Repository Structure
```
Reality-AI/
├── Models/
│ ├── encoder.pkl # Label encoder for categorical features
│ ├── prophet_models.pkl # Prophet models for multiple states
│
├── Notebooks/
│ ├── RealEstate.ipynb
│ ├── Timeseries.ipynb
│ ├── pipeline.ipynb
│ ├── Streamlit_app.ipynb
│
├── Script/
│ ├── prediction_vs.py
│
├── data/
│ ├── (datasets used for training)
│
└── README.md

```

---

## Features
** Property Price Prediction**  
- Predict prices using city, location, BHK, total area, price per sqft, bathrooms, and balconies  

** Time Series Forecasting**  
- Generate forecasts with confidence intervals for each state using Prophet  
- Compare forecasts across regions  
- Analyze historical trends and statistics  

** Visualization Dashboard**  
- Interactive and responsive Streamlit app to explore data and predictions  

** Data Processing Pipeline**  
- Includes preprocessing and encoding to support modeling  

---

## Installation

### 1. Clone the repository

git clone https://github.com/BarshaChangdar/Reality-AI.git
cd Reality-AI
2. Install dependencies
bash
Copy code
pip install -r requirements.txt
3. Run the Streamlit dashboard
bash
Copy code
streamlit run Notebooks/Streamlit_app.ipynb
4. Alternatively
You can open and run the notebooks directly in Google Colab or Jupyter Notebook to explore results.

Models
File	Description
encoder.pkl	Label encoder for categorical features
prophet_models.pkl	Prophet forecasting models for multiple states
bagging_model.joblib (optional)	ML pipeline for price prediction

Results
Forecasted property prices and trends with confidence intervals

Visualized multi-state comparisons and historical data

Integrated ML models with time series for real-world insights

Author
Barsha Changdar
Infosys RealtyAI Internship (Aug 2025)

Mentor: Aabid MK

Acknowledgement
Special thanks to my mentor Aabid MK for his invaluable guidance and continuous support,
and to Infosys for providing the opportunity to work on real-world forecasting and analytics.

<p align="center">Designed and developed by <b>Barsha Changdar</b></p>
License
This project is licensed under the MIT License — see the LICENSE file for details.

sql
Copy code
MIT License

Copyright (c) 2025 Aabid M K

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
<p align="center">© 2025 RealityAI — All rights reserved.</p> 
