---

# 🏡 RealtyAI – Smart Real Estate Insight Platform

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-ScikitLearn-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

> **RealtyAI** is an intelligent real estate analytics platform that predicts and forecasts housing prices using advanced **Machine Learning** and **Time Series Forecasting** models.
> It provides insights into market trends and property valuations through an **interactive Streamlit dashboard**.

---

## 🧭 Table of Contents

* [ Overview](#-overview)
* [ Project Structure](#-project-structure)
* [ Modules Breakdown](#️-modules-breakdown)

  * [🏘️ House Price Prediction](#️-1-house-price-prediction)
  * [💻 Streamlit App](#-2-streamlit-app)
  * [📈 Time Series Forecast](#-3-time-series-forecast)
* [⚙️ Installation](#️-installation)
* [ Usage](#usage)
* [ Features](#-features)
* [ Contributing](#-contributing)
* [ License](#-license)
* [ Acknowledgments](#acknowledgments)

---

## 📘 Overview

**RealtyAI** integrates cutting-edge machine learning models and forecasting techniques to provide:

* 🏠 **Accurate House Price Predictions** using ensemble learning.
* 📊 **Future Price Forecasting** powered by **Prophet** models.
* 🌐 **Interactive Dashboard** for visualization and real-time predictions.

This project aims to help users, developers, and researchers gain **data-driven insights** into real estate markets.

---

## 📂 Project Structure

```
RealtyAI Smart House Prediction Model/
├── House Price Prediction/
│   ├── Bagging_Regressor_Pipeline.ipynb
│   ├── feature_engineering.py
│   ├── Real Estate Data V21.csv
│   ├── real_estate_inference.py
│   ├── real_estate_pipeline_v20250924_122040.joblib
│
├── Streamlit/
│   ├── realty_ai.py
│   ├── feature_engineering.py
│   ├── all_states_prophet_models.pkl
│   ├── real_estate_pipeline_v20250924_122040.joblib
│   ├── __pycache__/
│
├── TimeSeries Forecast/
│   ├── realty_ai.py
│   ├── feature_engineering.py
│   ├── all_states_prophet_models.pkl
│   ├── State_time_series.csv
│   ├── real_estate_pipeline_v20250924_122040.joblib
│
├── AI Project_ RealtyAI Smart Real Estate Insight Platform.pdf
├── LICENSE
└── README.md
```

---

## 🏗️ Modules Breakdown

### 🏘️ 1. House Price Prediction

This module contains the **core machine learning pipeline** for house price prediction using **Bagging Regressor**.

**Key Components:**

| File                                           | Description                                              |
| ---------------------------------------------- | -------------------------------------------------------- |
| `Bagging_Regressor_Pipeline.ipynb`             | Model training and evaluation notebook.                  |
| `feature_engineering.py`                       | Contains preprocessing and feature engineering logic.    |
| `real_estate_inference.py`                     | Script for generating predictions using the saved model. |
| `Real Estate Data V21.csv`                     | Dataset for training and validation.                     |
| `real_estate_pipeline_v20250924_122040.joblib` | Serialized trained model.                                |

---

### 💻 2. Streamlit App

A **user-friendly dashboard** that allows interactive predictions and visual trend analysis.

**Key Components:**

| File                                           | Description                              |
| ---------------------------------------------- | ---------------------------------------- |
| `realty_ai.py`                                 | Main Streamlit application.              |
| `feature_engineering.py`                       | Reusable preprocessing module.           |
| `all_states_prophet_models.pkl`                | Pre-trained Prophet forecasting models.  |
| `real_estate_pipeline_v20250924_122040.joblib` | ML model for property price predictions. |

**Run the App:**

```bash
cd Streamlit
streamlit run realty_ai.py
```

---

### 📈 3. Time Series Forecast

This module performs **temporal forecasting** of housing prices using the **Prophet** model for state-wise predictions.

**Key Components:**

| File                            | Description                                |
| ------------------------------- | ------------------------------------------ |
| `realty_ai.py`                  | Forecasting script.                        |
| `feature_engineering.py`        | Time-series preprocessing.                 |
| `all_states_prophet_models.pkl` | Prophet forecasting models.                |
| `State_time_series.csv`         | Dataset for historical real estate prices. |

---

## ⚙️ Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/vishnuvardhan7569/realtyai.git
   cd realtyai
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

##  Usage

### 🏠 For House Price Prediction

```bash
cd "House Price Prediction"
jupyter notebook Bagging_Regressor_Pipeline.ipynb
```

Train the model and test it on real estate data.

### 🌐 For Streamlit Dashboard

```bash
cd Streamlit
streamlit run realty_ai.py
```

Access the web app to predict prices and view trends interactively.

### 📊 For Time Series Forecasting

```bash
cd "TimeSeries Forecast"
python realty_ai.py
```

Generate long-term forecasts for property trends.

---

## 💡 Features

✅ **Machine Learning Pipeline** – Predict prices using Bagging Regressor.

✅ **Time Series Forecasting** – Predict future trends using Prophet.

✅ **Interactive Streamlit Dashboard** – For real-time insights and visualizations.

✅ **Reusable Feature Engineering** – Modular design for scalability and clarity.

✅ **Clean Architecture** – Organized structure for maintainability.

---

## 🤝 Contributing

We welcome contributions from the community!

1. Fork this repository
2. Create a new branch:

   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and commit them:

   ```bash
   git commit -m "Added new feature"
   ```
4. Push to your fork and open a pull request:

   ```bash
   git push origin feature-branch
   ```

---

## 📜 License

This project is licensed under the [MIT License](./LICENSE).
You are free to use, modify, and distribute this software with attribution.

---

##   Acknowledgments

This project is built with the help of these amazing open-source technologies:

*  [Scikit-learn](https://scikit-learn.org/) – Machine Learning Pipelines
*  [Streamlit](https://streamlit.io/) – Interactive Web Apps
*  [Facebook Prophet](https://facebook.github.io/prophet/) – Time Series Forecasting

---

###  *Empowering Real Estate Decisions through Artificial Intelligence* 

---
