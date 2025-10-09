# RealtyAI - Smart Real Estate Insight Platform

Welcome to the **RealtyAI** repository! This project is designed to predict and forecast real estate prices using advanced machine learning models. It is structured into three main components:

## Project Structure

```
RealtyAI/
├── House Price Prediction/
│   ├── Bagging_Regressor_Pipeline.ipynb
│   ├── feature_engineering.py
│   ├── Real Estate Data V21.csv
│   ├── real_estate_inference.py
│   ├── real_estate_pipeline_v20250924_122040.joblib
├── Streamlit/
│   ├── all_states_prophet_models.pkl
│   ├── feature_engineering.py
│   ├── real_estate_pipeline_v20250924_122040.joblib
│   ├── realty_ai.py
│   ├── __pycache__/
├── TimeSeries Forecast/
│   ├── all_states_prophet_models.pkl
│   ├── feature_engineering.py
│   ├── real_estate_pipeline_v20250924_122040.joblib
│   ├── realty_ai.py
│   ├── State_time_series.csv
├── AI Project_ RealtyAI Smart Real Estate Insight Platform.pdf
├── LICENSE
└── README.md                      # This file
```

### 1. **House Price Prediction**
This folder contains the pipeline for predicting house prices using a Bagging Regressor model. Key files include:
- **`Bagging_Regressor_Pipeline.ipynb`**: Jupyter notebook for training and evaluating the model.
- **`feature_engineering.py`**: Contains the `RealEstateFeatureEngineer` class for data preprocessing.
- **`real_estate_inference.py`**: Script for making predictions using the trained pipeline.
- **`Real Estate Data V21.csv`**: Dataset used for training and testing.
- **`real_estate_pipeline_v20250924_122040.joblib`**: Serialized model pipeline.

### 2. **Streamlit**
This folder contains the Streamlit application for interactive real estate price prediction and forecasting. Key files include:
- **`realty_ai.py`**: Main Streamlit app script.
- **`feature_engineering.py`**: Preprocessing utilities for the app.
- **`all_states_prophet_models.pkl`**: Serialized Prophet models for time series forecasting.
- **`real_estate_pipeline_v20250924_122040.joblib`**: Serialized price prediction pipeline.

### 3. **TimeSeries Forecast**
This folder focuses on time series forecasting for real estate prices. Key files include:
- **`realty_ai.py`**: Script for forecasting real estate trends.
- **`feature_engineering.py`**: Preprocessing utilities for time series data.
- **`all_states_prophet_models.pkl`**: Serialized Prophet models for forecasting.
- **`State_time_series.csv`**: Dataset for time series analysis.

## Features
- **House Price Prediction**: Predict property prices based on features like location, size, and amenities.
- **Time Series Forecasting**: Forecast future real estate trends using Prophet models.
- **Interactive Dashboard**: Streamlit-based app for user-friendly interaction.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/vishnuvardhan7569/realtyai.git
   cd realtyai
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### House Price Prediction
1. Open the Jupyter notebook in the `House Price Prediction` folder.
2. Run the cells to train the model and make predictions.

### Streamlit App
1. Navigate to the `Streamlit` folder.
2. Run the app:
   ```bash
   streamlit run realty_ai.py
   ```

### Time Series Forecasting
1. Use the scripts in the `TimeSeries Forecast` folder to analyze and forecast real estate trends.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgments
- **Scikit-learn** for machine learning pipelines.
- **Streamlit** for the interactive dashboard.
- **Prophet** for time series forecasting.

---
Happy coding! 🚀
