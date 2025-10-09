###  RealtyAI  Smart Real Estate Insight Platform

Develop an AI platform that evaluates property conditions, predicts price trends, and 
segments satellite images of real estate regions. This system is useful for property 
buyers, investors, and urban planners.

## ✨ Core Features

This application currently offers two core functionalities:

1.  **Real-Time Price Prediction**
    * Leverages a custom-built machine learning pipeline (`property_price_pipeline.joblib`).
    * Users can input property features such as location, total area, number of bedrooms (BHK), and bathrooms.
    * The model instantly predicts the estimated market price, providing valuable insights for buyers and sellers.

2.  **State-Level Market Forecasting**
    * Utilizes a collection of pre-trained Prophet models (`prophet_models.joblib`), one for each US state.
    * Analyzes historical data to forecast future trends in the Zillow Home Value Index (ZHVI).
    * Users can select a state and a forecast period (in months) to visualize projected market behavior and trends.

## 🛣️ Platform Roadmap

RealtyAI is an evolving platform. Future development will focus on expanding its capabilities to include:
* **Automated Property Condition Evaluation:** Using computer vision to assess the condition of properties from images.
* **Satellite Image Segmentation:** Analyzing satellite imagery to identify and classify real estate regions, aiding in urban planning and large-scale investment.

## 📂 Recommended Folder Structure

For clarity and organization, it's best to structure your project files as follows. You may need to create the `models`, `data`, and `notebooks` directories and move your files into them.

.
├── app.py                      # Main Streamlit application script
├── requirements.txt              # Python dependencies for the project
├── README.md                     # Project documentation (this file)
│
├── assets/
│   └── image_e0bf3c.png          # Screenshot for the README
│
├── data/
│   └── State_time_series.csv     # Dataset for time-series analysis
│
├── models/
│   ├── prophet_models.joblib         # Pre-trained models for forecasting
│   └── property_price_pipeline.joblib  # Pre-trained pipeline for price prediction
│
└── notebooks/
    └── Time_Series_EDA.ipynb     # Jupyter Notebook for exploratory data analysis


## 🚀 Getting Started

Follow these instructions to set up and run the application on your local machine.

### Prerequisites

* Python 3.8 or higher
* Git

### Installation & How to Run

1.  **Clone the repository:**
    Open your terminal or command prompt and clone this repository to your local machine.

2.  **Navigate to the project directory:**
    ```bash
    cd your-repository-name
    ```

3.  **Create and activate a virtual environment (Recommended):**
    This helps isolate project dependencies.

    * **On Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **On macOS & Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

4.  **Install the required libraries:**
    Use the `requirements.txt` file to install all necessary Python packages.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Streamlit application:**
    Now you are ready to start the app!
    ```bash
    streamlit run app.py
    ```

6.  **View the app in your browser:**
    After running the command, your web browser should open a new tab with the application running. If not, your terminal will provide a local URL (usually `http://localhost:8501`) that you can navigate to.

## 🤖 Models Included

This project uses two pre-trained models:

1.  **`property_price_pipeline.joblib`**: A Scikit-learn pipeline that takes property features (like location, area, and number of rooms) as input and predicts its market price.
2.  **`prophet_models.joblib`**: A dictionary of trained Prophet models, one for each US state, used to forecast future trends in the housing market based on historical data.

## 📊 Data

The time-series analysis and forecasting are based on the `State_time_series.csv` dataset, which contains various monthly real estate metrics from Zillow. Key features include the Zillow Home Value Index (ZHVI), Zillow Rent Index (ZRI), median listing prices, and inventory counts.

The initial exploratory data analysis (EDA), which informed the model development, is availa

