#!/usr/bin/env python3
import pandas as pd
import joblib
import os
from typing import Dict

from feature_engineering import RealEstateFeatureEngineer

class RealEstatePredictor:

    def __init__(self, model_dir: str = "/content"):
        self.model_dir = model_dir
        self.pipeline = None

    def load_model(self):
        model_files = [f for f in os.listdir(self.model_dir)
                      if f.startswith("real_estate_pipeline_") and f.endswith(".joblib")]
        if not model_files:
            raise FileNotFoundError("No model files found in 'models' directory!")

        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(self.model_dir, latest_model)
        self.pipeline = joblib.load(model_path)
        print(f"✅ Model loaded: {latest_model}")

    def predict(self, property_data: Dict) -> float:
        if self.pipeline is None:
            self.load_model()
        df = pd.DataFrame([property_data])
        prediction = self.pipeline.predict(df)
        return float(prediction[0])

# Helper function
def predict_price(property_data: Dict) -> float:
    predictor = RealEstatePredictor()
    return predictor.predict(property_data)

# Example usage
def main():
    sample_data = [
        {
            'Property Title': '4 BHK Flat for sale in Kanathur Reddikuppam, Chennai',
            'Price': '₹1.99 Cr',
            'Location': 'Kanathur Reddikuppam, Chennai',
            'Total_Area': 2583,
            'Price_per_SQFT': 7700.0,
            'Baths': 4,
            'Balcony': 'Yes'
        },
        {
            'Property Title': '10 BHK Independent House for sale in Pozhichalur, Chennai',
            'Price': '₹2.25 Cr',
            'Location': 'Ramanathan Nagar, Pozhichalur, Chennai',
            'Total_Area': 7000,
            'Price_per_SQFT': 3210.0,
            'Baths': 6,
            'Balcony': 'Yes'
        },
        {
            'Property Title': '3 BHK Flat for sale in West Tambaram, Chennai',
            'Price': '₹1.0 Cr',
            'Location': 'Kasthuribai Nagar, West Tambaram, Chennai',
            'Total_Area': 1320,
            'Price_per_SQFT': 7580.0,
            'Baths': 3,
            'Balcony': 'No'
        },
        {
            'Property Title': '7 BHK Independent House for sale in Triplicane, Chennai',
            'Price': '₹3.33 Cr',
            'Location': 'Naveenilaya, Chepauk, Triplicane, Chennai',
            'Total_Area': 4250,
            'Price_per_SQFT': 7840.0,
            'Baths': 5,
            'Balcony': 'Yes'
        },
        {
            'Property Title': '2 BHK Flat for sale in Avadi, Chennai',
            'Price': '₹48.0 L',
            'Location': 'Avadi, Chennai',
            'Total_Area': 960,
            'Price_per_SQFT': 5000.0,
            'Baths': 3,
            'Balcony': 'Yes'
        },
        {
            'Property Title': '2 BHK Flat for sale in Siruseri, Chennai',
            'Price': '₹40.0 L',
            'Location': 'Siruseri, Chennai',
            'Total_Area': 940,
            'Price_per_SQFT': 4250.0,
            'Baths': 3,
            'Balcony': 'No'
        }
    ]

    for property_data in sample_data:
        price = predict_price(property_data)
        print(f"{property_data['Property Title']}")
        print(f"Predicted Price: ₹{price:.1f} Lakhs (₹{price/100:.2f} Cr)\n")

if __name__ == "__main__":
    main()