#!/usr/bin/env python3
import pandas as pd
import joblib
import os
from typing import Dict

from feature_engineering import RealEstateFeatureEngineer

class RealEstatePredictor:
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.pipeline = None
        
    def load_model(self):
        model_files = [f for f in os.listdir(self.model_dir) 
                      if f.startswith("real_estate_pipeline_") and f.endswith(".joblib")]
        
        if not model_files:
            raise FileNotFoundError("No model files found!")
        
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(self.model_dir, latest_model)
        
        self.pipeline = joblib.load(model_path)
        print(f" Model loaded: {latest_model}")
        
    def predict(self, property_data: Dict) -> float:

        if self.pipeline is None:
            self.load_model()
            
        df = pd.DataFrame([property_data])
        prediction = self.pipeline.predict(df)
        return float(prediction[0])


def predict_price(property_data: Dict) -> float:
    predictor = RealEstatePredictor()
    return predictor.predict(property_data)

def main():
    property_data = {
        'Location': 'Whitefield',
        'City': 'Bangalore',
        'BHK': 3,
        'Total_Area': 1000,
        'Price_per_SQFT': 5000,
        'Bathroom': 2,
        'Balcony': True
    }
    
    price = predict_price(property_data)
    
    print(f"Property: {property_data['BHK']} BHK in {property_data['Location']}")
    print(f"Predicted Price: ₹{price:.1f} Lakhs (₹{price/100:.2f} Crores)")

if __name__ == "__main__":
    main()