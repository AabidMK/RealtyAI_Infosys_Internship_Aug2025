# src/inference.py
import pandas as pd
import joblib
import os
from feature_engineering import RealEstateFeatureEngineer

class RealEstatePredictor:
    def __init__(self, model_path="models/real_estate_pipeline_adaboost.joblib"):
        self.model_path = model_path
        self.load_pipeline()

    def load_pipeline(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Model file not found!")
        saved = joblib.load(self.model_path)
        self.pipeline = saved['pipeline']
        self.features = saved['features']
        self.numeric_features = saved['numeric_features']
        self.categorical_features = saved['categorical_features']

    def preprocess(self, df):
        fe = RealEstateFeatureEngineer(remove_outliers=False)  # no outlier removal in inference
        df = fe.transform(df)

        # Ensure all expected columns exist
        for col in self.features:
            if col not in df.columns:
                df[col] = 0  # default fill for missing engineered features

        return df[self.features]

    def predict(self, property_data: dict):
        df = pd.DataFrame([property_data])
        df = self.preprocess(df)
        prediction = self.pipeline.predict(df)
        return float(prediction[0])


if __name__ == "__main__":
    predictor = RealEstatePredictor()
    sample_property = {
        'Location': 'Whitefield, Bangalore',
        'Property Title': '1 RK Apartment',
        'Total_Area': 1200,
        'Price_per_SQFT': 5000,
        'Baths': 2,
        'Balcony': 'Yes'
    }
    price = predictor.predict(sample_property)
    print(f"Property: {sample_property['Property Title']} in {sample_property['Location']}")
    print(f"Predicted Price: ₹{price:.2f} Lakhs (₹{price/100:.2f} Crores)")
