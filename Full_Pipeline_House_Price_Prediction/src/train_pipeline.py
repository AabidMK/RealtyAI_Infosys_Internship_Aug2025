import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from feature_engineering import RealEstateFeatureEngineer


def main():
    # Load data 
    data = pd.read_csv("data/Real Estate Data V21.csv")

    # Feature Engineering
    fe = RealEstateFeatureEngineer()
    data_featured = fe.fit_transform(data)

    # Features 
    numeric_features = [
        'log_area', 'Baths', 'Balcony_flag', 'BHK_or_RK_flag', 'log_area_per_room',
        'Bath_to_BHK_ratio', 'Total_Rooms', 'Area_Efficiency', 'Area_x_Baths',
        'log_Area_x_BHK', 'Is_Premium_Size', 'Has_Multiple_Baths',
        'Advanced_Luxury_Score','Price_per_Room'
    ]

    categorical_features = [
        'City', 'Locality', 'Property_Size_Category', 'BHK_Category'
    ]

    target_column = 'Price_Lakhs'
    all_features = numeric_features + categorical_features

    final_data = data_featured[all_features + [target_column]].copy()
    final_data = final_data.dropna(subset=[target_column])

    # Split dataset
    X = final_data[all_features]
    y = final_data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define Pipeline
    pipeline = Pipeline([
        ("encoder", TargetEncoder(cols=categorical_features)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", AdaBoostRegressor(
            n_estimators=200,
            learning_rate=0.1,
            random_state=42
        ))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    # Training performance
    y_train_pred = pipeline.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    # Testing performance
    y_test_pred = pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(" Model Performance:")
    print(f"    Train R² Score: {train_r2:.4f}")
    print(f"    Train RMSE   : {train_rmse:.4f}")
    print(f"    Test R² Score: {test_r2:.4f}")
    print(f"    Test RMSE    : {test_rmse:.4f}")

    # Save pipeline
    os.makedirs("models", exist_ok=True)
    pipeline_path = "models/real_estate_pipeline_adaboost.joblib"
    joblib.dump({
        "pipeline": pipeline,
        "features": all_features,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features
    }, pipeline_path)

    print(f"Pipeline saved to {pipeline_path}")


if __name__ == "__main__":
    main()
