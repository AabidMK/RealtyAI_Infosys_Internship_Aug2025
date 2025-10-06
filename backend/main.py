from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import sys, os

# ----------------------------
# Add src folder to Python path
# ----------------------------
sys.path.append(os.path.join(os.getcwd(), "Full_Pipeline_House_Price_Prediction", "src"))
from inference import RealEstatePredictor

# ----------------------------
# Model Paths
# ----------------------------
PRICE_MODEL_PATH = "Full_Pipeline_House_Price_Prediction/models/real_estate_pipeline_adaboost.joblib"
TS_MODELS_PATH = "House_Price_Prediction_Time_Series_Forecasting/models/all_states_models.pkl"

# ----------------------------
# FastAPI Setup
# ----------------------------
app = FastAPI(title="Real Estate AI API")

# Load models
predictor = RealEstatePredictor(model_path=PRICE_MODEL_PATH)
ts_models = joblib.load(TS_MODELS_PATH)

# ----------------------------
# Request Models
# ----------------------------
class PriceRequest(BaseModel):
    Location: str
    City: str
    BHK: int
    Total_Area: float
    Price_per_SQFT: float
    Baths: int
    Balcony: str


class ForecastRequest(BaseModel):
    region: str
    horizon: int


# ----------------------------
# Routes
# ----------------------------
@app.post("/predict_price")
def predict_price(request: PriceRequest):
    try:
        property_data = {
            "Location": request.Location,
            "Property Title": f"{request.BHK} BHK Apartment",
            "Total_Area": request.Total_Area,
            "Price_per_SQFT": request.Price_per_SQFT,
            "Baths": request.Baths,
            "Balcony": request.Balcony,
            "City": request.City,
        }
        prediction = predictor.predict(property_data)
        return {"predicted_price": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast")
def forecast(request: ForecastRequest):
    try:
        if request.region not in ts_models:
            raise HTTPException(status_code=404, detail="Region not found")

        model = ts_models[request.region]
        if isinstance(model, dict):  # Handle dict inside dict case
            model = list(model.values())[0]

        future = pd.DataFrame({
            "ds": pd.date_range(start=pd.Timestamp.today(), periods=request.horizon, freq="M")
        })
        forecast = model.predict(future)
        forecast_df = forecast[["ds", "yhat"]].rename(
            columns={"ds": "Month", "yhat": "Forecasted Price"}
        )

        return {"forecast": forecast_df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available_regions")
def get_available_regions():
    try:
        regions = list(ts_models.keys())
        return {"regions": regions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

