###  RealtyAI  Smart Real Estate Insight Platform

Streamlit app to explore Prophet-based home price forecasts and run XGBoost price predictions.

## Features
- Forecasting tab:
  - Loads Prophet forecast CSVs and plots interactive charts (Plotly) with confidence interval shading
  - Region selector (if available) and date-range filter
  - Table view and summary stats
- Prediction tab:
  - Batch CSV predictions with downloadable results
  - Single-record form for quick what-if prediction
  - Accepts a bundled model file in the folder or a model uploaded from the UI
- Caching of data/model loads, graceful fallbacks, and clear diagnostics

## Folder structure (key files)
- `app.py` — Streamlit UI
- `requirements.txt` — Python dependencies
- Prophet data (any of these):
  - `Prophet_AllRegions_Forecasts_Renamed.csv` (preferred)
  - `Forecast_California_House_Prices.csv` (fallback)
- Optional historical data: `State_time_series_ZHVI_cleaned.csv`
- Prediction models (put at least one next to `app.py`):
  - `xgb_pipeline.joblib` (preferred; full sklearn Pipeline)
  - `xgb_model.joblib` (raw model; you must ensure input features match)
- Optional sample inputs: `unseen_real_estate_data.csv`

## Quick start (Windows, PowerShell)
```powershell
cd "C:\Users\jvsti\Downloads\REALESTATE IN INDIAN CITES"

# optional venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# run app
streamlit run app.py
# if streamlit not found: python -m streamlit run app.py
```
The browser will open at `http://localhost:8501`.

## Place your model
Put your trained model next to `app.py` with one of these filenames:
- `xgb_pipeline.joblib` — an sklearn Pipeline that includes preprocessing + model
- `xgb_model.joblib` — a model with a `.predict()` method

Or, load from the UI:
- Open the Prediction tab → “Upload model (.joblib/.pkl)” → select your model file.

> Note: The app validates the object has a `predict()` method. Uploading arrays (e.g., predictions saved as `.joblib`) won’t work.

## Prepare prediction input CSV
- Your CSV must contain the columns your model expects. Column order doesn’t matter, names do.
- You can use `unseen_real_estate_data.csv` as a template. Common numeric fields used by the single-record form:
  - `Total_Area`, `Bedrooms`, `Baths`, `Balcony`, `Price_per_SQFT`
- Additional categorical fields (e.g., `Location`) should match what your pipeline handles.

## Forecast data
- Place `Prophet_AllRegions_Forecasts_Renamed.csv` in the folder for multi-region forecasts.
- If unavailable, the app will fallback to `Forecast_California_House_Prices.csv`.
- Optional: `State_time_series_ZHVI_cleaned.csv` for historical ZHVI viewing.

## Troubleshooting
- Streamlit not recognized:
  - `python -m streamlit run app.py`
- Port in use:
  - `streamlit run app.py --server.port 8502`
- Model not loaded / invalid:
  - Ensure the file is a trained model/pipeline (object with `.predict()`), not an array
  - Try re-saving from training:
    ```python
    from joblib import dump
    dump(pipeline, "xgb_pipeline.joblib")  # preferred
    # or
    dump(model, "xgb_model.joblib")
    ```
- Feature mismatch:
  - Make sure your CSV headers exactly match the training features expected by the pipeline
- Cache issues:
  - In the app menu (top-right), choose “Clear cache and rerun”

## Notes
- Prophet isn’t required at runtime because the app reads forecast CSVs. On Windows, Prophet install may be skipped.
- Python 3.9–3.11 recommended for widest library compatibility.
