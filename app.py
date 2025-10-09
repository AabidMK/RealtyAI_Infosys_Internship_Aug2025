import os
import io
import json
import math
import warnings
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st

# Plotly for richer charts
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> Optional[pd.DataFrame]:
	if not os.path.exists(path):
		return None
	try:
		return pd.read_csv(path)
	except Exception:
		return None

@st.cache_data(show_spinner=False)
def list_unique(series: pd.Series) -> list:
	try:
		return sorted([x for x in series.dropna().unique().tolist() if str(x).strip() != ""])
	except Exception:
		return []

@st.cache_resource(show_spinner=False)
def try_load_model(path: str) -> Tuple[Optional[object], Optional[str]]:
	"""Try joblib, then cloudpickle. Returns (obj, error_message)."""
	if not os.path.exists(path):
		return None, f"File not found: {path}"
	# Try joblib
	try:
		from joblib import load as joblib_load
		return joblib_load(path), None
	except Exception as e1:
		# Fallback: cloudpickle
		try:
			import cloudpickle
			with open(path, "rb") as f:
				obj = cloudpickle.load(f)
			return obj, None
		except Exception as e2:
			return None, f"joblib error: {e1}; cloudpickle error: {e2}"

@st.cache_resource(show_spinner=False)
def ensure_predict_model(obj: object) -> Tuple[Optional[object], Optional[string]]:  # type: ignore[name-defined]
	"""Return (obj, None) if it has predict(); otherwise (None, reason)."""
	if obj is None:
		return None, "Model object is None"
	pred = getattr(obj, "predict", None)
	if callable(pred):
		return obj, None
	return None, f"Loaded object type {type(obj).__name__} has no predict(). Upload a trained pipeline/model."

# ---------- Data availability ----------
DATA_FILES = {
	"prophet_forecasts": "Prophet_AllRegions_Forecasts_Renamed.csv",
	"state_ts": "State_time_series_ZHVI_cleaned.csv",
	"forecast_california": "Forecast_California_House_Prices.csv",
	"predictions_unseen": "predictions_on_unseen.csv",
	"unseen_input": "unseen_real_estate_data.csv",
	"raw_listings": "Real Estate Data.csv",
}

MODEL_FILES = {
	"xgb_model": "xgb_model.joblib",
	"xgb_pipeline": "xgb_pipeline.joblib",
	"prophet_models": "Prophet_AllRegions_Models.joblib",
}

# Preload
_df_prophet_primary = load_csv(DATA_FILES["prophet_forecasts"])
if _df_prophet_primary is None or (_df_prophet_primary is not None and _df_prophet_primary.empty):
	prophet_df = load_csv(DATA_FILES["forecast_california"])  # fallback
else:
	prophet_df = _df_prophet_primary

state_ts_df = load_csv(DATA_FILES["state_ts"])
predictions_unseen_df = load_csv(DATA_FILES["predictions_unseen"])  # may or may not exist
unseen_input_df = load_csv(DATA_FILES["unseen_input"])  # optional
raw_listings_df = load_csv(DATA_FILES["raw_listings"])  # optional

# Load XGB pipeline/model with diagnostics
xgb_pipeline, _err_pipe = try_load_model(MODEL_FILES["xgb_pipeline"])
if xgb_pipeline is None:
	xgb_pipeline, _err_model = try_load_model(MODEL_FILES["xgb_model"])
else:
	_err_model = None
xgb_error = _err_pipe if _err_pipe else _err_model

# Validate that loaded object can predict
xgb_pipeline, _validate_err = ensure_predict_model(xgb_pipeline)
if _validate_err:
	xgb_error = f"{xgb_error or ''} {_validate_err}".strip()
	xgb_pipeline = None

# Prophet models not required at runtime; load best-effort
prophet_models, _ = try_load_model(MODEL_FILES["prophet_models"]) if os.path.exists(MODEL_FILES["prophet_models"]) else (None, None)

# ---------- UI ----------
st.set_page_config(page_title="Real Estate Forecast & Price Prediction", layout="wide")
st.title("Real Estate Forecast & Price Prediction")

# Info banner on available assets
with st.expander("Data & Models loaded", expanded=False):
	cols = st.columns(2)
	with cols[0]:
		st.write("Data files:")
		for key, path in DATA_FILES.items():
			st.write(f"- {key}: {'✅' if os.path.exists(path) else '❌'} ({path})")
	with cols[1]:
		st.write("Models:")
		for key, path in MODEL_FILES.items():
			ok = os.path.exists(path)
			status = '✅' if ok else '❌'
			st.write(f"- {key}: {status} ({path})")
		if xgb_error:
			st.caption(f"XGB load note: {xgb_error}")

forecast_tab, predict_tab = st.tabs(["Forecasting", "Prediction"])

# ---------- Forecasting Tab ----------
with forecast_tab:
	st.subheader("Home Price Forecasts (Prophet)")
	if prophet_df is None or prophet_df.empty:
		st.warning("No Prophet forecast CSV found. Please add Prophet_AllRegions_Forecasts_Renamed.csv or Forecast_California_House_Prices.csv")
	else:
		# Normalize expected columns
		col_map_candidates = [
			{"date": "Date", "yhat": "Predicted_Home_Price", "yhat_lower": "Lower_Confidence_Interval", "yhat_upper": "Upper_Confidence_Interval", "region": "RegionName"},
			{"date": "Date", "yhat": "Predicted_Home_Price", "yhat_lower": "Lower_Confidence_Interval", "yhat_upper": "Upper_Confidence_Interval", "region": None},
		]
		df_fore = prophet_df.copy()
		selected_map = None
		for m in col_map_candidates:
			if m["date"] in df_fore.columns and m["yhat"] in df_fore.columns:
				selected_map = m
				break
		if selected_map is None:
			st.error("Forecast CSV is missing required columns.")
		else:
			date_col = selected_map["date"]
			region_col = selected_map["region"]
			# Region filter if available
			if region_col and region_col in df_fore.columns:
				regions = list_unique(df_fore[region_col])
				default_region = regions[0] if regions else None
				region = st.selectbox("Region", options=regions, index=0 if default_region else None)
				if region:
					df_show = df_fore[df_fore[region_col] == region]
				else:
					df_show = df_fore
			else:
				df_show = df_fore
			# Date parse
			try:
				df_show[date_col] = pd.to_datetime(df_show[date_col])
			except Exception:
				pass
			# Date range filter
			if df_show[date_col].dtype == "datetime64[ns]" and not df_show.empty:
				min_d, max_d = df_show[date_col].min(), df_show[date_col].max()
				start_d, end_d = st.date_input("Date range", value=(min_d.date(), max_d.date()))
				try:
					mask = (df_show[date_col] >= pd.to_datetime(start_d)) & (df_show[date_col] <= pd.to_datetime(end_d))
					df_show = df_show.loc[mask]
				except Exception:
					pass
			# Plotly chart with confidence interval if available
			y_col = selected_map["yhat"]
			lo_col = selected_map["yhat_lower"]
			hi_col = selected_map["yhat_upper"]
			if lo_col in df_show.columns and hi_col in df_show.columns:
				fig = go.Figure()
				fig.add_trace(go.Scatter(x=df_show[date_col], y=df_show[hi_col], line=dict(width=0), showlegend=False, name="Upper"))
				fig.add_trace(go.Scatter(x=df_show[date_col], y=df_show[lo_col], fill="tonexty", fillcolor="rgba(99,110,250,0.2)", line=dict(width=0), showlegend=False, name="Lower"))
				fig.add_trace(go.Scatter(x=df_show[date_col], y=df_show[y_col], line=dict(color="#636EFA"), name="Predicted"))
				fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=360, yaxis_title="Price")
				st.plotly_chart(fig, use_container_width=True)
			else:
				st.line_chart(df_show.set_index(date_col)[y_col])
			with st.expander("Show forecast table"):
				st.dataframe(df_show[[c for c in [date_col, y_col, lo_col, hi_col, region_col] if c and c in df_show.columns]].reset_index(drop=True), use_container_width=True)
			# Summary stats
			st.caption("Summary of predicted prices")
			st.write(df_show[y_col].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))

	# Optional: show historical time series if present
	st.divider()
	st.subheader("Historical ZHVI (optional)")
	if state_ts_df is None or state_ts_df.empty:
		st.info("Add State_time_series_ZHVI_cleaned.csv to view historical series by region.")
	else:
		if "RegionName" in state_ts_df.columns and "Date" in state_ts_df.columns and "ZHVI_AllHomes" in state_ts_df.columns:
			regions_hist = list_unique(state_ts_df["RegionName"])
			region_hist = st.selectbox("Historical region", options=regions_hist)
			df_hist = state_ts_df[state_ts_df["RegionName"] == region_hist].copy()
			try:
				df_hist["Date"] = pd.to_datetime(df_hist["Date"])
			except Exception:
				pass
			st.line_chart(df_hist.set_index("Date")["ZHVI_AllHomes"]) 
		else:
			st.warning("Historical CSV missing expected columns.")

# ---------- Prediction Tab ----------
with predict_tab:
	st.subheader("Price Prediction (XGBoost Pipeline)")
	if xgb_pipeline is None:
		st.warning("Model not loaded or invalid. Upload a trained pipeline/model to enable predictions.")
		if xgb_error:
			st.caption(f"Load diagnostics: {xgb_error}")

	# Allow manual model upload
	with st.expander("Upload model (.joblib/.pkl)", expanded=False):
		model_file = st.file_uploader("Upload trained pipeline/model", type=["joblib", "pkl"], key="model_upload")
		if model_file is not None:
			try:
				import cloudpickle
				bytes_io = io.BytesIO(model_file.read())
				loaded_obj = cloudpickle.load(bytes_io)
				xgb_pipeline, msg = ensure_predict_model(loaded_obj)
				if xgb_pipeline is None:
					st.error(msg)
				else:
					st.success("Model loaded from uploaded file.")
			except Exception as e:
				st.error(f"Failed to load uploaded model: {e}")

	col1, col2 = st.columns([1, 1])

	# Batch prediction
	with col1:
		st.markdown("**Batch CSV prediction**")
		st.caption("Upload a CSV with feature columns matching the training pipeline.")
		uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
		if uploaded is not None:
			try:
				df_in = pd.read_csv(uploaded)
			except Exception as e:
				st.error(f"Failed to read CSV: {e}")
				df_in = None
			if df_in is not None:
				st.write("Input preview:")
				st.dataframe(df_in.head(), use_container_width=True)
				if xgb_pipeline is not None:
					try:
						preds = xgb_pipeline.predict(df_in)
						out = df_in.copy()
						out["Predicted_Price"] = preds
						st.success("Predictions generated")
						st.dataframe(out.head(), use_container_width=True)
						csv = out.to_csv(index=False).encode("utf-8")
						st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
					except Exception as e:
						st.error(f"Prediction failed: {e}")
				else:
					st.info("Model not loaded; showing schema only.")
					st.write({"columns": df_in.columns.tolist()})
		elif unseen_input_df is not None and not unseen_input_df.empty:
			st.caption("Using bundled unseen_real_estate_data.csv as sample input.")
			st.dataframe(unseen_input_df.head(), use_container_width=True)
			if xgb_pipeline is not None:
				try:
					preds = xgb_pipeline.predict(unseen_input_df)
					out = unseen_input_df.copy()
					out["Predicted_Price"] = preds
					st.dataframe(out.head(), use_container_width=True)
					csv = out.to_csv(index=False).encode("utf-8")
					st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
				except Exception as e:
					st.error(f"Prediction failed: {e}")

	# Single prediction form (refined schema)
	with col2:
		st.markdown("**Single record prediction**")
		st.caption("Enter features to predict a single price. Adjust fields as per your pipeline.")

		# Prefer sensible numeric features if present
		feature_candidates = ["Total_Area", "Bedrooms", "Baths", "Balcony", "Price_per_SQFT"]
		available_numeric = [c for c in feature_candidates if unseen_input_df is not None and c in unseen_input_df.columns]
		if not available_numeric:
			if unseen_input_df is not None and not unseen_input_df.empty:
				available_numeric = [c for c in unseen_input_df.columns if unseen_input_df[c].dtype != "object"]
			elif predictions_unseen_df is not None and not predictions_unseen_df.empty:
				available_numeric = [c for c in predictions_unseen_df.columns if predictions_unseen_df[c].dtype != "object" and c != "Predicted_Price"]

		inputs = {}
		for col in available_numeric[:10]:
			val = st.number_input(f"{col}", value=float(0))
			inputs[col] = val

		# Add some common fields seen in provided CSVs
		n_categorical = {}
		locations = []
		if unseen_input_df is not None and "Location" in unseen_input_df.columns:
			locations = list_unique(unseen_input_df["Location"])
		elif raw_listings_df is not None and "Location" in raw_listings_df.columns:
			locations = list_unique(raw_listings_df["Location"])
		if locations:
			loc_val = st.selectbox("Location", options=locations)
			n_categorical["Location"] = loc_val

		if st.button("Predict price", type="primary"):
			if xgb_pipeline is None:
				st.error("Model not loaded or invalid. Upload a trained pipeline/model.")
			else:
				# Build single-row DataFrame using whatever inputs are provided
				row = {}
				row.update(inputs)
				row.update(n_categorical)
				X = pd.DataFrame([row])
				try:
					pred = xgb_pipeline.predict(X)[0]
					st.success(f"Predicted price: {pred}")
				except Exception as e:
					st.error(f"Prediction failed: {e}")
