import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class HousePriceForecaster:
    """House Price Forecasting using Prophet models"""
    
    def __init__(self, models_path="prophet_models_all_regions.joblib"):
        """Initialize the forecaster with trained Prophet models"""
        try:
            self.prophet_models = joblib.load(models_path)
            self.available_regions = list(self.prophet_models.keys())
            print(f"Loaded {len(self.available_regions)} Prophet models")
        except FileNotFoundError:
            print(f"Models file not found at {models_path}")
            self.prophet_models = {}
            self.available_regions = []
    
    def get_available_regions(self):
        """Get list of available regions for forecasting"""
        return self.available_regions
    
    def forecast_region(self, region_name, periods=365, include_history=True):
        """
        Generate forecast for a specific region
        
        Args:
            region_name (str): Name of the region
            periods (int): Number of days to forecast
            include_history (bool): Whether to include historical data
            
        Returns:
            dict: Contains forecast dataframe and model components
        """
        if region_name not in self.prophet_models:
            raise ValueError(f"Region '{region_name}' not found in trained models")
        
        model = self.prophet_models[region_name]
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq='D')
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Filter to only future dates if requested
        if not include_history:
            forecast = forecast[forecast['ds'] > forecast['ds'].max() - pd.Timedelta(days=periods)]
        
        return {
            'forecast': forecast,
            'model': model,
            'region': region_name
        }
    
    def forecast_multiple_regions(self, region_names, periods=365, include_history=True):
        """
        Generate forecasts for multiple regions
        
        Args:
            region_names (list): List of region names
            periods (int): Number of days to forecast
            include_history (bool): Whether to include historical data
            
        Returns:
            dict: Dictionary with region names as keys and forecast data as values
        """
        results = {}
        
        for region in region_names:
            if region in self.available_regions:
                try:
                    results[region] = self.forecast_region(region, periods, include_history)
                except Exception as e:
                    print(f"Error forecasting {region}: {str(e)}")
            else:
                print(f"Region '{region}' not available")
        
        return results
    
    def create_forecast_plot(self, forecast_data, title_suffix=""):
        """
        Create interactive forecast plot using Plotly
        
        Args:
            forecast_data (dict): Forecast data from forecast_region method
            title_suffix (str): Additional text for plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive plot
        """
        forecast = forecast_data['forecast']
        region = forecast_data['region']
        
        fig = go.Figure()
        
        # Add historical data (if available)
        historical = forecast[forecast['ds'] <= forecast['ds'].max() - pd.Timedelta(days=365)]
        if not historical.empty:
            fig.add_trace(go.Scatter(
                x=historical['ds'],
                y=historical['yhat'],
                mode='lines',
                name='Historical Trend',
                line=dict(color='blue', width=2)
            ))
        
        # Add forecast
        future_forecast = forecast[forecast['ds'] > forecast['ds'].max() - pd.Timedelta(days=365)]
        if not future_forecast.empty:
            fig.add_trace(go.Scatter(
                x=future_forecast['ds'],
                y=future_forecast['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=future_forecast['ds'],
                y=future_forecast['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=future_forecast['ds'],
                y=future_forecast['yhat_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                name='Confidence Interval'
            ))
        
        fig.update_layout(
            title=f'ZHVI Forecast for {region} {title_suffix}',
            xaxis_title='Date',
            yaxis_title='ZHVI (Home Value Index)',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def create_components_plot(self, forecast_data):
        """
        Create components plot showing trend and seasonality
        
        Args:
            forecast_data (dict): Forecast data from forecast_region method
            
        Returns:
            plotly.graph_objects.Figure: Components plot
        """
        forecast = forecast_data['forecast']
        region = forecast_data['region']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Trend Component', 'Yearly Seasonality'),
            vertical_spacing=0.1
        )
        
        # Trend component
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['trend'],
                mode='lines',
                name='Trend',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Yearly seasonality (if available)
        if 'yearly' in forecast.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yearly'],
                    mode='lines',
                    name='Yearly Seasonality',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=f'Trend and Seasonality Components for {region}',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def get_forecast_summary(self, forecast_data):
        """
        Get summary statistics for the forecast
        
        Args:
            forecast_data (dict): Forecast data from forecast_region method
            
        Returns:
            dict: Summary statistics
        """
        forecast = forecast_data['forecast']
        
        # Get future forecast only
        future_forecast = forecast[forecast['ds'] > forecast['ds'].max() - pd.Timedelta(days=365)]
        
        if future_forecast.empty:
            return {}
        
        summary = {
            'current_value': forecast['yhat'].iloc[-366] if len(forecast) > 365 else forecast['yhat'].iloc[0],
            'forecast_end_value': future_forecast['yhat'].iloc[-1],
            'forecast_start_value': future_forecast['yhat'].iloc[0],
            'total_change': future_forecast['yhat'].iloc[-1] - future_forecast['yhat'].iloc[0],
            'percent_change': ((future_forecast['yhat'].iloc[-1] - future_forecast['yhat'].iloc[0]) / future_forecast['yhat'].iloc[0]) * 100,
            'max_forecast': future_forecast['yhat'].max(),
            'min_forecast': future_forecast['yhat'].min(),
            'forecast_periods': len(future_forecast)
        }
        
        return summary
    
    def compare_regions_forecast(self, region_names, periods=365):
        """
        Compare forecasts across multiple regions
        
        Args:
            region_names (list): List of region names to compare
            periods (int): Number of days to forecast
            
        Returns:
            plotly.graph_objects.Figure: Comparison plot
        """
        fig = go.Figure()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, region in enumerate(region_names):
            if region in self.available_regions:
                try:
                    forecast_data = self.forecast_region(region, periods, include_history=False)
                    forecast = forecast_data['forecast']
                    
                    color = colors[i % len(colors)]
                    
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat'],
                        mode='lines',
                        name=region,
                        line=dict(color=color, width=2)
                    ))
                    
                    # Add confidence intervals
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=f'rgba({color},0.2)',
                        showlegend=False
                    ))
                    
                except Exception as e:
                    print(f"Error creating forecast for {region}: {str(e)}")
        
        fig.update_layout(
            title='ZHVI Forecast Comparison Across Regions',
            xaxis_title='Date',
            yaxis_title='ZHVI (Home Value Index)',
            hovermode='x unified'
        )
        
        return fig

def load_forecasting_data():
    """Load the time series data for analysis"""
    try:
        df = pd.read_csv("Data/State_time_series.csv")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).reset_index(drop=True)
        return df
    except FileNotFoundError:
        print("Forecasting data file not found")
        return None

def get_region_statistics(df, region_name):
    """Get basic statistics for a region"""
    if df is None:
        return {}
    
    region_data = df[df['RegionName'] == region_name]
    
    if region_data.empty:
        return {}
    
    stats = {
        'total_records': len(region_data),
        'date_range': f"{region_data['Date'].min().strftime('%Y-%m-%d')} to {region_data['Date'].max().strftime('%Y-%m-%d')}",
        'latest_zhvi': region_data['ZHVI_AllHomes'].iloc[-1] if not region_data['ZHVI_AllHomes'].isna().all() else None,
        'zhvi_mean': region_data['ZHVI_AllHomes'].mean(),
        'zhvi_std': region_data['ZHVI_AllHomes'].std(),
        'zhvi_min': region_data['ZHVI_AllHomes'].min(),
        'zhvi_max': region_data['ZHVI_AllHomes'].max()
    }
    
    return stats
