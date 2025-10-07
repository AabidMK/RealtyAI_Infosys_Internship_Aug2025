import React, { useState, useEffect, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area } from 'recharts';
import { FaChartLine, FaGlobeAmericas, FaCalendarAlt, FaChartBar, FaInfoCircle, FaExchangeAlt, FaDatabase, FaTimes, FaPlus } from 'react-icons/fa';

const PriceForecasting = () => {
  const [selectedRegions, setSelectedRegions] = useState([]);
  const [forecastDays, setForecastDays] = useState(365);
  const [analysisType, setAnalysisType] = useState('single');
  const [forecastData, setForecastData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedRegionForStats, setSelectedRegionForStats] = useState('');
  const [newRegion, setNewRegion] = useState('');

  // Actual regions from the Prophet models
  const availableRegions = useMemo(() => [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 
    'DistrictofColumbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 
    'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 
    'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'NewHampshire', 'NewJersey', 'NewMexico', 
    'NewYork', 'NorthCarolina', 'NorthDakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'RhodeIsland', 
    'SouthCarolina', 'SouthDakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 
    'WestVirginia', 'Wisconsin', 'Wyoming'
  ], []);

  // Generate mock forecast data that simulates Prophet model output
  const generateMockForecastData = (region, days, includeHistory = true) => {
    const data = [];
    const startDate = new Date();
    startDate.setFullYear(startDate.getFullYear() - 2); // Start 2 years ago for historical data
    
    let currentValue = 200000 + Math.random() * 300000; // Base ZHVI value
    const trend = (Math.random() - 0.5) * 0.0005; // Daily trend
    const volatility = 0.002; // Daily volatility
    
    const totalDays = includeHistory ? 730 + days : days; // 2 years history + forecast period
    const historyCutoff = includeHistory ? 730 : 0; // Point where forecast begins
    
    for (let i = 0; i < totalDays; i++) {
      const date = new Date(startDate);
      date.setDate(startDate.getDate() + i);
      
      // Apply trend and some randomness
      currentValue = currentValue * (1 + trend + (Math.random() - 0.5) * volatility);
      
      // Add some seasonality
      const seasonal = Math.sin(i * 0.01) * 5000;
      
      // Add some noise
      const noise = (Math.random() - 0.5) * 10000;
      
      const value = Math.max(50000, currentValue + seasonal + noise); // Minimum value
      
      // For forecast period, add confidence intervals
      let yhat_lower = null;
      let yhat_upper = null;
      let trend_component = null;
      
      if (i >= historyCutoff) {
        // Forecast period - add confidence intervals
        const uncertainty = value * 0.1; // 10% uncertainty
        yhat_lower = Math.max(0, value - uncertainty - Math.random() * 20000);
        yhat_upper = value + uncertainty + Math.random() * 20000;
        trend_component = value + (Math.random() - 0.5) * 10000;
      }
      
      data.push({
        ds: date.toISOString().split('T')[0],
        yhat: value,
        yhat_lower: yhat_lower,
        yhat_upper: yhat_upper,
        trend: trend_component,
        isForecast: i >= historyCutoff
      });
    }
    
    return data;
  };

  // Generate mock statistics data
  const generateMockStatistics = (region) => {
    return {
      total_records: Math.floor(1000 + Math.random() * 2000),
      date_range: '2010-01-01 to 2023-12-31',
      latest_zhvi: 250000 + Math.random() * 200000,
      zhvi_mean: 200000 + Math.random() * 150000,
      zhvi_std: 30000 + Math.random() * 20000,
      zhvi_min: 100000 + Math.random() * 50000,
      zhvi_max: 400000 + Math.random() * 300000
    };
  };

  const handleForecast = async () => {
    if (selectedRegions.length === 0) {
      setError('Please select at least one region');
      return;
    }
    
    setIsLoading(true);
    setError('');
    
    try {
      // In a real application, this would call your backend API
      // For now, we'll simulate the forecasting
      setTimeout(() => {
        if (analysisType === 'single' && selectedRegions.length > 0) {
          // Pass false for includeHistory since we removed the option
          const data = generateMockForecastData(selectedRegions[0], forecastDays, false);
          setForecastData({
            type: 'single',
            region: selectedRegions[0],
            data: data
          });
        } else if (analysisType === 'comparison' && selectedRegions.length >= 2) {
          const comparisonData = selectedRegions.slice(0, 5).map(region => ({
            region: region,
            data: generateMockForecastData(region, forecastDays, false) // No history for comparison
          }));
          setForecastData({
            type: 'comparison',
            data: comparisonData
          });
        } else if (analysisType === 'statistics') {
          // Mock statistics data
          const statsData = selectedRegions.map(region => ({
            region: region,
            ...generateMockStatistics(region)
          }));
          setForecastData({
            type: 'statistics',
            data: statsData
          });
        }
        setIsLoading(false);
      }, 2000);
    } catch (err) {
      setError('Error generating forecast: ' + err.message);
      setIsLoading(false);
    }
  };

  // Handle region selection for single region
  const handleSingleRegionChange = (e) => {
    const region = e.target.value;
    if (region) {
      setSelectedRegions([region]);
    } else {
      setSelectedRegions([]);
    }
  };

  // Add a new region to the selection
  const addRegion = () => {
    if (newRegion && !selectedRegions.includes(newRegion)) {
      setSelectedRegions([...selectedRegions, newRegion]);
      setNewRegion('');
    }
  };

  // Remove a region from the selection
  const removeRegion = (regionToRemove) => {
    setSelectedRegions(selectedRegions.filter(region => region !== regionToRemove));
  };

  // Initialize with first region selected
  useEffect(() => {
    if (selectedRegions.length === 0 && availableRegions.length > 0) {
      setSelectedRegions([availableRegions[0]]);
      setSelectedRegionForStats(availableRegions[0]);
    }
  }, [availableRegions, selectedRegions.length]);

  // Get forecast summary for single region
  const getForecastSummary = (forecastData) => {
    if (!forecastData || forecastData.length === 0) return null;
    
    // Find the point where forecast begins
    const forecastStartIndex = forecastData.findIndex(d => d.isForecast);
    if (forecastStartIndex === -1) return null;
    
    const forecastStart = forecastData[forecastStartIndex];
    const forecastEnd = forecastData[forecastData.length - 1];
    
    if (!forecastStart || !forecastEnd) return null;
    
    const currentValue = forecastStart.yhat;
    const forecastEndValue = forecastEnd.yhat;
    const totalChange = forecastEndValue - currentValue;
    const percentChange = (totalChange / currentValue) * 100;
    
    return {
      current_value: currentValue,
      forecast_end_value: forecastEndValue,
      forecast_start_value: currentValue,
      total_change: totalChange,
      percent_change: percentChange,
      max_forecast: Math.max(...forecastData.slice(forecastStartIndex).map(d => d.yhat)),
      min_forecast: Math.min(...forecastData.slice(forecastStartIndex).map(d => d.yhat)),
      forecast_periods: forecastData.length - forecastStartIndex
    };
  };

  // Render single region forecast
  const renderSingleForecast = () => {
    if (!forecastData || forecastData.type !== 'single') return null;
    
    const summary = getForecastSummary(forecastData.data);
    
    // Separate historical and forecast data
    const historicalData = forecastData.data.filter(d => !d.isForecast);
    const forecastOnlyData = forecastData.data.filter(d => d.isForecast);
    
    return (
      <div className="single-forecast animated-result">
        <h3><FaChartLine /> Forecast for {forecastData.region}</h3>
        <div className="chart-container animated-card">
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={forecastData.data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
              <XAxis 
                dataKey="ds" 
                tickFormatter={(value) => {
                  const date = new Date(value);
                  return `${date.getMonth()+1}/${date.getFullYear()}`;
                }}
                stroke="#64748b"
              />
              <YAxis 
                tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`}
                stroke="#64748b"
              />
              <Tooltip 
                formatter={(value, name) => {
                  if (name === 'yhat') return [`$${value.toLocaleString()}`, 'ZHVI'];
                  if (name === 'yhat_lower' || name === 'yhat_upper') return [`$${value.toLocaleString()}`, name === 'yhat_lower' ? 'Lower Bound' : 'Upper Bound'];
                  return [value, name];
                }}
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
                contentStyle={{ 
                  backgroundColor: 'rgba(255, 255, 255, 0.95)', 
                  border: '1px solid #cbd5e1', 
                  borderRadius: '10px',
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Legend />
              {historicalData.length > 0 && (
                <Line 
                  type="monotone" 
                  dataKey="yhat" 
                  name="Historical Trend" 
                  stroke="#0ea5e9" 
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 6 }}
                />
              )}
              {forecastOnlyData.length > 0 && (
                <>
                  <Line 
                    type="monotone" 
                    dataKey="yhat" 
                    name="Forecast" 
                    stroke="#0891b2" 
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    activeDot={{ r: 6 }}
                  />
                  {forecastOnlyData[0] && forecastOnlyData[0].yhat_lower && (
                    <Area
                      data={forecastOnlyData}
                      type="monotone"
                      dataKey="yhat_lower"
                      fill="none"
                      stroke="none"
                    />
                  )}
                  {forecastOnlyData[0] && forecastOnlyData[0].yhat_upper && (
                    <Area
                      data={forecastOnlyData}
                      type="monotone"
                      dataKey="yhat_upper"
                      fill="#0ea5e9"
                      fillOpacity={0.15}
                      stroke="none"
                      baseValue={(data) => data.yhat_lower}
                    />
                  )}
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        {summary && (
          <div className="forecast-summary animated-card">
            <h4><FaInfoCircle /> Forecast Summary</h4>
            <div className="summary-cards">
              <div className="summary-card">
                <strong>Current ZHVI</strong>
                <span>${summary.current_value.toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
              </div>
              <div className="summary-card">
                <strong>Forecast End Value</strong>
                <span>${summary.forecast_end_value.toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
              </div>
              <div className="summary-card">
                <strong>Expected Change</strong>
                <span className={summary.percent_change >= 0 ? 'positive' : 'negative'}>
                  {summary.percent_change.toFixed(1)}%
                </span>
              </div>
              <div className="summary-card">
                <strong>Forecast Period</strong>
                <span>{forecastDays} days</span>
              </div>
            </div>
            
            <div className="insights animated-card">
              <h4><FaInfoCircle /> Key Insights</h4>
              {summary.percent_change > 5 ? (
                <div className="insight positive">
                  <strong>📈 Positive Trend</strong>: {forecastData.region} shows strong growth potential with {summary.percent_change.toFixed(1)}% expected increase.
                </div>
              ) : summary.percent_change < -5 ? (
                <div className="insight negative">
                  <strong>📉 Declining Trend</strong>: {forecastData.region} shows declining values with {summary.percent_change.toFixed(1)}% expected decrease.
                </div>
              ) : (
                <div className="insight neutral">
                  <strong>📊 Stable Market</strong>: {forecastData.region} shows relatively stable values with {summary.percent_change.toFixed(1)}% expected change.
                </div>
              )}
              <div className="insight info">
                <strong>Forecast Range</strong>: ${summary.min_forecast.toLocaleString(undefined, {maximumFractionDigits: 0})} - ${summary.max_forecast.toLocaleString(undefined, {maximumFractionDigits: 0})}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Render multi-region comparison
  const renderComparisonForecast = () => {
    if (!forecastData || forecastData.type !== 'comparison') return null;
    
    return (
      <div className="comparison-forecast animated-result">
        <h3><FaExchangeAlt /> Multi-Region Forecast Comparison</h3>
        <div className="chart-container animated-card">
          <ResponsiveContainer width="100%" height={400}>
            <LineChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" />
              <XAxis 
                tickFormatter={(value) => {
                  const date = new Date(value);
                  return `${date.getMonth()+1}/${date.getFullYear()}`;
                }}
                stroke="#64748b"
              />
              <YAxis 
                tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`}
                stroke="#64748b"
              />
              <Tooltip 
                formatter={(value) => [`$${value.toLocaleString()}`, 'ZHVI']}
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
                contentStyle={{ 
                  backgroundColor: 'rgba(255, 255, 255, 0.95)', 
                  border: '1px solid #cbd5e1', 
                  borderRadius: '10px',
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Legend />
              {forecastData.data.map((regionData, index) => {
                const colors = ['#0ea5e9', '#0891b2', '#0284c7', '#0369a1', '#0c4a6e'];
                return (
                  <Line 
                    key={regionData.region}
                    data={regionData.data}
                    type="monotone" 
                    dataKey="yhat" 
                    name={regionData.region} 
                    stroke={colors[index % colors.length]} 
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 6 }}
                  />
                );
              })}
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        <div className="comparison-table animated-card">
          <h4><FaChartBar /> Forecast Summary by Region</h4>
          <table>
            <thead>
              <tr>
                <th>Region</th>
                <th>Current ZHVI</th>
                <th>Forecast End</th>
                <th>Change %</th>
                <th>Change $</th>
              </tr>
            </thead>
            <tbody>
              {forecastData.data.map(regionData => {
                if (regionData.data.length === 0) return null;
                
                const firstValue = regionData.data[0].yhat;
                const lastValue = regionData.data[regionData.data.length - 1].yhat;
                const changePercent = ((lastValue - firstValue) / firstValue) * 100;
                const changeDollar = lastValue - firstValue;
                
                return (
                  <tr key={regionData.region}>
                    <td>{regionData.region}</td>
                    <td>${firstValue.toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
                    <td>${lastValue.toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
                    <td className={changePercent >= 0 ? 'positive' : 'negative'}>
                      {changePercent.toFixed(1)}%
                    </td>
                    <td className={changeDollar >= 0 ? 'positive' : 'negative'}>
                      ${changeDollar.toLocaleString(undefined, {maximumFractionDigits: 0})}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  // Render region statistics
  const renderStatistics = () => {
    if (!forecastData || forecastData.type !== 'statistics') return null;
    
    return (
      <div className="statistics-forecast animated-result">
        <h3><FaDatabase /> Region Statistics & Analysis</h3>
        
        <div className="region-selector-stats animated-card">
          <label htmlFor="regionSelect"><FaGlobeAmericas /> Select Region for Detailed Analysis:</label>
          <select 
            id="regionSelect"
            value={selectedRegionForStats}
            onChange={(e) => setSelectedRegionForStats(e.target.value)}
          >
            {selectedRegions.map(region => (
              <option key={region} value={region}>{region}</option>
            ))}
          </select>
        </div>
        
        {selectedRegionForStats && (
          <div className="region-detail animated-card">
            {forecastData.data
              .filter(item => item.region === selectedRegionForStats)
              .map(item => (
                <div key={item.region}>
                  <h4><FaInfoCircle /> Statistics for {item.region}</h4>
                  <div className="summary-cards">
                    <div className="summary-card">
                      <strong>Total Records</strong>
                      <span>{item.total_records.toLocaleString()}</span>
                    </div>
                    <div className="summary-card">
                      <strong>Latest ZHVI</strong>
                      <span>${item.latest_zhvi.toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
                    </div>
                    <div className="summary-card">
                      <strong>Average ZHVI</strong>
                      <span>${item.zhvi_mean.toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
                    </div>
                    <div className="summary-card">
                      <strong>ZHVI Range</strong>
                      <span>${item.zhvi_min.toLocaleString(undefined, {maximumFractionDigits: 0})} - ${item.zhvi_max.toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
                    </div>
                  </div>
                  <div className="info-message">
                    <strong>Data Period</strong>: {item.date_range}
                  </div>
                </div>
              ))}
          </div>
        )}
        
        {selectedRegions.length > 1 && (
          <div className="comparison-table animated-card">
            <h4><FaExchangeAlt /> Quick Region Comparison</h4>
            <table>
              <thead>
                <tr>
                  <th>Region</th>
                  <th>Latest ZHVI</th>
                  <th>Average ZHVI</th>
                  <th>Records</th>
                </tr>
              </thead>
              <tbody>
                {forecastData.data
                  .filter(item => selectedRegions.includes(item.region))
                  .map(item => (
                    <tr key={item.region}>
                      <td>{item.region}</td>
                      <td>${item.latest_zhvi.toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
                      <td>${item.zhvi_mean.toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
                      <td>{item.total_records.toLocaleString()}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="price-forecasting">
      <h2><FaChartLine /> Price Forecasting</h2>
      
      <div className="form-description">
        <p>Select regions and configure parameters to generate price forecasts</p>
      </div>
      
      <div className="forecasting-controls animated-card">
        <div className="control-group">
          {/* Removed the "Forecast Configuration" heading to save space */}
          
          <div className="form-group">
            <label><FaGlobeAmericas /> Select Regions</label>
            {analysisType === 'single' ? (
              <select
                value={selectedRegions[0] || ''}
                onChange={handleSingleRegionChange}
                className="animated-input"
              >
                <option value="">Select a region</option>
                {availableRegions.map(region => (
                  <option key={region} value={region}>{region}</option>
                ))}
              </select>
            ) : (
              <div className="multi-region-selector">
                <div className="region-input-container">
                  <select
                    value={newRegion}
                    onChange={(e) => setNewRegion(e.target.value)}
                    className="region-select-input animated-input"
                  >
                    <option value="">Select a region to add</option>
                    {availableRegions
                      .filter(region => !selectedRegions.includes(region))
                      .map(region => (
                        <option key={region} value={region}>{region}</option>
                      ))}
                  </select>
                  <button 
                    onClick={addRegion} 
                    className="add-region-button animated-button"
                    disabled={!newRegion}
                  >
                    <FaPlus />
                  </button>
                </div>
                
                <div className="selected-regions">
                  {selectedRegions.map(region => (
                    <div key={region} className="region-tag animated-card">
                      <span className="region-name">{region}</span>
                      <button 
                        onClick={() => removeRegion(region)} 
                        className="remove-region-button"
                      >
                        <FaTimes />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          <div className="form-group">
            <label><FaCalendarAlt /> Forecast Period (Days)</label>
            <input
              type="range"
              min="30"
              max="1095"
              value={forecastDays}
              onChange={(e) => setForecastDays(parseInt(e.target.value))}
              className="animated-input forecast-slider"
            />
            <div className="slider-value-container">
              <span className="slider-value">{forecastDays} days</span>
            </div>
          </div>

          <div className="form-group">
            <label><FaChartBar /> Analysis Type</label>
            <select
              value={analysisType}
              onChange={(e) => setAnalysisType(e.target.value)}
              className="animated-input"
            >
              <option value="single">Single Region Forecast</option>
              <option value="comparison">Multi-Region Comparison</option>
              <option value="statistics">Region Statistics</option>
            </select>
          </div>

          <div className="form-actions">
            <button 
              className="forecast-button animated-button"
              onClick={handleForecast}
              disabled={isLoading || selectedRegions.length === 0}
            >
              {isLoading ? 'Generating Forecast...' : <><FaChartLine /> Generate Forecast</>}
            </button>
          </div>
        </div>
      </div>
      
      {error && (
        <div className="error-message animated-error">
          {error}
        </div>
      )}
      
      <div className="forecast-results">
        {forecastData && (
          <>
            {forecastData.type === 'single' && renderSingleForecast()}
            {forecastData.type === 'comparison' && renderComparisonForecast()}
            {forecastData.type === 'statistics' && renderStatistics()}
          </>
        )}
      </div>
    </div>
  );
};

export default PriceForecasting;