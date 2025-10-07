import React, { useState } from 'react';
import { FaHome, FaBath, FaRulerCombined, FaMapMarkerAlt, FaBuilding, FaRupeeSign, FaMoneyBillWave } from 'react-icons/fa';

const PricePrediction = () => {
  const [propertyTitle, setPropertyTitle] = useState('');
  const [location, setLocation] = useState('');
  const [totalArea, setTotalArea] = useState(1200);
  const [baths, setBaths] = useState(2);
  const [balcony, setBalcony] = useState('Yes');
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Extract BHK from property title
  const extractBHK = (title) => {
    const match = title.match(/(\d+)\s*BHK/i);
    return match ? parseInt(match[1]) : null;
  };

  // Extract property type from title
  const extractPropertyType = (title) => {
    const lowerTitle = title.toLowerCase();
    if (lowerTitle.includes('independent house')) return 'Independent House';
    if (lowerTitle.includes('flat')) return 'Flat';
    if (lowerTitle.includes('villa')) return 'Villa';
    return 'Other';
  };

  // Extract city from location
  const extractCity = (location) => {
    const parts = location.split(',');
    return parts[parts.length - 1].trim();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setPredictionResult(null);
    
    // Validate required fields
    if (!propertyTitle || !location) {
      setError('Please fill in all required fields (Property Title and Location)');
      return;
    }
    
    setIsLoading(true);
    
    try {
      // In a real application, this would call your backend API
      // For now, we'll simulate a prediction
      setTimeout(() => {
        const bhk = extractBHK(propertyTitle) || 0;
        const propertyType = extractPropertyType(propertyTitle);
        const city = extractCity(location);
        
        // More realistic simulation based on the features used in the real model
        let basePrice = 1000 * totalArea; // Base price per sq ft
        const bhkFactor = 1 + (bhk * 0.2); // BHK premium
        const balconyFactor = balcony === 'Yes' ? 1.05 : 1; // Balcony premium
        const bathroomFactor = 1 + (baths * 0.05); // Bathroom premium
        const cityPremium = city === 'Mumbai' ? 1.5 : city === 'Delhi' ? 1.3 : city === 'Chennai' ? 1.2 : 1; // City premium
        
        const predictedPrice = (basePrice * bhkFactor * balconyFactor * bathroomFactor * cityPremium) / 100000;
        
        setPredictionResult({
          price: predictedPrice,
          priceInRupees: predictedPrice * 100000,
          pricePerSqFt: predictedPrice * 100000 / totalArea,
          bhk: bhk,
          propertyType: propertyType,
          city: city
        });
        setIsLoading(false);
      }, 1500);
    } catch (err) {
      setError('Error making prediction: ' + err.message);
      setIsLoading(false);
    }
  };

  const bhk = extractBHK(propertyTitle);
  const propertyType = extractPropertyType(propertyTitle);

  return (
    <div className="price-prediction">
      <h2><FaHome /> House Price Prediction</h2>
      
      <div className="form-description">
        <p>Enter the property details below to get the predicted price</p>
      </div>
      
      <form onSubmit={handleSubmit} className="prediction-form">
        <div className="form-row">
          <div className="form-column">
            <h3><FaBuilding /> Property Details</h3>
            
            <div className="form-group">
              <label htmlFor="propertyTitle"><FaHome /> Property Title *</label>
              <input
                type="text"
                id="propertyTitle"
                value={propertyTitle}
                onChange={(e) => setPropertyTitle(e.target.value)}
                placeholder="e.g., 2 BHK Flat in Andheri"
                className="animated-input"
              />
              <small>Enter the property type including BHK information</small>
            </div>
            
            <div className="form-group">
              <label htmlFor="location"><FaMapMarkerAlt /> Location *</label>
              <input
                type="text"
                id="location"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                placeholder="e.g., Andheri, Mumbai"
                className="animated-input"
              />
              <small>Enter location in format: Area, City</small>
            </div>
            
            <div className="form-group">
              <label htmlFor="totalArea"><FaRulerCombined /> Total Area (sq ft)</label>
              <input
                type="number"
                id="totalArea"
                min="100"
                max="10000"
                value={totalArea}
                onChange={(e) => setTotalArea(parseInt(e.target.value) || 0)}
                className="animated-input"
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="baths"><FaBath /> Number of Bathrooms</label>
              <input
                type="number"
                id="baths"
                min="1"
                max="10"
                value={baths}
                onChange={(e) => setBaths(parseInt(e.target.value) || 0)}
                className="animated-input"
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="balcony">Balcony</label>
              <select
                id="balcony"
                value={balcony}
                onChange={(e) => setBalcony(e.target.value)}
                className="animated-input"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
            
            {propertyTitle && (
              <div className="extracted-info animated-card">
                <h3>📋 Extracted Information</h3>
                <div className="info-cards">
                  <div className="info-card">
                    <strong>PropertyParams Type:</strong> {propertyType}
                  </div>
                  {bhk ? (
                    <div className="info-card success">
                      <strong>BHK:</strong> {bhk} (extracted from title)
                    </div>
                  ) : (
                    <div className="info-card warning">
                      <strong>BHK:</strong> Not found in title
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
        
        <div className="form-actions">
          <button 
            type="submit" 
            className="predict-button animated-button"
            disabled={isLoading}
          >
            {isLoading ? 'Predicting...' : <><FaMoneyBillWave /> Predict Price</>}
          </button>
        </div>
      </form>
      
      {error && (
        <div className="error-message animated-error">
          {error}
        </div>
      )}
      
      {predictionResult && (
        <div className="prediction-result animated-result">
          <h3>🎯 Predicted Price</h3>
          <div className="price-display">
            <span className="price-value"><FaRupeeSign />{predictionResult.price.toFixed(2)} Lakhs</span>
            <span className="price-rupees">(<FaRupeeSign />{predictionResult.priceInRupees.toLocaleString()})</span>
          </div>
          
          <div className="prediction-details">
            <div className="detail-card">
              <strong>Price per sq ft</strong>
              <span><FaRupeeSign />{predictionResult.pricePerSqFt.toFixed(0)}</span>
            </div>
            <div className="detail-card">
              <strong>BHK</strong>
              <span>{predictionResult.bhk}</span>
            </div>
            <div className="detail-card">
              <strong>Bathrooms</strong>
              <span>{baths}</span>
            </div>
            <div className="detail-card">
              <strong>PropertyParams Type</strong>
              <span>{predictionResult.propertyType}</span>
            </div>
          </div>
          
          <div className="extracted-details">
            <h4>📋 Extracted Information</h4>
            <div className="info-cards">
              <div className="info-card">
                <strong>PropertyParams Type:</strong> {predictionResult.propertyType}
              </div>
              {predictionResult.bhk ? (
                <div className="info-card success">
                  <strong>BHK:</strong> {predictionResult.bhk} (extracted from title)
                </div>
              ) : (
                <div className="info-card warning">
                  <strong>BHK:</strong> Not found in title
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PricePrediction;