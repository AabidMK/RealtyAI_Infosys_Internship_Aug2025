import pandas as pd
import numpy as np
import joblib

# Load pipeline and location encoding dict
pipeline, location_target = joblib.load("pipeline.pkl")

# Sample unseen data
data = pd.DataFrame({
    'Property Title': ['2 BHK Flat in Andheri', '4 BHK Villa in Jubilee Hills'],
    'Location': ['Andheri, Mumbai', 'Jubilee Hills, Hyderabad'],
    'Total_Area': [1200, 2500],
    'Baths': [2, 4],
    'Balcony': ['Yes', 'No']
})

# BHK
import re
data['BHK'] = data['Property Title'].apply(lambda t: int(re.search(r'(\d+)\s*BHK', str(t), re.IGNORECASE).group(1)) 
                                           if re.search(r'(\d+)\s*BHK', str(t), re.IGNORECASE) else 0)

# Balcony
data['Balcony'] = data['Balcony'].map({'Yes':1,'No':0}).fillna(0)

# Property_Type
def prop_type(t):
    t = str(t).lower()
    if "independent house" in t: return "Independent House"
    elif "flat" in t: return "Flat"
    elif "villa" in t: return "Villa"
    else: return "Other"
data['Property_Type'] = data['Property Title'].apply(prop_type)

# City
data['City'] = data['Location'].apply(lambda x: str(x).split(",")[-1].strip())

# Location_Encoded
data['Location_Encoded'] = data['Location'].map(location_target)
data['Location_Encoded'] = data['Location_Encoded'].fillna(np.mean(list(location_target.values())))

# Log_Total_Area & interaction features
data['Log_Total_Area'] = np.log(data['Total_Area'] + 1)
data['BHK_x_Total_Area'] = data['BHK'] * data['Total_Area']
data['BHK_x_Baths'] = data['BHK'] * data['Baths']
data['Balcony_x_BHK'] = data['Balcony'] * data['BHK']
data['Log_Total_Area_x_BHK'] = data['Log_Total_Area'] * data['BHK']
data['Log_Total_Area_squared'] = data['Log_Total_Area'] ** 2

# Feature columns
feature_cols = ['Log_Total_Area','Baths','Balcony','BHK','Location_Encoded',
                'BHK_x_Total_Area','BHK_x_Baths','Balcony_x_BHK','Log_Total_Area_x_BHK','Log_Total_Area_squared',
                'City','Property_Type']

X_pred = data[feature_cols]

# Predict
log_price_pred = pipeline.predict(X_pred)
data['Predicted Price (Lakhs)'] = np.exp(log_price_pred) - 1

print(data[['Property Title','Location','Predicted Price (Lakhs)']])
