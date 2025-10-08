import pandas as pd
import numpy as np
import re
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# -------- Load Data --------
df = pd.read_csv("Data/Real Estate Data V21.csv")



# Convert numeric columns to proper data types
df['Total_Area'] = pd.to_numeric(df['Total_Area'], errors='coerce')
df['Price_per_SQFT'] = pd.to_numeric(df['Price_per_SQFT'], errors='coerce')
df['Baths'] = pd.to_numeric(df['Baths'], errors='coerce')

def price_to_lakhs(p):
    if pd.isnull(p): 
        return np.nan
    p = str(p).replace('₹','').replace(',','').strip().lower()
    try:
        if 'cr' in p: 
            return float(p.replace('cr','').strip())*100
        elif 'l' in p: 
            return float(p.replace('l','').strip())
        else:
            # Unknown format
            return np.nan
    except:
        return np.nan

df['Price_Lakhs'] = df['Price'].apply(price_to_lakhs)
df = df.dropna(subset=['Price_Lakhs'])


# -------- Outlier Removal (IQR) --------
for col in ['Price_Lakhs','Total_Area','Price_per_SQFT']:
    # Skip columns with all NaN values
    if df[col].isna().all():
        print(f"Skipping {col} as it contains only NaN values")
        continue
    # Remove NaN values before calculating quantiles
    clean_col = df[col].dropna()
    if len(clean_col) == 0:
        print(f"Skipping {col} as it contains no valid values")
        continue
    Q1 = clean_col.quantile(0.25)
    Q3 = clean_col.quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col].isna()) | ((df[col]>=Q1-1.5*IQR) & (df[col]<=Q3+1.5*IQR))]

# -------- Extract BHK --------
df['BHK'] = df['Property Title'].apply(lambda t: int(re.search(r'(\d+)\s*BHK', str(t), re.IGNORECASE).group(1)) 
                                      if re.search(r'(\d+)\s*BHK', str(t), re.IGNORECASE) else np.nan)
df = df.dropna(subset=['BHK'])
df['BHK'] = df['BHK'].astype(int)

# Balcony numeric
df['Balcony'] = df['Balcony'].map({'Yes':1,'No':0}).fillna(0)

# Property_Type
def prop_type(t):
    t = str(t).lower()
    if "independent house" in t: return "Independent House"
    elif "flat" in t: return "Flat"
    elif "villa" in t: return "Villa"
    else: return "Other"
df['Property_Type'] = df['Property Title'].apply(prop_type)

# City
df['City'] = df['Location'].apply(lambda x: str(x).split(",")[-1].strip())

# -------- Target Encoding for Location --------
location_target = df.groupby('Location')['Price_Lakhs'].mean().to_dict()
df['Location_Encoded'] = df['Location'].map(location_target)
df['Location_Encoded'] = df['Location_Encoded'].fillna(df['Price_Lakhs'].mean())

# Log transform
df['Log_Total_Area'] = np.log(df['Total_Area'] + 1)
df['Log_Price'] = np.log(df['Price_Lakhs'] + 1)

# Interaction features
df['BHK_x_Total_Area'] = df['BHK'] * df['Total_Area']
df['BHK_x_Baths'] = df['BHK'] * df['Baths']
df['Balcony_x_BHK'] = df['Balcony'] * df['BHK']
df['Log_Total_Area_x_BHK'] = df['Log_Total_Area'] * df['BHK']
df['Log_Total_Area_squared'] = df['Log_Total_Area'] ** 2

# Features & target
feature_cols = ['Log_Total_Area','Baths','Balcony','BHK','Location_Encoded',
                'BHK_x_Total_Area','BHK_x_Baths','Balcony_x_BHK','Log_Total_Area_x_BHK','Log_Total_Area_squared',
                'City','Property_Type']
X = df[feature_cols]
y = df['Log_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessor
num_features = ['Log_Total_Area','Baths','Balcony','BHK','Location_Encoded',
                'BHK_x_Total_Area','BHK_x_Baths','Balcony_x_BHK','Log_Total_Area_x_BHK','Log_Total_Area_squared']
cat_features = ['City','Property_Type']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
])

# Pipeline
pipeline = Pipeline([
    ('preproc', preprocessor),
    ('model', GradientBoostingRegressor(
        subsample=0.7, n_estimators=200, max_features=0.8,
        max_depth=6, loss='huber', learning_rate=0.03, random_state=42
    ))
])

pipeline.fit(X_train, y_train)
joblib.dump((pipeline, location_target), "pipeline.pkl")
print("✅ Pipeline trained and saved!")
