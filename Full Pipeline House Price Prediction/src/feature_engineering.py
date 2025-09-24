import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin

class RealEstateFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, remove_outliers=True, top_localities_n=30):
        self.remove_outliers = remove_outliers
        self.top_localities_n = top_localities_n

    def fit(self, X, y=None):
        # Store top localities from training data
        if "Location" in X.columns:
            X_copy = X.copy()
            X_copy['Location_split'] = X_copy['Location'].str.split(',')
            X_copy['Locality'] = X_copy['Location_split'].apply(lambda x: x[0].strip().lower() if x else '')
            top_localities = X_copy['Locality'].value_counts().nlargest(self.top_localities_n).index.tolist()
            self.top_localities_ = top_localities
        return self

    def transform(self, X):
        df = X.copy()

        # ----------------- Price conversion -----------------
        if "Price" in df.columns and "Price_Lakhs" not in df.columns:
            df["Price_Lakhs"] = df["Price"].apply(self.convert_price_to_lakhs)

        price_available = "Price_Lakhs" in df.columns and not df["Price_Lakhs"].isnull().all()

        # ----------------- Split Location -----------------
        if "Location" in df.columns:
            df['Location_split'] = df['Location'].str.split(',')
            df['Locality'] = df['Location_split'].apply(lambda x: x[0].strip().lower() if x else '')
            df['City'] = df['Location_split'].apply(lambda x: x[-1].strip().lower() if x else '')
            df.drop(columns=['Location_split'], inplace=True)

            # Map less frequent localities to "Other"
            if hasattr(self, 'top_localities_'):
                df.loc[~df['Locality'].isin(self.top_localities_), 'Locality'] = 'Other'

        # ----------------- Extract BHK/RK -----------------
        if "Property Title" in df.columns:
            df[["bhk_or_rk", "bhk_category"]] = df["Property Title"].apply(
                lambda x: pd.Series(self.extract_category(x))
            )
            df["BHK"] = df["bhk_category"].apply(self.extract_num)

        # ----------------- Binary Flags -----------------
        if "Balcony" in df.columns:
            df['Balcony_flag'] = df['Balcony'].map({'Yes':1,'Y':1,'No':0,'N':0}).fillna(0)

        if "bhk_or_rk" in df.columns:
            df['BHK_or_RK_flag'] = df['bhk_or_rk'].map({'BHK':1,'bhk':1,'RK':0,'rk':0}).fillna(0)

        # ----------------- Numeric conversion -----------------
        for col in ['Total_Area', 'Price_per_SQFT', 'Baths']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # ----------------- Outlier removal -----------------
        if self.remove_outliers and price_available:
            df = self.remove_outliers_iqr(df)

        # ----------------- Feature Engineering -----------------
        df = self.create_extra_features(df, price_available)

        return df

    # ----------------- Helper Methods -----------------
    def convert_price_to_lakhs(self, price_str):
        try:
            price_str = str(price_str).replace(",", "").strip()
            match = re.match(r'₹?\s*([\d]+(?:\.\d+)?)\s*(L|Cr|Crore|Crores)?', price_str, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                if unit and unit.lower().startswith("c"): value *= 100
                elif unit and unit.lower().startswith("l"): value *= 1
                else: value /= 1e5
                return value
        except:
            return np.nan
        return np.nan

    def extract_category(self, title):
        if pd.isna(title): return None, None
        title = str(title)
        bhk_match = re.search(r'(\d+)\s*BHK', title, re.IGNORECASE)
        if bhk_match: return "BHK", f"{int(bhk_match.group(1))} BHK"
        rk_match = re.search(r'(\d+)?\s*RK', title, re.IGNORECASE)
        if rk_match:
            num = int(rk_match.group(1)) if rk_match.group(1) else 1
            return "RK", f"{num} RK"
        return "BHK", "1 BHK"

    def extract_num(self, category):
        if pd.isna(category): return np.nan
        match = re.search(r'(\d+)', str(category))
        return int(match.group(1)) if match else np.nan

    def remove_outliers_iqr(self, df, factor=1.5):
        df_clean = df.copy()
        for col in ['Price_Lakhs','Total_Area','Price_per_SQFT']:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                df_clean = df_clean[(df_clean[col] >= Q1 - factor*IQR) & (df_clean[col] <= Q3 + factor*IQR)]
        return df_clean

    def create_extra_features(self, df, price_available=True):
        df = df.copy()
        # --- Basic transforms ---
        df['log_area'] = np.log1p(df['Total_Area'].clip(lower=1))
        df['sqrt_area'] = np.sqrt(df['Total_Area'].clip(lower=0))
        df['inv_area'] = 1 / np.maximum(df['Total_Area'], 1)

        # --- Ratios ---
        df['Area_per_Bath'] = df['Total_Area'] / np.maximum(df['Baths'], 1)
        df['Baths_per_BHK'] = df['Baths'] / np.maximum(df['BHK'], 1)
        df['Area_per_Room'] = df['Total_Area'] / np.maximum(df['BHK'] + df['Baths'], 1)
        df['log_area_per_room'] = np.log1p(df['Area_per_Room'])
        df['Bath_to_BHK_ratio'] = df['Baths'] / np.maximum(df['BHK'], 1)

        # --- Flags ---
        df['is_Compact'] = (df['Total_Area'] < 500).astype(int)
        df['is_Luxury'] = (df['Total_Area'] > 2000).astype(int)
        df['Is_Premium_Size'] = (df['Total_Area'] > df['Total_Area'].median()).astype(int)
        df['Has_Multiple_Baths'] = (df['Baths'] >= 2).astype(int)
        df['Luxury_Score'] = df['is_Luxury'] + df['Has_Multiple_Baths'] + df.get('Balcony_flag', 0)

        # --- Interaction features ---
        df['Area_x_BHK'] = df['Total_Area'] * np.maximum(df['BHK'], 1)
        df['Area_x_Baths'] = df['Total_Area'] * np.maximum(df['Baths'], 1)
        df['log_Area_x_BHK'] = np.log1p(df['Area_x_BHK'])
        df['BHK_x_Baths'] = df['BHK'] * df['Baths']
        df['Balcony_x_BHK'] = df.get('Balcony_flag', 0) * df['BHK']

        # --- Percentiles ---
        df['Area_percentile'] = df['Total_Area'].rank(pct=True)

        # --- Categories ---
        df['Property_Size_Category'] = df['Total_Area'].apply(self.categorize_property_size)
        df['BHK_Category'] = df['BHK'].astype(str) + " " + df['bhk_or_rk']

        # --- Advanced luxury score ---
        df['Advanced_Luxury_Score'] = df.apply(self.advanced_luxury_score, axis=1)

        # --- Price-dependent features ---
        if price_available:
            df['log_price'] = np.log1p(df['Price_Lakhs'])
            df['Price_per_Room'] = df['Price_per_SQFT'] * df['Area_per_Room']
            df['Total_Rooms'] = df['BHK'] + df['Baths']
            df['Area_Efficiency'] = df['Total_Area'] / np.maximum(df['Total_Rooms'], 1)
            df['Locality_Median_Price'] = df.groupby('Locality')["Price_Lakhs"].transform("median")
            df['Price_vs_Locality'] = df['Price_Lakhs'] / df['Locality_Median_Price']
            df['City_Median_Price'] = df.groupby('City')["Price_Lakhs"].transform("median")
            df['Price_vs_City'] = df['Price_Lakhs'] / df['City_Median_Price']

        return df

    def categorize_property_size(self, area):
        if area < 500: return 'Compact'
        elif area < 1000: return 'Medium'
        elif area < 2000: return 'Large'
        else: return 'Luxury'

    def advanced_luxury_score(self, row):
        score = 0
        if row['Total_Area']>2000: score+=3
        elif row['Total_Area']>1200: score+=2
        elif row['Total_Area']>800: score+=1
        if row['BHK']>=4: score+=2
        elif row['BHK']>=3: score+=1
        if row['Baths']>=3: score+=2
        elif row['Baths']>=2: score+=1
        if row.get('Balcony_flag',0)==1: score+=1
        if 'villa' in str(row.get('Property Title','')).lower() or 'penthouse' in str(row.get('Property Title','')).lower(): score+=2
        return score
