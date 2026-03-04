import pandas as pd
import numpy as np
import time
from category_encoders import TargetEncoder
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from . import config
from .utils import normalize_categorical
from scipy.stats import chi2_contingency

class DataProcessor:
    def __init__(self):
        self.encoder = None
        self.scaler = StandardScaler()
        self.missing_handling = {}
        self.impute_cols = None
        self.scale_cols = None
        self.final_feature_order = None # For MICE consistency

    @staticmethod
    def load_data():
        train_df = pd.read_csv(config.TRAIN_PATH)
        test_df = pd.read_csv(config.TEST_PATH)
        return train_df, test_df

    def _feature_engineering(self, df, y=None):
        # ... (previous FE code remains)
        
        # 0. Dynamic Missingness Indicators (Chi2 based)
        print(f"  [LOG] Analyzing missingness for {len(df.columns)} columns...")
        start_t = time.time()
        for i, col in enumerate(df.columns):
            pct = df[col].isna().mean()
            if pct > 0:
                # Add indicator if missingness is high or has relationship with target
                add_indicator = False
                if pct > 0.05: # High missingness
                    add_indicator = True
                elif y is not None: # Check dependency
                    table = pd.crosstab(df[col].isna(), y)
                    if table.shape[0] >= 2:
                        _, p, _, _ = chi2_contingency(table)
                        if p < 0.05: add_indicator = True
                
                if add_indicator:
                    df[f"{col}_isna"] = df[col].isna().astype(int)
                    if y is not None: # Only record during fit
                         self.missing_handling[col] = "indicator"
            
            if (i+1) % 20 == 0:
                print(f"    - Processed {i+1} columns...")
        
        print(f"  [LOG] Missingness analysis complete. Runtime: {time.time()-start_t:.2f}s")

        # 1. Zindi "Gold" Numeric Features
        if "business_turnover" in df.columns and "business_expenses" in df.columns:
            t = pd.to_numeric(df["business_turnover"], errors="coerce")
            e = pd.to_numeric(df["business_expenses"],  errors="coerce")
            df["profit"]          = t - e
            df["profit_margin"]   = df["profit"] / (t + 1.0)
            df["turnover_ratio"]  = t / (e + 1.0)
            df["is_profitable"]   = (t > e).astype(int)
        
        if "personal_income" in df.columns and "business_turnover" in df.columns:
            p = pd.to_numeric(df["personal_income"],   errors="coerce")
            t = pd.to_numeric(df["business_turnover"], errors="coerce")
            df["income_ratio"] = p / (t + 1.0)
            df["income_to_expense_ratio"] = p / (pd.to_numeric(df["business_expenses"], errors="coerce") + 1.0)

        if "business_age_years" in df.columns and "business_age_months" in df.columns:
            yr = pd.to_numeric(df["business_age_years"],  errors="coerce")
            mo = pd.to_numeric(df["business_age_months"], errors="coerce")
            df["biz_months"] = yr * 12 + mo
            df["turnover_per_month"] = pd.to_numeric(df["business_turnover"], errors="coerce") / (df["biz_months"] + 1)

        # 2. Log Transforms (Winning strategy used clipped log1p)
        for src, out in [("business_turnover", "turnover_log"),
                         ("business_expenses",  "expenses_log"),
                         ("personal_income",    "income_log")]:
            if src in df.columns:
                val = pd.to_numeric(df[src], errors="coerce")
                df[out] = np.log1p(np.clip(val, 0, None))

        # 3. Insurance Aggregation
        insurance_cols = ['medical_insurance', 'funeral_insurance']
        df['total_insurance_score'] = 0
        for col in insurance_cols:
            if col in df.columns:
                df['total_insurance_score'] += df[col].map({
                    'Never had': 0, 
                    'Used to have but don’t have now': 1, 
                    'Currently have': 2
                }).fillna(0)

        # 4. Categorical Normalization (norm_cat)
        cat_cols = [c for c in df.columns if c in config.CATEGORICAL_FEATURES]
        for c in cat_cols:
            df[c] = df[c].map(normalize_categorical)

        # 5. Age bins
        age_bins = [0, 25, 35, 45, 55, 65, 120]
        age_labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
        if 'owner_age' in df.columns:
            df['owner_age_bins'] = pd.cut(df['owner_age'], bins=age_bins, labels=age_labels)
        
        return df

    def _impute_and_format(self, df):
        # Mode imputation for categorical
        for col in config.CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
                df[col] = df[col].astype('category')
        
        if 'owner_age_bins' in df.columns:
            df['owner_age_bins'] = df['owner_age_bins'].fillna(df['owner_age_bins'].mode()[0])
            df['owner_age_bins'] = df['owner_age_bins'].astype('category')
            
        return df

    def fit_transform(self, train_df):
        print(f"\n[LOG] Starting fit_transform on {len(train_df)} records...")
        
        # Initial cleaning and FE
        y = train_df[config.TARGET_COL].map(config.TARGET_MAPPING)
        print("  [LOG] Running Feature Engineering...")
        train_df = self._feature_engineering(train_df, y=y)
        train_df = self._impute_and_format(train_df)
        
        X = train_df.drop(columns=[config.ID_COL, config.TARGET_COL])
        
        # 1. Target Encoding (Dummy fit for main.py signature, redundant for CV)
        print("  [LOG] Initializing Preprocessors...")
        self.encoder = TargetEncoder(cols=config.CATEGORICAL_FEATURES)
        self.encoder.fit(X, y)
        
        # NOTE: MICE and Scaling are handled INSIDE PipelineEngine for better performance
        # and to prevent leakage during nested CV. Returning X for all outputs.
        X_encoded = X.copy()
        X_scaled = X.copy()
        
        return X, y, X_encoded, X_scaled

    def transform(self, test_df):
        print(f"\n[LOG] Starting transform on {len(test_df)} records...")
        test_df = self._feature_engineering(test_df)
        
        # Add indicators for columns that were identified as missing in train
        for col in self.missing_handling:
            if col in test_df.columns:
                test_df[f"{col}_isna"] = test_df[col].isna().astype(int)

        test_df = self._impute_and_format(test_df)
        X_test = test_df.drop(columns=[config.ID_COL])
        
        # Returning X_test for all to avoid redundant expensive computations
        return X_test, X_test.copy(), X_test.copy()
