import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from . import config
from .utils import normalize_categorical
from scipy.stats import chi2_contingency

class DataProcessor:
    def __init__(self):
        self.missing_handling = {}
        self.impute_vals = {}

    @staticmethod
    def load_data():
        train_df = pd.read_csv(config.TRAIN_PATH)
        test_df = pd.read_csv(config.TEST_PATH)
        return train_df, test_df

    def _check_missingness_dependency(self, series, y):
        if y is None: return False
        table = pd.crosstab(series.isna(), y)
        if table.shape[0] < 2: return False
        try:
            _, p, _, _ = chi2_contingency(table)
            return p < 0.05
        except:
            return False

    def _feature_engineering(self, df, y=None, is_tree_path=True):
        df = df.copy()
        
        # 0. Structural Cleaning (Pre-Normalization)
        if "current_problem_cash_flow" in df.columns:
            df["current_problem_cash_flow"] = df["current_problem_cash_flow"].replace('0', 'No')
            
        # 0b. Dynamic Missingness Indicators
        if y is not None:
            for col in df.columns:
                pct = df[col].isna().mean()
                if pct > 0:
                    if self._check_missingness_dependency(df[col], y) or pct > 0.20:
                        self.missing_handling[col] = "indicator"
                    if pd.api.types.is_numeric_dtype(df[col]):
                        self.impute_vals[col] = df[col].median()
                    else:
                        m = df[col].mode()
                        self.impute_vals[col] = m.iloc[0] if len(m) > 0 else "missing"

        # Module-Based Structural Indicators (Identified in Deep Analysis)
        modules = {
            'mod_banking': ['has_internet_banking', 'has_debit_card'],
            'mod_insurance': ['medical_insurance', 'funeral_insurance'],
            'mod_records': ['offers_credit_to_customers', 'attitude_satisfied_with_achievement'],
            'mod_owner': ['has_cellphone', 'owner_sex']
        }
        for mod_name, cols in modules.items():
            if all(c in df.columns for c in cols):
                df[f"{mod_name}_isna"] = df[cols].isna().all(axis=1).astype(int)

        for col, strategy in self.missing_handling.items():
            if col in df.columns and strategy == "indicator":
                # Only add if not already covered by a module indicator
                if not any(col in mc for mc in modules.values()):
                    df[f"{col}_isna"] = df[col].isna().astype(int)

        # 1. Numeric Features
        if "business_turnover" in df.columns and "business_expenses" in df.columns:
            t = pd.to_numeric(df["business_turnover"], errors="coerce")
            e = pd.to_numeric(df["business_expenses"],  errors="coerce")
            df["profit"] = t - e
            df["profit_margin"] = df["profit"] / (t + 1.0)
            df["health_ratio"] = t / (e + 1.0)
        
        if "personal_income" in df.columns and "business_expenses" in df.columns:
            p, e = pd.to_numeric(df["personal_income"], "coerce"), pd.to_numeric(df["business_expenses"], "coerce")
            df["personal_coverage"] = p / (e + 1.0)

        if "business_age_years" in df.columns and "business_age_months" in df.columns:
            yr, mo = pd.to_numeric(df["business_age_years"], "coerce"), pd.to_numeric(df["business_age_months"], "coerce")
            # Logic: If yr is 0, mo is likely the signal.
            df["biz_months"] = yr * 12 + mo.fillna(0)

        # 2. Categorical Normalization & Interactions
        cat_cols_to_norm = [c for c in df.columns if c in config.CATEGORICAL_FEATURES]
        for c in cat_cols_to_norm:
            df[c] = df[c].map(normalize_categorical)
            
        # 2b. High-Signal Interactions (Identified in Deep Analysis)
        # Funeral Insurance is a massive discriminator
        if "funeral_insurance" in df.columns:
            if "medical_insurance" in df.columns:
                df["feat_full_insurance"] = ((df["funeral_insurance"] == 'yes') & (df["medical_insurance"] == 'yes')).astype(int)
            if "has_internet_banking" in df.columns:
                df["feat_funeral_banking"] = ((df["funeral_insurance"] == 'yes') & (df["has_internet_banking"] == 'yes')).astype(int)
                
        # 2c. Financial Usage Index
        finance_cols = ['has_mobile_money', 'has_credit_card', 'has_loan_account', 'has_internet_banking', 'has_debit_card', 'has_insurance']
        available_finance = [c for c in finance_cols if c in df.columns]
        if available_finance:
            df["financial_usage_index"] = df[available_finance].eq("yes").sum(axis=1)

        # 2d. Age Binning (Identified in Deep Analysis)
        if "owner_age" in df.columns:
            age = pd.to_numeric(df["owner_age"], errors="coerce")
            df["owner_age_bins"] = pd.cut(age, bins=[0, 18, 25, 35, 45, 55, 65, 120], 
                                         labels=['<18', '18-25', '26-35', '36-45', '46-55', '56-65', '65+']).astype(str)
            df["owner_age_bins"] = df["owner_age_bins"].fillna("missing")

        # 3. Log Branch
        if not is_tree_path:
            for src in ["business_turnover", "business_expenses", "personal_income", "profit"]:
                if src in df.columns:
                    df[f"{src}_log"] = np.log1p(np.clip(pd.to_numeric(df[src], "coerce"), 0, None))

        return df

    def _impute_and_format(self, df):
        for col in config.CATEGORICAL_FEATURES:
            if col in df.columns:
                m = df[col].mode()
                df[col] = df[col].fillna(m.iloc[0] if len(m) > 0 else "missing").astype('category')
        return df

    def fit_transform(self, train_df):
        y = train_df[config.TARGET_COL].map(config.TARGET_MAPPING)
        X_tree = self._impute_and_format(self._feature_engineering(train_df, y=y, is_tree_path=True))
        X_tree = X_tree.drop(columns=[config.ID_COL, config.TARGET_COL])
        X_linear = self._feature_engineering(train_df, y=y, is_tree_path=False).drop(columns=[config.ID_COL, config.TARGET_COL])
        return X_tree, y, X_linear

    def transform(self, test_df):
        X_tree_test = self._impute_and_format(self._feature_engineering(test_df, is_tree_path=True))
        X_tree_test = X_tree_test.drop(columns=[config.ID_COL])
        X_linear_test = self._feature_engineering(test_df, is_tree_path=False).drop(columns=[config.ID_COL])
        return X_tree_test, X_linear_test
