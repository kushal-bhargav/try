import pandas as pd
import numpy as np
import re
import os
import sys
from collections import Counter

# Force UTF-8 for Windows terminal
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

def clean_val(v):
    if v is None: return "nan"
    # Replace problematic chars for printing
    return str(v).replace("\u200e", "").replace("\ufffd", "?")

def analyze():
    path = 'c:/Users/KushalBhargav/OneDrive - Systech Solutions, Inc/L&D_Automation/code_bak/data/train.csv'
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return
    
    df = pd.read_csv(path)
    target = 'Target'
    
    print(f"--- Dataset Overview ---")
    print(f"Shape: {df.shape}")
    
    # 1. Categorical Semantic & Quality Analysis
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print("\n--- Categorical Feature Deep Dive ---")
    for col in cat_cols:
        if col == 'ID': continue
        counts = df[col].value_counts(dropna=False)
        print(f"\n[{clean_val(col)}]")
        print(f"Unique values: {len(counts)}")
        print(f"Top 5 values: {{k: v for k, v in [(clean_val(k), v) for k, v in counts.head(5).items()]}}")
        
        unique_vals = [str(x) for x in counts.index if pd.notna(x)]
        for i, v1 in enumerate(unique_vals):
            for v2 in unique_vals[i+1:]:
                s1 = re.sub(r'[^a-z0-9]', '', v1.lower())
                s2 = re.sub(r'[^a-z0-9]', '', v2.lower())
                if s1 == s2 and v1 != v2:
                    print(f"  [OVERLAP] '{clean_val(v1)}' vs '{clean_val(v2)}'")
                if (('know' in s1 and 'know' in s2) or ('refuse' in s1 and 'refuse' in s2)):
                    if s1 != s2:
                         print(f"  [SEMANTIC] '{clean_val(v1)}' vs '{clean_val(v2)}'")

    # 2. Numerical Quality & Junk Detection
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\n--- Numerical Feature Quality ---")
    for col in num_cols:
        stats = df[col].describe()
        zeros = (df[col] == 0).sum()
        negatives = (df[col] < 0).sum()
        missing = df[col].isna().sum()
        
        print(f"\n[{clean_val(col)}]")
        print(f"Missing: {missing} ({missing/len(df)*100:.1f}%)")
        print(f"Mean: {stats['mean']:.2f} | Zeros: {zeros} | Negatives: {negatives}")
        
        # Detect constant or outlier-heavy columns
        if stats['std'] == 0:
            print(f"  [WARNING] Constant column detected.")

    # 3. Correlation & Signal Analysis (Target-Focused)
    print("\n--- Feature-Target Correlation & Signal Discovery ---")
    if target in df.columns:
        y_codes = df[target].astype('category').cat.codes
        
        # A. Numerical Interactions
        print("\n[Numerical Ratios/Products with High Correlation Delta]")
        base_num_corr = {col: df[col].corr(y_codes) for col in num_cols if pd.api.types.is_numeric_dtype(df[col])}
        
        # Check simple interactions
        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i+1:]:
                # Ratio interaction
                ratio = df[c1] / (df[c2] + 1e-6)
                corr = abs(ratio.corr(y_codes))
                if not pd.isna(corr) and corr > max(abs(base_num_corr.get(c1, 0)), abs(base_num_corr.get(c2, 0))) + 0.1:
                    print(f"  [CANDIDATE] Ratio interaction: {c1}/{c2} | Corr: {corr:.3f} (Gain: {corr - max(abs(base_num_corr.get(c1,0)), abs(base_num_corr.get(c2,0))):.3f})")

        # B. Categorical Interaction (Crosstabs)
        print("\n[High Interaction Categorical Pairs (Heatmap Candidates)]")
        # Sample for speed if needed, but 9k is fine
        for i, c1 in enumerate(cat_cols):
            if c1 in ['ID', target]: continue
            for c2 in cat_cols[i+1:]:
                if c2 in ['ID', target]: continue
                # Identify pairs where specific combinations strongly predict a target
                ct = pd.crosstab([df[c1], df[c2]], df[target], normalize='index')
                if ct.max(axis=1).mean() > 0.8: # Strong predictive power on average
                     print(f"  [SIGNAL] Interaction {c1} & {c2} shows strong class separation.")

    # 4. Final Recommendations based on Correlations
    print("\n--- Final Feature Engineering Recommendations ---")
    print("1. Create 'Total Financial Service Usage' index (Sum of binary 'has_...' columns).")
    print("2. Create 'Business Health Ratio' (Turnover / Expenses) if not exists.")
    print("3. Interaction feature for high-signal categorical pairs like 'country' and 'covid_essential_service'.")

from collections import Counter
if __name__ == '__main__':
    analyze()
