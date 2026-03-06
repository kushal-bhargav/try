import re
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy, chi2_contingency
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score
from catboost import CatBoostClassifier, Pool
from . import config

class CSMOUTE:
    """
    Combined SMOTE + SMUTE hybrid resampler for multiclass imbalance.
    Extracted from Zindi high-scoring notebook.
    """
    def __init__(self, smote_ratio=0.5, k_neighbors=5, random_state=None):
        if not 0.0 <= smote_ratio <= 1.0:
            raise ValueError("smote_ratio must be between 0.0 and 1.0")
        self.smote_ratio = smote_ratio
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        majority_count = counts.max()

        X_parts = [X]
        y_parts = [y]

        # SMOTE: oversample each minority class
        for cls, cnt in zip(classes, counts):
            if cls == majority_class:
                continue

            gap = majority_count - cnt
            n_synthetic = int(round(gap * self.smote_ratio))
            if n_synthetic <= 0:
                continue

            X_cls = X[y == cls]
            synthetics = self._smote_class(X_cls, n_synthetic)
            X_parts.append(synthetics)
            y_parts.append(np.full(len(synthetics), cls, dtype=y.dtype))

        X_res = np.vstack(X_parts)
        y_res = np.concatenate(y_parts)

        # SMUTE: undersample majority class
        new_counts = np.bincount(y_res.astype(int))
        new_majority_cnt = new_counts[majority_class]
        new_minority_avg = int(np.mean([new_counts[c] for c in classes if c != majority_class]))

        gap_after = new_majority_cnt - new_minority_avg
        n_remove = int(round(gap_after * (1.0 - self.smote_ratio)))
        n_remove = max(0, min(n_remove, new_majority_cnt - 1))

        if n_remove > 0:
            maj_mask = y_res == majority_class
            X_maj = X_res[maj_mask]
            X_rest = X_res[~maj_mask]
            y_maj = y_res[maj_mask]
            y_rest = y_res[~maj_mask]

            keep_idx = self._smute_select(X_maj, n_keep=len(X_maj) - n_remove)
            X_res = np.vstack([X_rest, X_maj[keep_idx]])
            y_res = np.concatenate([y_rest, y_maj[keep_idx]])

        perm = self._rng.permutation(len(y_res))
        return X_res[perm], y_res[perm]

    def _smote_class(self, X_cls, n_synthetic):
        k = min(self.k_neighbors, len(X_cls) - 1)
        if k < 1:
            noise = self._rng.normal(0, 1e-6, size=(n_synthetic, X_cls.shape[1]))
            return X_cls[0:1].repeat(n_synthetic, axis=0) + noise

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X_cls)
        _, idxs = nn.kneighbors(X_cls)
        idxs = idxs[:, 1:]

        synthetics = np.empty((n_synthetic, X_cls.shape[1]), dtype=np.float64)
        for i in range(n_synthetic):
            base = self._rng.randint(0, len(X_cls))
            nn_pick = idxs[base, self._rng.randint(0, k)]
            lam = self._rng.uniform(0, 1)
            synthetics[i] = X_cls[base] + lam * (X_cls[nn_pick] - X_cls[base])
        return synthetics

    def _smute_select(self, X_maj, n_keep):
        centroid = X_maj.mean(axis=0)
        dists = np.linalg.norm(X_maj - centroid, axis=1)
        keep_idx = np.argsort(dists)[::-1][:n_keep]
        return keep_idx

def normalize_categorical(val):
    if pd.isna(val) or val == '':
        return 'missing'
    
    # Standardize to lowercase and strip whitespace/LRM marks
    s = str(val).lower().strip().replace('\u200e', '')
    
    # 1. Unify "Don't Know" variants
    if re.search(r"don[?'\u2019]?t\s+know|do\s+not\s+know|don\?t\s+know|no\s+s\u00e9|know\s+or\s+n/a", s):
        return 'unknown'
    
    # 2. Unify "Refused" variants
    if re.search(r"refuse|no\s+respon|declined", s):
        return 'refused'
    
    # 3. Unify "Used to have but don't have now" variants
    if re.search(r"used\s+to\s+have|previously\s+had", s):
        return 'previously_had'
    
    # 4. Specific Junk cleaning
    if s == '0' or s == 'none':
        return 'no'
    
    # 5. Passthrough for standard values, stripping non-alpha for consistency
    if s in ['yes', 'no']: return s
    if 'sometimes' in s: return 'yes_sometimes'
    if 'always' in s: return 'yes_always'
    
    return s
from category_encoders import TargetEncoder

def oof_target_encode(X_tr, X_va, X_te, y_tr, cat_cols, seed=42):
    """
    Perform OOF Target Encoding for train, val, and test data.
    """
    te = TargetEncoder(cols=cat_cols, smoothing=10)
    te.fit(X_tr[cat_cols], y_tr)
    return te.transform(X_tr[cat_cols]), te.transform(X_va[cat_cols]), te.transform(X_te[cat_cols])

def analyze_class_imbalance(y, output_dir="."):
    """Computes dynamic class weights using Gold v5 Severity scoring."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    props = counts / total

    # 1. Imbalance Metrics
    ir = counts.max() / counts.min()
    cv = np.std(counts) / np.mean(counts)
    ent = entropy(props, base=len(classes))
    sorted_props = np.sort(props)
    n = len(sorted_props)
    gini = (2 * np.sum(np.arange(1, n+1) * sorted_props)) / (n * np.sum(sorted_props)) - (n+1)/n

    # 2. Severity Scoring (0-7)
    severity = 0
    if ir > 20: severity += 3
    elif ir > 10: severity += 2
    elif ir > 5: severity += 1
    
    min_prop = props.min()
    if min_prop < 0.05: severity += 3
    elif min_prop < 0.10: severity += 2
    elif min_prop < 0.20: severity += 1
    
    if ent < 0.80: severity += 1

    # 3. Strategy Selection
    if severity >= 5:   strategy = "AGGRESSIVE"
    elif severity >= 3: strategy = "MODERATE"
    elif severity >= 1: strategy = "MILD"
    else:               strategy = "NONE"

    # 4. Weight Calculation
    if strategy == "AGGRESSIVE":
        raw = total / (len(classes) * counts)
        weights = np.sqrt(raw)
        weights = weights / weights.min()
        weights = np.minimum(weights, 5.0)
    elif strategy == "MODERATE":
        raw = total / (len(classes) * counts)
        weights = np.power(raw, 0.5)
        weights = weights / weights.min()
        weights = np.minimum(weights, 3.0)
    else:
        weights = np.ones(len(classes))

    print(f"[Imbalance] IR: {ir:.2f} | Ent: {ent:.2f} | Severity: {severity}/7 -> {strategy}")
    return weights

def compute_dynamic_alpha(X, y, beta=0.999, lambda_miss=0.3, n_splits=5, seed=42):
    """Difficulty-based focal alpha using CatBoost OOF recall."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)

    # 1. Effective number weighting
    base = (1 - beta) / (1 - np.power(beta, counts))
    base_arr = np.asarray(base)
    if base_arr.size > 0:
        base = base_arr / base_arr.mean()
    else:
        base = np.ones(len(classes))

    # 2. Missingness adjustment
    global_miss = X.isna().mean().mean() if isinstance(X, pd.DataFrame) else 0.0
    miss_adj = []
    for c in classes:
        if isinstance(X, pd.DataFrame):
            mc = X.loc[y == c].isna().mean().mean()
        else:
            mc = global_miss
        adj = 1 - lambda_miss * (mc - global_miss)
        miss_adj.append(adj)
    miss_adj_arr = np.array(miss_adj)
    miss_adj = np.clip(miss_adj_arr, 0.7, 1.3)

    # 3. Difficulty estimation (Simplified CatBoost OOF)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros_like(y)
    
    # We only take numeric for difficulty estimation speed
    X_num = X.select_dtypes(include=[np.number]) if isinstance(X, pd.DataFrame) else X

    for tr, va in skf.split(X_num, y):
        model = CatBoostClassifier(iterations=200, depth=4, verbose=False, random_seed=seed)
        model.fit(X_num.iloc[tr], y[tr])
        oof_pred[va] = model.predict(X_num.iloc[va]).flatten()

    recalls = recall_score(y, oof_pred, average=None)
    difficulty = 1.0 / (recalls + 1e-6)
    diff_arr = np.asarray(difficulty)
    if diff_arr.size > 0:
        difficulty = diff_arr / diff_arr.mean()
    else:
        difficulty = np.ones(len(classes))

    alpha = base * miss_adj * difficulty
    alpha = alpha / alpha.sum()
    alpha = np.clip(alpha, 0.05, 0.7)
    return alpha / alpha.sum()
