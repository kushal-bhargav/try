import numpy as np
import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import classification_report, confusion_matrix, log_loss, f1_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from tqdm import tqdm
from . import config
from .models import ModelFactory
from .utils import CSMOUTE

class PipelineEngine:
    def __init__(self):
        self.skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
        self.meta_model_l1 = ModelFactory.get_meta_learner() # Usually RF or Logistic
        self.meta_model_l2 = None # Will be initialized in train_level2
        self.model_names = ["LightGBM", "CatBoost", "XGBoost", "SVM", "Neural Network"]
        self.oofs = None
        self.test_preds = None

    def _oof_target_encode(self, train_df, val_df, test_df, y_train, cat_cols):
        """Fold-based Target Encoding to prevent leakage."""
        train_enc = pd.DataFrame(index=train_df.index)
        val_enc = pd.DataFrame(index=val_df.index)
        test_enc_fold = pd.DataFrame(index=test_df.index)
        
        kf = KFold(n_splits=config.TE_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        K = 3 # 3 classes
        alpha = 30 # Smoothing
        global_p = np.bincount(y_train, minlength=K) / len(y_train)

        for c in cat_cols:
            if c not in train_df.columns: continue
            
            # OOF encoding for train/val
            oof_train = [np.zeros(len(train_df)) for _ in range(K)]
            for tr_idx, va_idx in kf.split(train_df):
                tr_c = train_df.iloc[tr_idx][c]
                va_c = train_df.iloc[va_idx][c]
                y_tr_fold = y_train[tr_idx]
                
                grp = pd.DataFrame({"cat": tr_c.values, "y": y_tr_fold})
                cnts = grp.groupby("cat", observed=False)["y"].count()
                for k in range(K):
                    pos = grp["y"].eq(k).groupby(grp["cat"], observed=False).sum()
                    te_map = (pos + alpha * global_p[k]) / (cnts + alpha)
                    oof_train[k][va_idx] = va_c.astype(object).map(te_map).fillna(global_p[k]).values

            # Full train encoding for val and test
            full_grp = pd.DataFrame({"cat": train_df[c].values, "y": y_train})
            full_cnts = full_grp.groupby("cat", observed=False)["y"].count()
            for k in range(K):
                full_pos = full_grp["y"].eq(k).groupby(full_grp["cat"], observed=False).sum()
                full_map = (full_pos + alpha * global_p[k]) / (full_cnts + alpha)
                
                # Assign to result DataFrames
                train_enc[f"{c}_te_{k}"] = oof_train[k]
                val_enc[f"{c}_te_{k}"] = val_df[c].astype(object).map(full_map).fillna(global_p[k]).values
                test_enc_fold[f"{c}_te_{k}"] = test_df[c].astype(object).map(full_map).fillna(global_p[k]).values
        
        return train_enc, val_enc, test_enc_fold

    def run_cross_validation(self, X, y, X_test, sample_weights=None):
        n_classes = 3
        n_models = len(self.model_names)
        self.oofs = [np.zeros((X.shape[0], n_classes)) for _ in range(n_models)]
        self.test_preds = [[] for _ in range(n_models)]

        # 1. Feature Consistency: Align X_test with X
        # Ensure X_test has all columns in X and nothing else, in the same order
        print(f"  [LOG] Aligning test features with training schema...")
        for col in X.columns:
            if col not in X_test.columns:
                X_test[col] = 0 # Missing columns in test get 0 (standard for FE indicators)
        X_test = X_test[X.columns]
        
        # Robust Categorical Detection
        cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        num_cols = [c for c in X.columns if c not in cat_cols]
        print(f"  [LOG] Identified {len(cat_cols)} categorical features: {cat_cols[:5]}...")

        # --- Part 1: Tree Models (10-Fol CV) ---
        print(f"\n[Phase 1/2] Training Tree Models ({config.N_SPLITS}-fold CV)...")
        pbar_trees = tqdm(total=config.N_SPLITS, desc="Tree Models CV")
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(X, y)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            sw_fold = sample_weights[train_idx] if sample_weights is not None else None
            
            # Nested TE for LGBM/XGB
            tr_te, va_te, te_te = self._oof_target_encode(X_train_fold, X_val_fold, X_test, y_train_fold.values, cat_cols)
            X_tr_enc = pd.concat([X_train_fold[num_cols].reset_index(drop=True), tr_te.reset_index(drop=True)], axis=1)
            X_va_enc = pd.concat([X_val_fold[num_cols].reset_index(drop=True), va_te.reset_index(drop=True)], axis=1)
            X_te_enc = pd.concat([X_test[num_cols].reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)

            # 1. CatBoost
            cat = ModelFactory.get_catboost(cat_features=cat_cols)
            cat.fit(X_train_fold, y_train_fold, sample_weight=sw_fold, verbose=0)
            self.oofs[1][val_idx] = cat.predict_proba(X_val_fold)
            self.test_preds[1].append(cat.predict_proba(X_test))

            # 2. LightGBM
            lgbm = ModelFactory.get_lgbm()
            lgbm.fit(X_tr_enc, y_train_fold, sample_weight=sw_fold)
            self.oofs[0][val_idx] = lgbm.predict_proba(X_va_enc)
            self.test_preds[0].append(lgbm.predict_proba(X_te_enc))

            # 3. XGBoost
            xgb = ModelFactory.get_xgboost()
            xgb.fit(X_tr_enc, y_train_fold, sample_weight=sw_fold)
            self.oofs[2][val_idx] = xgb.predict_proba(X_va_enc)
            self.test_preds[2].append(xgb.predict_proba(X_te_enc))
            
            pbar_trees.update(1)
            pbar_trees.set_postfix({"Fold": fold+1})
        pbar_trees.close()

        # --- Part 2: SVM & Neural Network (5-fold CV) ---
        print(f"\n[Phase 2/2] Training SVM & NN ({config.N_SPLITS_SVM}-fold CV)...")
        skf_svm = StratifiedKFold(n_splits=config.N_SPLITS_SVM, shuffle=True, random_state=config.RANDOM_STATE)
        pbar_svm = tqdm(total=config.N_SPLITS_SVM, desc="SVM/NN CV")
        
        for fold, (train_idx, val_idx) in enumerate(skf_svm.split(X, y)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            
            # Preparation (Encoding + Imputation + Scaling)
            tr_te, va_te, te_te = self._oof_target_encode(X_train_fold, X_val_fold, X_test, y_train_fold.values, cat_cols)
            X_tr_enc = pd.concat([X_train_fold[num_cols].reset_index(drop=True), tr_te.reset_index(drop=True)], axis=1)
            X_va_enc = pd.concat([X_val_fold[num_cols].reset_index(drop=True), va_te.reset_index(drop=True)], axis=1)
            X_te_enc = pd.concat([X_test[num_cols].reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)

            imputer = SimpleImputer(strategy='median')
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(imputer.fit_transform(X_tr_enc))
            X_va_sc = scaler.transform(imputer.transform(X_va_enc))
            X_te_sc = scaler.transform(imputer.transform(X_te_enc))

            # 1. SVM (with CSMOUTE)
            csmoute = CSMOUTE(smote_ratio=config.CSMOUTE_RATIO, random_state=config.RANDOM_STATE + fold)
            X_tr_res, y_tr_res = csmoute.fit_resample(X_tr_sc, y_train_fold)
            svm = ModelFactory.get_svm()
            svm.fit(X_tr_res, y_tr_res)
            self.oofs[3][val_idx] = svm.predict_proba(X_va_sc)
            self.test_preds[3].append(svm.predict_proba(X_te_sc))

            # 2. Neural Network
            nn = ModelFactory.get_nn()
            nn.fit(X_tr_sc, y_train_fold)
            self.oofs[4][val_idx] = nn.predict_proba(X_va_sc)
            self.test_preds[4].append(nn.predict_proba(X_te_sc))
            
            pbar_svm.update(1)
            pbar_svm.set_postfix({"Fold": fold+1})
        pbar_svm.close()

        print("\nCross-Validation Complete.")
        avg_test_preds = [np.mean(fold_preds, axis=0) for fold_preds in self.test_preds]
        return self.oofs, avg_test_preds

    def train_metastack(self, oofs, y_true):
        """2-Level Stacking Architecture."""
        print(f"Level 1 Stacking ({config.META_MODEL_TYPE})...")
        
        # Adjust stacking features: Cat(1), LGB(0), XGB(2), 0.5*SVM(3), NN(4)
        scaled_oofs = oofs.copy()
        scaled_oofs[3] = scaled_oofs[3] * config.SVM_STACK_WEIGHT
        
        X_meta_l1 = np.concatenate(scaled_oofs, axis=1)
        self.meta_model_l1.fit(X_meta_l1, y_true)
        
        # Level 2 probabilities (OOF)
        l1_oof_probs = self.meta_model_l1.predict_proba(X_meta_l1)
        
        print("Level 2 Stacking (CatBoost)...")
        self.meta_model_l2 = CatBoostClassifier(**config.META2_PARAMS)
        self.meta_model_l2.fit(l1_oof_probs, y_true)

    def predict_stack(self, test_preds):
        """Final tiered prediction."""
        scaled_test = test_preds.copy()
        scaled_test[3] = scaled_test[3] * config.SVM_STACK_WEIGHT
        
        X_meta_test_l1 = np.concatenate(scaled_test, axis=1)
        l1_test_probs = self.meta_model_l1.predict_proba(X_meta_test_l1)
        return self.meta_model_l2.predict_proba(l1_test_probs) # Return probs for blending

    def apply_country_blend(self, X_train, y_train, X_test, global_preds_probs, cat_features=None):
        """Refines predictions per-country if enough data exists."""
        if 'country' not in X_train.columns:
            return global_preds_probs
        
        print("\n[Optional] Applying Country Blending...")
        countries = X_train['country'].unique()
        country_proba = np.zeros((len(X_test), 3))
        
        for ctry in countries:
            tr_mask = X_train['country'] == ctry
            te_mask = X_test['country'] == ctry
            if tr_mask.sum() < 200: # Min samples for local model
                continue
            
            print(f"  Refining for {ctry} ({tr_mask.sum()} samples)...")
            
            # [FIX] Robustly ensure categorical dtypes for the local slice
            X_tr_local = X_train[tr_mask].copy()
            X_te_local = X_test[te_mask].copy()
            
            if cat_features:
                for col in cat_features:
                    X_tr_local[col] = X_tr_local[col].astype('category')
                    X_te_local[col] = X_te_local[col].astype('category')
            
            local_model = ModelFactory.get_catboost(cat_features=cat_features)
            local_model.fit(X_tr_local, y_train[tr_mask])
            country_proba[te_mask] = local_model.predict_proba(X_te_local)
            
        # Blend: 75% Global stack, 25% Local country model
        blended = (1 - config.COUNTRY_BLEND_WEIGHT) * global_preds_probs + \
                  config.COUNTRY_BLEND_WEIGHT * country_proba
        return blended

    def evaluate_oof(self, oof_preds, y, model_name="Model"):
        y_pred = np.argmax(oof_preds, axis=1)
        print(f"\n{'='*20} OOF Evaluation: {model_name} {'='*20}")
        
        # 1. Main Metrics
        acc = accuracy_score(y, y_pred)
        f1_w = f1_score(y, y_pred, average='weighted')
        loss = log_loss(y, oof_preds)
        
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  LogLoss:   {loss:.4f}")
        print(f"  F1-Score (weighted): {f1_w:.4f}")
        
        # 2. Per-class Recall/F1
        print("\n  Per-Class Performance:")
        report = classification_report(y, y_pred, target_names=config.TARGET_MAPPING.keys(), output_dict=True)
        for cls_name in config.TARGET_MAPPING.keys():
            metrics = report[cls_name]
            print(f"    - {cls_name:7}: F1={metrics['f1-score']:.4f}, Recall={metrics['recall']:.4f}")
        
        print(f"{'='*56}\n")
        return report
