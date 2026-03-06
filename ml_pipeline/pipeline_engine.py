import numpy as np
import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import classification_report, confusion_matrix, log_loss, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from catboost import CatBoostClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from . import config
from .models import ModelFactory
from .utils import CSMOUTE, oof_target_encode, compute_dynamic_alpha, analyze_class_imbalance

class PipelineEngine:
    def __init__(self):
        self.skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
        self.meta_model_l1 = ModelFactory.get_meta_learner()
        self.meta_model_l2 = None
        self.model_names = ["LightGBM", "CatBoost", "XGBoost", "SVM", "Neural Network"]
        self.oofs = None
        self.test_preds = None

    def run_cross_validation(self, X_tree, y, X_linear, X_tree_test, X_linear_test):
        n_classes = 3
        n_models = len(self.model_names)
        self.oofs = [np.zeros((X_tree.shape[0], n_classes)) for _ in range(n_models)]
        self.test_preds = [[] for _ in range(n_models)]

        # --- Phase 1: Tree Models ---
        print(f"\n[Phase 1/2] Training Tree Models ({config.N_SPLITS}-fold CV)...")
        cat_cols = X_tree.select_dtypes(include=['category', 'object']).columns.tolist()
        num_cols = [c for c in X_tree.columns if c not in cat_cols]
        
        alpha_weights = compute_dynamic_alpha(X_tree, y)
        class_weights = analyze_class_imbalance(y)
        
        print(f"  [LOG] Dynamic Weights Ready. Alpha (normalized): {alpha_weights}")

        pbar_trees = tqdm(total=config.N_SPLITS, desc="Trees")
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(X_tree, y)):
            X_tr_tree, X_va_tree = X_tree.iloc[train_idx], X_tree.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            sw_fold = None # Reserved for external weighting if needed
            
            # 1. CatBoost (Main signal)
            cb = ModelFactory.get_catboost(cat_features=cat_cols, class_weights=alpha_weights.tolist())
            cb.fit(X_tr_tree, y_tr, sample_weight=sw_fold, verbose=0)
            self.oofs[1][val_idx] = cb.predict_proba(X_va_tree)
            self.test_preds[1].append(cb.predict_proba(X_tree_test))

            # 2. Nested TE for LGBM/XGB
            tr_te, va_te, te_te = oof_target_encode(X_tr_tree, X_va_tree, X_tree_test, y_tr, cat_cols, seed=config.RANDOM_STATE+fold)
            X_tr_enc = pd.concat([X_tr_tree[num_cols].reset_index(drop=True), tr_te.reset_index(drop=True)], axis=1)
            X_va_enc = pd.concat([X_va_tree[num_cols].reset_index(drop=True), va_te.reset_index(drop=True)], axis=1)
            X_te_enc = pd.concat([X_tree_test[num_cols].reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)

            # 2. LightGBM
            lgbm = ModelFactory.get_lgbm(class_weights=class_weights)
            lgbm.fit(X_tr_enc, y_tr, sample_weight=sw_fold)
            self.oofs[0][val_idx] = lgbm.predict_proba(X_va_enc)
            self.test_preds[0].append(lgbm.predict_proba(X_te_enc))

            # 3. XGBoost
            xgb = ModelFactory.get_xgboost()
            # Combine external sample_weight with class_weights for XGB
            final_sw = np.array([class_weights[int(label)] for label in y_tr])
            if sw_fold is not None:
                final_sw *= sw_fold
            xgb.fit(X_tr_enc, y_tr, sample_weight=final_sw)
            self.oofs[2][val_idx] = xgb.predict_proba(X_va_enc)
            self.test_preds[2].append(xgb.predict_proba(X_te_enc))
            
            pbar_trees.update(1)
        pbar_trees.close()

        # --- Phase 2: SVM & NN ---
        print(f"\n[Phase 2/2] Training SVM & NN ({config.N_SPLITS_SVM}-fold CV)...")
        skf_svm = StratifiedKFold(n_splits=config.N_SPLITS_SVM, shuffle=True, random_state=config.RANDOM_STATE)
        pbar_svm = tqdm(total=config.N_SPLITS_SVM, desc="SVM/NN")
        
        cat_cols_lin = [c for c in X_linear.columns if not pd.api.types.is_numeric_dtype(X_linear[c])]
        num_cols_lin = [c for c in X_linear.columns if c not in cat_cols_lin]

        for fold, (train_idx, val_idx) in enumerate(skf_svm.split(X_linear, y)):
            X_tr_lin, X_va_lin = X_linear.iloc[train_idx], X_linear.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            
            tr_te, va_te, te_te = oof_target_encode(X_tr_lin, X_va_lin, X_linear_test, y_tr, cat_cols_lin, seed=config.RANDOM_STATE+fold)
            X_tr_enc = pd.concat([X_tr_lin[num_cols_lin].reset_index(drop=True), tr_te.reset_index(drop=True)], axis=1)
            X_va_enc = pd.concat([X_va_lin[num_cols_lin].reset_index(drop=True), va_te.reset_index(drop=True)], axis=1)
            X_te_enc = pd.concat([X_linear_test[num_cols_lin].reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)

            # MICE
            mice = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=50, random_state=config.RANDOM_STATE),
                                   max_iter=10, random_state=config.RANDOM_STATE, initial_strategy="median")
            X_tr_imp = mice.fit_transform(X_tr_enc)
            X_va_imp = mice.transform(X_va_enc)
            X_te_imp = mice.transform(X_te_enc)

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr_imp)
            X_va_sc = scaler.transform(X_va_imp)
            X_te_sc = scaler.transform(X_te_imp)

            # SVM + CSMOUTE
            csmoute = CSMOUTE(smote_ratio=config.CSMOUTE_RATIO, random_state=config.RANDOM_STATE + fold)
            X_tr_res, y_tr_res = csmoute.fit_resample(X_tr_sc, y_tr.values)
            svm = ModelFactory.get_svm()
            svm.fit(X_tr_res, y_tr_res)
            self.oofs[3][val_idx] = svm.predict_proba(X_va_sc)
            self.test_preds[3].append(svm.predict_proba(X_te_sc))

            nn = ModelFactory.get_nn()
            nn.fit(X_tr_sc, y_tr.values)
            self.oofs[4][val_idx] = nn.predict_proba(X_va_sc)
            self.test_preds[4].append(nn.predict_proba(X_te_sc))
            
            pbar_svm.update(1)
        pbar_svm.close()

        avg_test_preds = [np.mean(fold_preds, axis=0) for fold_preds in self.test_preds]
        return self.oofs, avg_test_preds

    def train_metastack(self, oofs, y_true):
        print(f"Level 1 Stacking ({config.META_MODEL_TYPE})...")
        scaled_oofs = oofs.copy()
        scaled_oofs[3] = scaled_oofs[3] * config.SVM_STACK_WEIGHT
        X_meta_l1 = np.concatenate(scaled_oofs, axis=1)
        self.meta_model_l1.fit(X_meta_l1, y_true)
        l1_oof_probs = self.meta_model_l1.predict_proba(X_meta_l1)
        
        print("Level 2 Stacking (CatBoost)...")
        self.meta_model_l2 = CatBoostClassifier(**config.META2_PARAMS)
        self.meta_model_l2.fit(l1_oof_probs, y_true)

    def predict_stack(self, test_preds):
        scaled_test = test_preds.copy()
        scaled_test[3] = scaled_test[3] * config.SVM_STACK_WEIGHT
        X_meta_test_l1 = np.concatenate(scaled_test, axis=1)
        l1_test_probs = self.meta_model_l1.predict_proba(X_meta_test_l1)
        return self.meta_model_l2.predict_proba(l1_test_probs)

    def apply_country_blend(self, X_train, y_train, X_test, global_preds_probs, cat_features=None):
        if 'country' not in X_train.columns: return global_preds_probs
        countries = X_train['country'].unique()
        country_proba = np.zeros((len(X_test), 3))
        for ctry in countries:
            tr_mask = X_train['country'] == ctry
            te_mask = X_test['country'] == ctry
            if tr_mask.sum() < 200: continue
            X_tr_local, X_te_local = X_train[tr_mask].copy(), X_test[te_mask].copy()
            if cat_features:
                for col in cat_features:
                    X_tr_local[col] = X_tr_local[col].astype('category')
                    X_te_local[col] = X_te_local[col].astype('category')
            local_model = ModelFactory.get_catboost(cat_features=cat_features)
            local_model.fit(X_tr_local, y_train[tr_mask], verbose=0)
            country_proba[te_mask] = local_model.predict_proba(X_te_local)
        return (1 - config.COUNTRY_BLEND_WEIGHT) * global_preds_probs + config.COUNTRY_BLEND_WEIGHT * country_proba

    def evaluate_oof(self, oof_preds, y, model_name="Model"):
        y_pred = np.argmax(oof_preds, axis=1)
        print(f"\n{'='*20} OOF Evaluation: {model_name} {'='*20}")
        print(f"  Accuracy:  {accuracy_score(y, y_pred):.4f}")
        print(f"  LogLoss:   {log_loss(y, oof_preds):.4f}")
        print(f"  F1-Score:  {f1_score(y, y_pred, average='weighted'):.4f}")
        report = classification_report(y, y_pred, target_names=config.TARGET_MAPPING.keys(), output_dict=True)
        for cls in config.TARGET_MAPPING.keys():
            print(f"    - {cls:7}: F1={report[cls]['f1-score']:.4f}, Recall={report[cls]['recall']:.4f}")
        print(f"{'='*56}\n")
        return report
