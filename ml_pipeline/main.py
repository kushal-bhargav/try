import pandas as pd
import numpy as np
from . import config
from .data_processor import DataProcessor
from .pipeline_engine import PipelineEngine
from .utils import analyze_class_imbalance

def main():
    print("Starting ML Pipeline (Zindi Gold Integrated) Execution...")
    
    # 0. Load and Preprocess
    processor = DataProcessor()
    train_df, test_df = processor.load_data()
    test_ids = test_df[config.ID_COL].values
    
    print("\nPre-processing and Feature Engineering (Dynamic Indicators)...")
    X, y, X_encoded, X_scaled = processor.fit_transform(train_df)
    X_test, X_test_encoded, X_test_scaled = processor.transform(test_df)

    # 1. Calculate Dynamic Sample Weights for rebalancing
    print("\nAnalyzing Class Imbalance and Calculating Weights...")
    class_weights = analyze_class_imbalance(y.values)
    sample_weights = np.array([class_weights[int(val)] for val in y.values])
    
    # 2. Base Model Training (Cross-Validation)
    print("\nRunning Refined Ensemble Pipeline (Zindi Gold Mode)...")
    engine = PipelineEngine()
    oofs, test_preds = engine.run_cross_validation(X, y, X_test, sample_weights=sample_weights)
    
    # --- Base Model Evaluation ---
    for i, name in enumerate(engine.model_names):
        engine.evaluate_oof(oofs[i], y.values, model_name=name)

    # 3. Meta-Stacking (2-Level)
    print("\nTraining Multi-Level Meta-Stacking...")
    engine.train_metastack(oofs, y.values)
    final_probs = engine.predict_stack(test_preds)
    
    # --- Final Stack Evaluation (OOF) ---
    print("\n[SUMMARY] Meta-Stacking Performance:")
    # Re-calculate meta-probs for full OOF evaluation
    X_meta_l1 = np.concatenate(oofs, axis=1) # Simplified for eval
    l1_probs = engine.meta_model_l1.predict_proba(X_meta_l1)
    meta_oof_probs = engine.meta_model_l2.predict_proba(l1_probs)
    engine.evaluate_oof(meta_oof_probs, y.values, model_name="FINAL ENSEMBLE STACK")

    # 4. Country Blending (Local Refinement)
    print("\nApplying Country-Specific Blending...")
    # Robustly find CatBoost features for country blend
    cat_features = X.select_dtypes(include=['category', 'object']).columns.tolist()
    cat_indices = [X.columns.get_loc(c) for c in cat_features]
    final_probs = engine.apply_country_blend(X, y.values, X_test, final_probs, cat_features=cat_indices)
    
    # 5. Save Submission
    final_preds = np.argmax(final_probs, axis=1)
    submission_df = pd.DataFrame({
        config.ID_COL: test_ids,
        config.TARGET_COL: [config.INVERSE_TARGET_MAPPING[p] for p in final_preds]
    })
    
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"\nPhase 5 Complete! Submission saved to {config.SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
