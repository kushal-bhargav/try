import pandas as pd
import numpy as np
from . import config
from .data_processor import DataProcessor
from .pipeline_engine import PipelineEngine
from .utils import analyze_class_imbalance

def save_submission(ids, probs, filename):
    """Helper to save a submission file from probabilities."""
    preds = np.argmax(probs, axis=1)
    submission_df = pd.DataFrame({
        config.ID_COL: ids,
        config.TARGET_COL: [config.INVERSE_TARGET_MAPPING[p] for p in preds]
    })
    submission_df.to_csv(filename, index=False)
    print(f"  [SAVE] Saved submission to {filename}")

def main():
    print("Starting ML Pipeline (Zindi Gold Integrated) Execution...")
    
    # 0. Load and Preprocess
    processor = DataProcessor()
    train_df, test_df = processor.load_data()
    test_ids = test_df[config.ID_COL].values
    
    # [NEW] Optional Data Sampling for fast testing
    if config.SAMPLE_FRACTION < 1.0:
        print(f"  [LOG] Sampling {config.SAMPLE_FRACTION*100}% of training data...")
        train_df = train_df.sample(frac=config.SAMPLE_FRACTION, random_state=config.RANDOM_STATE)
        print(f"  [LOG] New training size: {len(train_df)}")
    
    print("\nPre-processing and Feature Engineering (Dynamic Indicators)...")
    X, y, X_encoded, X_scaled = processor.fit_transform(train_df)
    X_test, X_test_encoded, X_test_scaled = processor.transform(test_df)
    
    # [FIX] Align X_test with X schema (names and order)
    for col in X.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X.columns]

    # 1. Calculate Dynamic Sample Weights for rebalancing
    print("\nAnalyzing Class Imbalance and Calculating Weights...")
    class_weights = analyze_class_imbalance(y.values)
    sample_weights = np.array([class_weights[int(val)] for val in y.values])
    
    # 2. Base Model Training (Cross-Validation)
    print("\nRunning Refined Ensemble Pipeline (Zindi Gold Mode)...")
    engine = PipelineEngine()
    oofs, test_preds = engine.run_cross_validation(X, y, X_test, sample_weights=sample_weights)
    
    # --- Base Model Evaluation and Individual Submissions ---
    print("\nEvaluating Base Models and Saving Individual Submissions...")
    for i, name in enumerate(engine.model_names):
        engine.evaluate_oof(oofs[i], y.values, model_name=name)
        # Save individual model submission
        clean_name = name.replace(" ", "_")
        save_submission(test_ids, test_preds[i], f"submission_{clean_name}.csv")

    # 3. Meta-Stacking (2-Level)
    print("\nTraining Multi-Level Meta-Stacking...")
    engine.train_metastack(oofs, y.values)
    stack_probs = engine.predict_stack(test_preds)
    
    # --- Final Stack Evaluation (OOF) ---
    print("\n[SUMMARY] Meta-Stacking Performance:")
    # Re-calculate meta-probs for full OOF evaluation
    X_meta_l1 = np.concatenate(oofs, axis=1) # Simplified for eval
    l1_probs = engine.meta_model_l1.predict_proba(X_meta_l1)
    meta_oof_probs = engine.meta_model_l2.predict_proba(l1_probs)
    engine.evaluate_oof(meta_oof_probs, y.values, model_name="FINAL ENSEMBLE STACK")
    
    # Save Meta-Stacking Level 2 submission
    save_submission(test_ids, stack_probs, "submission_meta_stack.csv")

    # 4. Country Blending (Local Refinement)
    print("\nApplying Country-Specific Blending...")
    # Robustly find CatBoost features for country blend
    cat_features = X.select_dtypes(include=['category', 'object']).columns.tolist()
    final_probs = engine.apply_country_blend(X, y.values, X_test, stack_probs, cat_features=cat_features)
    
    # 5. Save Final Submission
    save_submission(test_ids, final_probs, config.SUBMISSION_PATH)
    print(f"\nPhase 5 Complete! All submissions saved. Final: {config.SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
