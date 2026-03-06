import numpy as np
import pandas as pd
import time
from . import config
from .data_processor import DataProcessor
from .pipeline_engine import PipelineEngine
from .utils import analyze_class_imbalance

def main():
    print(f"{'='*30} Zindi Upgraded Pipeline {'='*30}")
    
    # 1. Load
    processor = DataProcessor()
    train_df, test_df = processor.load_data()
    test_ids = test_df[config.ID_COL].values
    
    # 2. Process
    X_tree, y, X_linear = processor.fit_transform(train_df)
    X_tree_test, X_linear_test = processor.transform(test_df)
    
    # 3. CV
    engine = PipelineEngine()
    oofs, test_preds_list = engine.run_cross_validation(X_tree, y, X_linear, X_tree_test, X_linear_test)
    
    # 5. Eval
    for i, name in enumerate(engine.model_names):
        engine.evaluate_oof(oofs[i], y, model_name=name)
    
    # 6. Meta
    engine.train_metastack(oofs, y)
    stacked_probs = engine.predict_stack(test_preds_list)
    
    # Global Stack Eval
    scaled_oofs = oofs.copy()
    scaled_oofs[3] = scaled_oofs[3] * config.SVM_STACK_WEIGHT
    X_meta_l1 = np.concatenate(scaled_oofs, axis=1)
    l1_probs = engine.meta_model_l1.predict_proba(X_meta_l1)
    meta_oof_probs = engine.meta_model_l2.predict_proba(l1_probs)
    engine.evaluate_oof(meta_oof_probs, y, model_name="FINAL ENSEMBLE STACK")
    
    # 7. Blend
    final_probs = engine.apply_country_blend(X_tree, y, X_tree_test, stacked_probs, config.CATEGORICAL_FEATURES)
    final_preds = np.argmax(final_probs, axis=1)
    
    # 8. Submit
    inv_map = {v: k for k, v in config.TARGET_MAPPING.items()}
    test_df[config.TARGET_COL] = [inv_map[p] for p in final_preds]
    out_name = f"submission_upgraded_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    test_df[[config.ID_COL, config.TARGET_COL]].to_csv(out_name, index=False)
    print(f"\nSaved upgraded submission to {out_name}")

if __name__ == "__main__":
    main()
