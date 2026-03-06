import os

# --- Paths ---
# DATA_DIR = os.path.join(os.getcwd(), 'data') # Adjust as needed
DATA_DIR = "/kaggle/input/datasets/kushalbhargav/finhealthzindi/"
TRAIN_PATH = os.path.join(DATA_DIR, 'Train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'Test.csv')
SUBMISSION_PATH = 'submission_pipeline.csv'

# --- Random Seed ---
RANDOM_STATE = 42

# --- Execution Settings ---
SAMPLE_FRACTION = 1.00 # Set to < 1.0 for fast testing (e.g., 0.1)

# --- Target Constants ---
TARGET_COL = 'Target'
ID_COL = 'ID'
TARGET_MAPPING = {'Low': 0, 'Medium': 1, 'High': 2}
INVERSE_TARGET_MAPPING = {v: k for k, v in TARGET_MAPPING.items()}

# --- Feature Groups ---
NUMERICAL_FEATURES = [
    'owner_age', 'personal_income', 'business_expenses', 
    'business_turnover', 'business_age_years', 'business_age_months'
]

# Engineered numerical features to be added
ENGINEERED_NUMERICAL = [
    'profit_margin', 'income_to_expense_ratio', 'turnover_per_month',
    'total_insurance_score', 'is_profitable'
]

CATEGORICAL_FEATURES = [
    'country', 'attitude_stable_business_environment', 'attitude_worried_shutdown', 
    'compliance_income_tax', 'perception_insurance_doesnt_cover_losses', 
    'perception_cannot_afford_insurance', 'motor_vehicle_insurance', 
    'has_mobile_money', 'current_problem_cash_flow', 'has_cellphone', 
    'owner_sex', 'offers_credit_to_customers', 'attitude_satisfied_with_achievement', 
    'has_credit_card', 'keeps_financial_records', 
    'perception_insurance_companies_dont_insure_businesses_like_yours', 
    'perception_insurance_important', 'has_insurance', 'covid_essential_service', 
    'attitude_more_successful_next_year', 'problem_sourcing_money', 
    'marketing_word_of_mouth', 'has_loan_account', 'has_internet_banking', 
    'has_debit_card', 'future_risk_theft_stock', 'medical_insurance', 
    'funeral_insurance', 'motivation_make_more_money', 
    'uses_friends_family_savings', 'uses_informal_lender', 'owner_age_bins'
]

# --- Hyperparameters# --- Stacking and Dynamic Alpha Settings ---
BETA_ALPHA = 0.999
LAMBDA_MISSING = 0.3
SVM_STACK_WEIGHT = 0.5
COUNTRY_BLEND_WEIGHT = 0.25
TE_FOLDS = 5
COUNTRY_BLEND_WEIGHT = 0.25  # 75/25 split for global/country

# LightGBM Params (Winning)
LGBM_PARAMS = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'learning_rate': 0.02,
    'num_leaves': 127,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'seed': RANDOM_STATE,
    'verbosity': -1,
}

# CatBoost Params (Winning)
CATBOOST_PARAMS = {
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    'iterations': 30000, # Matched to Gold code
    'learning_rate': 0.02,
    'depth': 10,
    'l2_leaf_reg': 6,
    'bootstrap_type': 'Bayesian',
    'bagging_temperature': 1.0,
    'random_strength': 1.0,
    'od_type': 'Iter',
    'od_wait': 1500,
    'verbose': 0,
    'task_type': 'GPU', # Use GPU
    'devices': '0',
    'random_seed': RANDOM_STATE,
}

# XGBoost Params (Winning)
XGB_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'eta': 0.02,
    'max_depth': 6,
    'min_child_weight': 3,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': RANDOM_STATE,
    'verbosity': 0
}

# SVM Params (Winning)
# class_weight is now handled by CSMOUTE in processing loop
SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 10.0,
    'gamma': 'scale',
    'probability': True,
    'random_state': RANDOM_STATE,
    'decision_function_shape': 'ovr',
}

# Neural Network Params (Baseline - remains same for now)
NN_PARAMS = {
    'hidden_layer_sizes': (128, 64),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,
    'batch_size': 'auto',
    'learning_rate': 'constant',
    'learning_rate_init': 0.001,
    'max_iter': 500,
    'random_state': RANDOM_STATE,
}

# Meta-Learner Settings (Level 1)
# Winning code used Logistic
META_MODEL_TYPE = 'logistic'
META_RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'random_state': RANDOM_STATE
}

LOGISTIC_PARAMS = {
    'max_iter': 6000,
    'random_state': RANDOM_STATE,
    'multi_class': 'multinomial',
    'solver': 'lbfgs'
}

# Level 2 Meta-Learner Settings (CatBoost)
META2_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    'random_seed': RANDOM_STATE,
    'verbose': 0
}

# --- Cross-Validation ---
N_SPLITS = 10    # Increased to match Gold code (10 for trees)
N_SPLITS_SVM = 5 # SVM is slow, keep at 5
CSMOUTE_RATIO = 0.5 # Default from Gold code
