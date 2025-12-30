from sklearn.model_selection._validation import cross_val_predict
import joblib
import pandas as pd

from xgboost import XGBClassifier
from src.risk_scorer.config import DATA_DIR

from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
import numpy as np

scam_data = pd.read_csv(DATA_DIR / "scam_dataset2.csv")
safe_data = pd.read_csv(DATA_DIR / "safe_dataset.csv")
scam_token_transfer_data = pd.read_csv(DATA_DIR / "scam-token-transfer-dataset.csv")
safe_token_transfer_data = pd.read_csv(DATA_DIR / "safe-token-transfer-dataset.csv")

new_data = pd.read_csv(DATA_DIR / "new_addresses_dataset.csv")
new_token_transfer_data = pd.read_csv(DATA_DIR / "new-online-addresses-token-transfer-dataset.csv")

# 1 = true/is a a scam
# 0 = false/is not a scam
# cleaning data function
def preProcessScamData(df):
    df = df.copy()
    df["scam"] = 1
    return df


def preProcessSafeData(df):
    df = df.copy()
    df["scam"] = 0
    return df


# label data
scam_data = preProcessScamData(scam_data)
safe_data = preProcessSafeData(safe_data)
scam_token_transfer_data = preProcessScamData(scam_token_transfer_data)
safe_token_transfer_data = preProcessSafeData(safe_token_transfer_data)

# combine within each feature set
data = pd.concat([scam_data, safe_data, new_data], ignore_index=True)
token_transfer_data = pd.concat(
    [scam_token_transfer_data, safe_token_transfer_data, new_token_transfer_data], ignore_index=True
)

print(f"data length {len(data)}")

# Make sure (address, scam) is unique in each dataset first
data_u = data.drop_duplicates(subset=["address", "scam"])
tx_u = token_transfer_data.drop_duplicates(subset=["address", "scam"])

# Keep ALL addresses from both datasets
combined_data = data_u.merge(
    tx_u, on=["address", "scam"], how="outer", suffixes=("_base", "_tx")
)

# If you want "empty" values, keep NaN (don’t fill here)
# combined_data stays with NaNs where a dataset is missing
print("combined length:", len(combined_data))

# clean missing values
# combined_data = combined_data.fillna(0)

print("processed data")

# drop address only if you truly don't need it for debugging later
x = combined_data.drop(columns=["address", "scam", "dormancy"])
y = combined_data["scam"]


print("split data into x and y")


print("created the training and test data set")


xgb = XGBClassifier(
    random_state=42, 
    eval_metric='logloss'
)
# 2. Define the range of parameters to test
param_dist = {
    'n_estimators': [100, 300, 500, 700],        # Number of trees
    'learning_rate': [0.01, 0.05, 0.1, 0.2],     # Speed of learning (lower is slower but more precise)
    'max_depth': [3, 4, 5, 6, 8],                # Depth of each tree
    'subsample': [0.7, 0.8, 0.9, 1.0],           # % of rows to use for each tree (prevents overfitting)
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],    # % of columns to use for each tree
    'scale_pos_weight': [1, 2, 5]                # Weight for "Scam" class if it's rare (Imbalanced data)
}
# 3. Setup the search
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,           # Try 20 different random combinations
    scoring='accuracy',  # Or 'f1', 'roc_auc'
    n_jobs=-1,           # Use all CPU cores
    cv=5,                # 5-Fold Cross Validation for each try
    verbose=1,
    random_state=42
)
print("Starting Hyperparameter Tuning... this might take a minute.")
random_search.fit(x, y)
# 4. Get the best results
print(f"Best Parameters Found: {random_search.best_params_}")
print(f"Best Accuracy: {random_search.best_score_ * 100:.2f}%")
# 5. Use the best model for your final save
model = random_search.best_estimator_

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Starting Stratified K-Fold Cross-Validation (K=5)...")
# 3. Run the Evaluation
# This automatically splits x/y, trains, predicts, and scores 5 times.
cv_scores = cross_val_score(model, x, y, cv=skf, scoring='accuracy')
# 4. Print Robust Results
print("------------------------------------------------")
print(f"Individual Fold Accuracies: {cv_scores}")
print(f"Mean Accuracy:      {np.mean(cv_scores) * 100:.2f}%")
print(f"Standard Deviation: +/-{np.std(cv_scores) * 100:.2f}%")
print("------------------------------------------------")

print("trained the model")

y_pred = cross_val_predict(model, x, y, cv=skf)
print("\nClassification Report (Aggregated across Folds):")

print(classification_report(y, y_pred, digits=4))

print("\nConfusion Matrix:")

ConfusionMatrixDisplay.from_predictions(y, y_pred)

plt.title("Confusion Matrix (Cross-Validation)")
plt.show()

joblib.dump(model, "models/xgboost_optimized_v4.joblib")
