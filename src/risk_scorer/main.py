from sklearn.model_selection._validation import cross_val_predict
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.risk_scorer.config import DATA_DIR

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
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

# one hot encode all object columns
# x = pd.get_dummies(x, drop_first=True)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.3, random_state=42
# )

print("created the training and test data set")

model = RandomForestClassifier(
    n_estimators=300, random_state=42, class_weight="balanced"
)
# model.fit(x_train, y_train)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"Starting Stratified K-Fold Cross-Validation (K=5)...")
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

model.fit(x, y)
joblib.dump(model, "models/scam_detection_v2.joblib")