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

scam_data = pd.read_csv(DATA_DIR / "scam_dataset2.csv")
safe_data = pd.read_csv(DATA_DIR / "safe_dataset.csv")
scam_token_transfer_data = pd.read_csv(DATA_DIR / "scam-token-transfer-dataset.csv")
safe_token_transfer_data = pd.read_csv(DATA_DIR / "safe-token-transfer-dataset.csv")


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
data = pd.concat([scam_data, safe_data], ignore_index=True)
token_transfer_data = pd.concat([scam_token_transfer_data, safe_token_transfer_data], ignore_index=True)

print(f"data length {len(data)}")

# which addresses don't overlap
addresses_data = set(data["address"])
addresses_tx = set(token_transfer_data["address"])

print("Only in token_transfer_data:", len(addresses_tx - addresses_data))
print("Only in data:", len(addresses_data - addresses_tx))

# merge the two feature sets on address, keep only shared addresses
combined_data = data.merge(token_transfer_data, on=["address", "scam"], how="inner")

# clean missing values
combined_data = combined_data.fillna(0)

print("processed data")

# drop address only if you truly don't need it for debugging later
# x = combined_data.drop(columns=["address"])
# y = combined_data["scam"]

x = data.drop(columns=["address", "scam"])
y= data["scam"]

print("split data into x and y")

# one hot encode all object columns
x = pd.get_dummies(x, drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

print("created the training and test data set")

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(x_train, y_train)

print("trained the model")

pred = model.predict(x_test)

print(accuracy_score(y_test, pred))

print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification report:\n", classification_report(y_test, pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.show()
