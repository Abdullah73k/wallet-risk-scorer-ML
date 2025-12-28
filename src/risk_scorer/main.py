import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.risk_scorer.config import DATA_DIR

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

scam_data = pd.read_csv(DATA_DIR / "scam_dataset.csv")
scam_token_transfer_data = pd.read_csv(DATA_DIR / "scam-token-transfer-dataset.csv")
safe_token_transfer_data = pd.read_csv(DATA_DIR / "safe-token-transfer-dataset.csv")


# cleaning data function
def preProcessScamData(df):
    df.drop(columns=["address"], inplace=True)
    df["scam"] = True
    return df


def preProcessSafeData(df):
    df.drop(columns=["address"], inplace=True)
    df["scam"] = False
    return df


scam_data = preProcessScamData(scam_data)
scam_token_transfer_data = preProcessScamData(scam_token_transfer_data)
safe_token_transfer_data = preProcessSafeData(safe_token_transfer_data)

token_transfer_data = pd.concat(
    [scam_token_transfer_data, safe_token_transfer_data], ignore_index=True
)

print("processed data")

x_scam_data = scam_data.drop(columns="scam")
y_scam_data = scam_data["scam"]
x_token_transfer_data = token_transfer_data.drop(columns="scam")
y_token_transfer_data = token_transfer_data["scam"]


print("split data into x and y")

# one hot encode all object columns
x_scam_data = pd.get_dummies(x_scam_data, drop_first=True)
x_token_transfer_data = pd.get_dummies(x_token_transfer_data, drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(
    x_token_transfer_data, y_token_transfer_data, test_size=0.3, random_state=42
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

# TODO: need to get a dataset of safe wallets and add them to the model training
