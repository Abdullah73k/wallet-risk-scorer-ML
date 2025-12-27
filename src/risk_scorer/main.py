import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.risk_scorer.config import DATA_DIR

scam_data = pd.read_csv(DATA_DIR / "scam_dataset.csv")
scam_token_transfer_data = pd.read_csv(DATA_DIR / "scam-token-transfer-dataset.csv")


# cleaning data function
def preProcessData(df):
    df.drop(columns=["address"], inplace=True)
    df["scam"] = True
    return df


scam_data = preProcessData(scam_data)
scam_token_transfer_data = preProcessData(scam_token_transfer_data)

x_scam_data = scam_data.drop(columns="scam")
y_scam_data = scam_data["scam"]
x_scam_token_transfer_data = scam_token_transfer_data.drop(columns="scam")
y_scam_token_transfer_data = scam_token_transfer_data["scam"]


# one hot encode all object columns
x_scam_data = pd.get_dummies(x_scam_data, drop_first=True)
x_scam_token_transfer_data = pd.get_dummies(x_scam_token_transfer_data, drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(
    x_scam_data, y_scam_data, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(x_train, y_train)

pred = model.predict(x_test)

print(accuracy_score(y_test, pred))

# TODO: need to get a dataset of safe wallets and add them to the model training
