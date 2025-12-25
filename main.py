import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("scam_dataset.csv")

#cleaning data function
def preProcessData(df):
    df.drop(columns=["address"], inplace=True)
    df["scam"] = True
    return df

data = preProcessData(data)

x = data.drop(columns="scam")
y = data["scam"]

# one hot encode all object columns
x = pd.get_dummies(x, drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)
model.fit(x_train, y_train)

pred = model.predict(x_test)

print(accuracy_score(y_test, pred))

# TODO: need to get a dataset of safe wallets and add them to the model training