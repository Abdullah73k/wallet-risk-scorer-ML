import joblib
import pandas as pd


data = joblib.load("models/X_Address.joblib")

print(type(data))

addresses = data["Address"]

print(addresses.head())

possible_labels = [
    col for col in data.columns
    if data[col].nunique() <= 3 and col.lower() != "address"
]

print(possible_labels)