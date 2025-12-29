from risk_scorer.config import DATA_DIR
from risk_scorer.data_collection.fetch_safe_token_transfers import (
    calculate_token_metrics,
)
from risk_scorer.data_collection.fetch_safe_token_transfers import get_token_tx_history
from utils.fetch_data_functions import calculate_metrics
from utils.fetch_data_functions import get_tx_history
import joblib
import pandas as pd


def load_safe_addresses():
    return pd.read_csv(DATA_DIR / "test_wallets.csv", header=None).iloc[:, 0].tolist()


test_wallets = load_safe_addresses()
results = []

model = joblib.load("models/scam_detection_v1.joblib")


for index, wallet in enumerate(test_wallets):
    if index == 5:
        break
    tx_history = get_tx_history(wallet)
    metrics = calculate_metrics(tx_history, wallet)

    token_tx_history = get_token_tx_history(wallet)
    token_metrics = calculate_token_metrics(token_tx_history, wallet)

    if metrics is None or token_metrics is None:
        print(f"Skipping {wallet}: Insufficient data (No transactions found)")
        continue

    token_data = pd.DataFrame([token_metrics])
    metrics_data = pd.DataFrame([metrics])

    data = token_data.merge(
        metrics_data, on="address", how="outer", suffixes=("_base", "_tx")
    )

    data_for_prediction = data.drop(columns=["address", "dormancy"], errors="ignore")
    data_for_prediction = data_for_prediction[model.feature_names_in_]
    result = model.predict(data_for_prediction)

    if result == 0:
        result = False
    else:
        result = True
    results.append({"address": wallet, "isScam": result})
    print(f"address: {wallet}")
    print(f"is scam: {result}")
