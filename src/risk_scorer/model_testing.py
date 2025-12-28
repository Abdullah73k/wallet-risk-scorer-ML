from risk_scorer.data_collection.fetch_safe_token_transfers import (
    calculate_token_metrics,
)
from risk_scorer.data_collection.fetch_safe_token_transfers import get_token_tx_history
from utils.fetch_data_functions import calculate_metrics
from utils.fetch_data_functions import get_tx_history
import joblib
import pandas as pd

test_wallet = "0xEB9182fF32652249CbCBC2A4888798Ad673EDfbc"

model = joblib.load("models/scam_detection_v1.joblib")

tx_history = get_tx_history(test_wallet)
metrics = calculate_metrics(tx_history, test_wallet)

token_tx_history = get_token_tx_history(test_wallet)
token_metrics = calculate_token_metrics(token_tx_history, test_wallet)

token_data = pd.DataFrame([token_metrics])
metrics_data = pd.DataFrame([metrics])


data = token_data.merge(
    metrics_data, on="address", how="outer", suffixes=("_base", "_tx")
)

data_for_prediction = data.drop(columns=["address", "dormancy"], errors='ignore')
data_for_prediction = data_for_prediction[model.feature_names_in_]
result = model.predict(data_for_prediction)

print(result)
