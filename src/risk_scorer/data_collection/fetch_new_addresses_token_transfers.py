import pandas as pd
from src.risk_scorer.config import DATA_DIR
from risk_scorer.data_collection.fetch_safe_token_transfers import (
    calculate_token_metrics,
)
from risk_scorer.data_collection.fetch_safe_token_transfers import get_token_tx_history


def load_data():
    return pd.read_csv(DATA_DIR / "new_online_addresses.csv")

data = load_data()

addresses = data["Address"]
print(addresses)
# 0 = safe 1 = scam
flag = data["FLAG"]
print(flag)

metrics_data = []
count = 0

for address, current_flag in zip(addresses, flag):
    print(address)
    tx_list = get_token_tx_history(address)

    if tx_list is not None:
        if len(tx_list) > 0:
            metrics = calculate_token_metrics(tx_list, address)
            if metrics:
                metrics["flag"] = current_flag
                metrics_data.append(metrics)
        else:
            metrics_data.append(
                {
                    "address": address,
                    "token_diversity": 0,
                    "stablecoin_ratio": 0,
                    "spam_token_ratio": 0,
                    "repeated_dumps": 0,
                    "airdrop_like_behavior": 0,
                    "flag": current_flag,
                }
            )

    count += 1
    print("processed", count)
    if count % 10 == 0:
        print(f"Processed {count}/{len(addresses)}", flush=True)

if metrics_data:
    df_result = pd.DataFrame(metrics_data)
    output_path = DATA_DIR / "new-online-addresses-token-transfer-dataset.csv"
    df_result.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df_result)} records to {output_path}")
else:
    print("No data collected.")