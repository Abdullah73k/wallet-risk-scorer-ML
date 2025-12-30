import pandas as pd
import os
from dotenv import load_dotenv
from src.risk_scorer.config import DATA_DIR
from src.utils.fetch_data_functions import get_tx_history, calculate_metrics

load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")


def load_addresses():
    return pd.read_csv(DATA_DIR / "new_online_addresses.csv")


data = load_addresses()

addresses = data["Address"]
# 0 = safe 1 = scam
flag = data["FLAG"]


metrics_data = []

print("Starting data fetch...")
count = 0
for address, current_flag in zip(addresses, flag):
    tx_vector = get_tx_history(address)
    if tx_vector is not None:
        metrics = calculate_metrics(tx_vector, address)
        if metrics:
            metrics["flag"] = current_flag
            metrics_data.append(metrics)

    count += 1
    if count % 10 == 0:
        print(f"Processed {count}/{len(addresses)}", flush=True)

    # time.sleep(0.21)  # Rate limit

# Save to CSV
if metrics_data:
    df_result = pd.DataFrame(metrics_data)
    output_path = DATA_DIR / "new_addresses_dataset.csv"
    df_result.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df_result)} records to {output_path}")
else:
    print("No data collected. ")
