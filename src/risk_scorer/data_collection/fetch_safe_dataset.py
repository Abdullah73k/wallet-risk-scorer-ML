import os
import pandas as pd
from dotenv import load_dotenv
from src.risk_scorer.config import DATA_DIR
from src.utils.fetch_data_functions import get_tx_history, calculate_metrics

load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")


def load_safe_addresses():
    # The file contains only addresses and no header
    return pd.read_csv(DATA_DIR / "safe_wallets.csv", header=None).iloc[:, 0].tolist()


safe_addresses = load_safe_addresses()
print(f"Loaded {len(safe_addresses)} safe addresses")


metrics_data = []

print("Starting data fetch...")
count = 0
for address in safe_addresses:
    # Optional: Limiting for testing
    # if count >= 5:
    #     break

    tx_vector = get_tx_history(address)
    if tx_vector is not None:
        metrics = calculate_metrics(tx_vector, address)
        if metrics:
            metrics_data.append(metrics)

    count += 1
    if count % 10 == 0:
        print(f"Processed {count}/{len(safe_addresses)}", flush=True)

    # time.sleep(0.21)  # Rate limit

# Save to CSV
if metrics_data:
    df_result = pd.DataFrame(metrics_data)
    output_path = DATA_DIR / "safe_dataset.csv"
    df_result.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df_result)} records to {output_path}")
else:
    print("No data collected. ")
