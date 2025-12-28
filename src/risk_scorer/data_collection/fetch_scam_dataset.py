import json
import os
import pandas as pd
from dotenv import load_dotenv
from src.risk_scorer.config import DATA_DIR
from src.utils.fetch_data_functions import get_tx_history, calculate_metrics


load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")


def load_scam_addresses():
    try:
        with open(DATA_DIR / "scam-address.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: scam-address.json not found")
        return []


scam_addresses = load_scam_addresses()
print(f"Loaded {len(scam_addresses)} scam addresses")


metrics_data = []

# Rate limiting: Etherscan free tier is ~5 calls/sec. being safe with 0.25s sleep (4 calls/sec)

print("Starting data fetch...")
count = 0
for address in scam_addresses:
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
        print(f"Processed {count}/{len(scam_addresses)}", flush=True)

    # time.sleep(0.25)  # Rate limit ss
# Save to CSV
if metrics_data:
    df_result = pd.DataFrame(metrics_data)
    output_path = DATA_DIR / "scam_dataset2.csv"
    df_result.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df_result)} records to {output_path}")
else:
    print("No data collected. ")
