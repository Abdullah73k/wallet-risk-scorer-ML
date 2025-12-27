import requests
import json
import os
import time
import pandas as pd
from dotenv import load_dotenv
from src.risk_scorer.config import DATA_DIR

load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")


def load_safe_addresses():
    # The file contains only addresses and no header
    return pd.read_csv(DATA_DIR / "safe_wallets.csv", header=None).iloc[:, 0].tolist()


safe_addresses = load_safe_addresses()
print(f"Loaded {len(safe_addresses)} safe addresses")


def get_tx_history(address):
    url = f"https://api.etherscan.io/v2/api?chainid=1&module=account&action=txlist&address={address}&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey={ETHERSCAN_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()

        if data["status"] == "1":
            return data["result"]
        elif data["message"] == "No transactions found":
            return []
        else:
            print(f"Error fetching data for {address}: {data['message']}")
            return None
    except Exception as e:
        print(f"Exception for {address}: {e}")
        return None


def calculate_metrics(tx_list, address):
    if not tx_list:
        return None

    df = pd.DataFrame(tx_list)

    # Convert types
    df["timeStamp"] = pd.to_numeric(df["timeStamp"])
    df["value"] = df["value"].astype(float)
    df["isError"] = pd.to_numeric(df["isError"])

    # Basic Stats
    tx_count = len(df)
    first_tx_time = df["timeStamp"].min()
    last_tx_time = df["timeStamp"].max()
    active_days = (last_tx_time - first_tx_time) / 86400
    if active_days == 0:
        active_days = 1 / 86400  # Avoid division by zero if all tx in same second

    # Directions
    # Normalize address for comparison
    address_lower = address.lower()
    df["from"] = df["from"].str.lower()
    df["to"] = df["to"].str.lower()

    outgoing = df[df["from"] == address_lower]
    incoming = df[df["to"] == address_lower]

    in_count = len(incoming)
    out_count = len(outgoing)

    if out_count > 0:
        in_out_ratio = in_count / out_count
    else:
        in_out_ratio = in_count  # Or some high number/undefined behavior.

    # Financials (Value is in Wei, 1e18 Wei = 1 ETH)
    total_eth_in = incoming["value"].sum() / 1e18
    total_eth_out = outgoing["value"].sum() / 1e18

    # Counterparties
    unique_received_from = incoming["from"].nunique()
    unique_sent_to = outgoing["to"].nunique()
    unique_counterparties = unique_received_from + unique_sent_to

    # Advanced: Burstiness (Coefficient of Variation of inter-arrival times)
    # Sort by time just in case
    df = df.sort_values("timeStamp")
    inter_arrival_times = df["timeStamp"].diff().dropna()

    if len(inter_arrival_times) > 0:
        mean_inter_arrival = inter_arrival_times.mean()
        std_inter_arrival = inter_arrival_times.std()
        if mean_inter_arrival > 0:
            burstiness = std_inter_arrival / mean_inter_arrival
        else:
            burstiness = 0
    else:
        burstiness = 0

    # Dormancy
    current_timestamp = time.time()
    dormancy = (current_timestamp - last_tx_time) / 86400

    # Contract Interaction Ratio (input != '0x')
    contract_txs = df[df["input"] != "0x"]
    contract_interaction_ratio = len(contract_txs) / tx_count

    # Failed Tx Ratio
    failed_txs = df[df["isError"] == 1]
    failed_tx_ratio = len(failed_txs) / tx_count

    return {
        "address": address,
        "tx_count": tx_count,
        "active_days": active_days,
        "in_out_ratio": in_out_ratio,
        "burstiness": burstiness,
        "dormancy": dormancy,
        "total_eth_in": total_eth_in,
        "total_eth_out": total_eth_out,
        "unique_counterparties": unique_counterparties,
        "contract_interaction_ratio": contract_interaction_ratio,
        "failed_tx_ratio": failed_tx_ratio,
    }


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
