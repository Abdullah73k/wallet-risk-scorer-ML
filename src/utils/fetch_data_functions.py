import requests
import time
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
ETHERSCAN_API_KEY_2 = os.getenv("ETHERSCAN_API_KEY_2")

# Common Stablecoin Addresses (Mainnet)
STABLECOIN_ADDRESSES = {
    "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
    "0x6b175474e89094c44da98b954eedeac495271d0f",  # DAI
}

# Simple keywords often found in spam tokens (very basic heuristic)
SPAM_KEYWORDS = [
    "visit",
    "http",
    "www.",
    ".com",
    "claim",
    "prize",
    "airdrop",
    "access",
    "reward",
]


def get_tx_history(address):
    url = f"https://api.etherscan.io/v2/api?chainid=1&module=account&action=txlist&address={address}&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey={ETHERSCAN_API_KEY_2}"
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

    # Time-based patterns
    df["datetime"] = pd.to_datetime(df["timeStamp"], unit="s")
    df["hour"] = df["datetime"].dt.hour

    # Unique Active Hours (Variability)
    unique_active_hours = df["hour"].nunique()

    # Time Concentration
    if tx_count > 0:
        most_active_hour_count = df["hour"].value_counts().max()
        time_concentration = most_active_hour_count / tx_count
    else:
        time_concentration = 0

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
        "unique_active_hours": unique_active_hours,
        "time_concentration": time_concentration,
    }


def get_token_tx_history(address):
    # Etherscan endpoint for ERC20 token transfer events
    url = f"https://api.etherscan.io/v2/api?chainid=1&module=account&action=tokentx&address={address}&startblock=0&endblock=99999999&page=1&offset=10000&sort=asc&apikey={ETHERSCAN_API_KEY}"
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


def calculate_token_metrics(tx_list, address):
    if not tx_list:
        return None

    df = pd.DataFrame(tx_list)

    # Normalize address
    address_lower = address.lower()
    df["to"] = df["to"].str.lower()
    df["from"] = df["from"].str.lower()
    df["contractAddress"] = df["contractAddress"].str.lower()

    # Directions
    outgoing = df[df["from"] == address_lower]

    total_txs = len(df)
    if total_txs == 0:
        return None

    # 1. Token Diversity
    unique_tokens = df["contractAddress"].nunique()

    # 2. Stablecoin Ratio
    stablecoin_txs = df[df["contractAddress"].isin(STABLECOIN_ADDRESSES)]
    stablecoin_ratio = len(stablecoin_txs) / total_txs

    # 3. Spam Token Ratio
    def is_spam(row):
        name = str(row.get("tokenName", "")).lower()
        symbol = str(row.get("tokenSymbol", "")).lower()
        for kw in SPAM_KEYWORDS:
            if kw in name or kw in symbol:
                return True
        return False

    spam_count = df.apply(is_spam, axis=1).sum()
    spam_token_ratio = spam_count / total_txs

    # 4. Repeated Dumps
    # Max number of outgoing transactions for a single token
    if len(outgoing) > 0:
        token_counts = outgoing["contractAddress"].value_counts()
        repeated_dumps = token_counts.max()
    else:
        repeated_dumps = 0

    # 5. Airdrop-like behavior
    # Count of unique outgoing recipients
    if len(outgoing) > 0:
        unique_recipients = outgoing["to"].nunique()
    else:
        unique_recipients = 0

    return {
        "address": address,
        "token_diversity": unique_tokens,
        "stablecoin_ratio": stablecoin_ratio,
        "spam_token_ratio": spam_token_ratio,
        "repeated_dumps": repeated_dumps,
        "airdrop_like_behavior": unique_recipients,
    }


def eth_blockNumber():
    url = f"https://api.etherscan.io/v2/api?apiKey={ETHERSCAN_API_KEY}&chainid=1&module=proxy&action=eth_blockNumber"
    response = requests.get(url)
    return response.json()["result"]
