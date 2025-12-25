import requests
import json
import os
import time
import pandas as pd
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

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


def load_scam_addresses():
    try:
        with open("scam-address.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: scam-address.json not found")
        return []


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
    incoming = df[df["to"] == address_lower]

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


def main():
    scam_addresses = load_scam_addresses()
    print(f"Loaded {len(scam_addresses)} scam addresses")

    metrics_data = []

    print("Starting token data fetch...")
    count = 0

    for address in scam_addresses:
        # Rate limiting logic
        time.sleep(0.25)

        tx_list = get_token_tx_history(address)

        if tx_list is not None:
            # tx_list is empty list if no txs, but not None
            if len(tx_list) > 0:
                metrics = calculate_token_metrics(tx_list, address)
                if metrics:
                    metrics_data.append(metrics)
            else:
                # No token txs, still might want to record zeros?
                # Usually if no token txs, these metrics are 0.
                metrics_data.append(
                    {
                        "address": address,
                        "token_diversity": 0,
                        "stablecoin_ratio": 0,
                        "spam_token_ratio": 0,
                        "repeated_dumps": 0,
                        "airdrop_like_behavior": 0,
                    }
                )

        count += 1
        if count % 10 == 0:
            print(f"Processed {count}/{len(scam_addresses)}", flush=True)

        # SAFETY BREAK for this session to ensure I return results quickly
        # if count >= 20:
        #     print("Stopping after 20 for demonstration purposes.")
        #     break

    if metrics_data:
        df_result = pd.DataFrame(metrics_data)
        df_result.to_csv("scam-token-transfer-dataset.csv", index=False)
        print(
            f"Successfully saved {len(df_result)} records to scam-token-transfer-dataset.csv"
        )
    else:
        print("No data collected.")


if __name__ == "__main__":
    main()
