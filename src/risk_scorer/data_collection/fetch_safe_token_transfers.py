from utils.fetch_data_functions import calculate_token_metrics
from utils.fetch_data_functions import get_token_tx_history
import os
import time
import pandas as pd
from dotenv import load_dotenv
from src.risk_scorer.config import DATA_DIR

load_dotenv()

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


def load_safe_addresses():
    return pd.read_csv(DATA_DIR / "safe_wallets.csv", header=None).iloc[:, 0].tolist()

def main():
    safe_addresses = load_safe_addresses()
    print(f"Loaded {len(safe_addresses)} safe addresses")

    metrics_data = []

    print("Starting token data fetch...")
    count = 0

    for address in safe_addresses:
        # Rate limiting logic
        time.sleep(0.19)

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
            print(f"Processed {count}/{len(safe_addresses)}", flush=True)

        # SAFETY BREAK for this session to ensure I return results quickly
        # if count >= 20:
        #     print("Stopping after 20 for demonstration purposes.")
        #     break

    if metrics_data:
        df_result = pd.DataFrame(metrics_data)
        output_path = DATA_DIR / "safe-token-transfer-dataset.csv"
        df_result.to_csv(output_path, index=False)
        print(f"Successfully saved {len(df_result)} records to {output_path}")
    else:
        print("No data collected.")


if __name__ == "__main__":
    main()