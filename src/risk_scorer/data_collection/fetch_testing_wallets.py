from utils.fetch_data_functions import eth_blockNumber
from src.risk_scorer.config import DATA_DIR
import pandas as pd
from dotenv import load_dotenv
import os
import requests
import time

load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
TARGET = 100000
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
wallets = []
seen = set()


latest_hex = eth_blockNumber()
latest_block = int(latest_hex, 16)

print(latest_hex)
print(latest_block)


while len(wallets) < TARGET:
    block_transaction_url = f"https://api.etherscan.io/v2/api?chainid=1&module=proxy&action=eth_getBlockByNumber&tag={hex(latest_block)}&boolean=true&apikey={ETHERSCAN_API_KEY}"
    block_data = requests.get(block_transaction_url).json()["result"]
    for tx in block_data["transactions"]:
        addr = tx["from"].lower()
        if addr not in seen and len(addr) == 42 and addr != ZERO_ADDRESS:
            wallets.append(addr)
            seen.add(addr)
            if len(wallets) >= TARGET:
                break
    latest_block -= 1
    time.sleep(0.21)
    print(f"Processed {latest_block}")

df_result = pd.DataFrame(wallets)
output_path = DATA_DIR / "test_wallets.csv"
df_result.to_csv(output_path, index=False)
print(f"Successfully saved {len(df_result)} records to {output_path}")