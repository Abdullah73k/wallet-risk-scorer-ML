from risk_scorer.config import DATA_DIR
import pandas as pd
from dotenv import load_dotenv
import os


load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

def load_data():
    return pd.read_csv(DATA_DIR / "transaction_dataset.csv")

data = load_data()

# 1 = scam 0 = not scam
new_df = data[["Address", "FLAG"]]

new_df.to_csv(DATA_DIR / "new_online_addresses.csv")