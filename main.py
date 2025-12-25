import json
import os
from dotenv import load_dotenv

load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")


def load_scam_addresses():
    try:
        with open("scam-address.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: scam-address.json not found")
        return []


scam_addresses = load_scam_addresses()
print(f"Loaded {len(scam_addresses)} scam addresses")
