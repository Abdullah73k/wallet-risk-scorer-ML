import json
import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv
from src.risk_scorer.config import DATA_DIR
from src.risk_scorer.data_collection.fetch_scam_dataset import (
    calculate_metrics,
    get_tx_history,
)

load_dotenv()


def fetch_missing_data():
    # 1. Load all target addresses
    with open(DATA_DIR / "scam-address.json", "r") as f:
        all_addresses = set(json.load(f))

    # 2. Load already processed addresses
    # Using scam_dataset2.csv since that's where you saved your last run
    current_csv = DATA_DIR / "scam_dataset2.csv"
    if os.path.exists(current_csv):
        df_existing = pd.read_csv(current_csv)
        processed_addresses = set(df_existing["address"].unique())
    else:
        processed_addresses = set()

    # 3. Find missing
    missing_addresses = list(all_addresses - processed_addresses)
    print(f"Total addresses: {len(all_addresses)}")
    print(f"Already processed: {len(processed_addresses)}")
    print(f"Missing to fetch: {len(missing_addresses)}")

    if not missing_addresses:
        print("No missing addresses found!")
        return

    # 4. Fetch missing
    new_metrics_data = []
    print("Starting fetch for missing addresses...")

    for i, address in enumerate(missing_addresses):
        print(f"Fetching {i + 1}/{len(missing_addresses)}: {address}", end="\r")

        tx_vector = get_tx_history(address)
        if tx_vector is not None:
            metrics = calculate_metrics(tx_vector, address)
            if metrics:
                new_metrics_data.append(metrics)
        else:
            # If it fails again, print it out so we know
            print(f"\nFailed to fetch: {address}")

        time.sleep(0.3)  # CRITICAL: slightly higher rate limit to be safe

    # 5. Save and Merge
    if new_metrics_data:
        df_new = pd.DataFrame(new_metrics_data)

        # Append to the existing file
        # We read it again just to be sure we have the full set
        df_final = pd.concat([df_existing, df_new], ignore_index=True)

        # Save back to the main filename (scam_dataset.csv) to standardise
        output_path = DATA_DIR / "scam_dataset.csv"
        df_final.to_csv(output_path, index=False)

        print(f"\n\nSuccess! Added {len(df_new)} new records.")
        print(f"Total records in {output_path}: {len(df_final)}")
    else:
        print("\nCould not fetch any of the missing addresses.")


if __name__ == "__main__":
    fetch_missing_data()
