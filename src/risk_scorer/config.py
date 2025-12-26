import os
from pathlib import Path

# Get the project root directory (2 levels up from this file)
# src/risk_scorer/config.py -> src/risk_scorer -> src -> ROOT
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Ensure data dir exists
os.makedirs(DATA_DIR, exist_ok=True)
