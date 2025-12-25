# 🛡️ Wallet Risk Scorer

> **Goal:** Instantly assess the risk profile of any cryptocurrency wallet address.

![Python](https://img.shields.io/badge/python-3.12-green.svg)
![Status](https://img.shields.io/badge/status-proto-orange.svg)

## 📖 Overview

The **Wallet Risk Scorer** is a Machine Learning-powered tool designed to analyze the transaction history of a blockchain address and output a **Risk Score (0-100)**.

Beyond just a number, it provides the **"Why"**—explainable features giving you insight into potentially malicious or suspicious activity.

## 🎯 Project Goals

- **Input:** A public wallet address (e.g., `0x123...`).
- **Output:**
  - 🚨 **Risk Score:** `0` (Safe) to `100` (High Risk).
  - 🔍 **Risk Factors:** Detailed breakdown of contributing features (e.g., "High interaction with mixers", "Wash trading patterns").

## ⚙️ How It Works

The system operates in a streamlined pipeline:

1.  **Data Ingestion**:
    - Pulls raw transaction data from external sources like **Etherscan** or **Moralis**.
    - Supports CSV exports for offline analysis.
2.  **Feature Engineering**:
    - Transforms raw logs into behavioral signals.
    - _Examples:_ Transaction frequency, unique counterparties, failed transaction ratios, and interaction with known malicious contracts.
3.  **Risk Modeling**:
    - Uses a supervised learning model (Random Forest / Logistic Regression) to classify the address.
4.  **Inference**:
    - Delivers a real-time score and significant feature importances.

## 🛠️ Tech Stack

- **Language:** Python 3.12+
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **API Integration:** Etherscan API / Moralis API

## 🚀 Getting Started

### Prerequisites

- Python 3.12+ installed.
- API Keys for Etherscan or Moralis (optional for CSV mode).

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wallet-risk-scorer.git

# Navigate to the directory
cd wallet-risk-scorer

# Install dependencies (Coming Soon)
# pip install -r requirements.txt
```

### Usage

```bash
# Run the scorer on an address
python main.py 0xYourWalletAddressHere
```

## 🔮 Roadmap

- [ ] Data Pipeline Implementation (API & CSV)
- [ ] Feature Engineering Module
- [ ] Model Training & Evaluation
- [ ] CLI Interface Development
- [ ] Web UI Dashboard

---

_Disclaimer: This tool is for educational and research purposes only. Risk scores are probabilistic estimates and should not be taken as financial or legal advice._
