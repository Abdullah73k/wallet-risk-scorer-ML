# 🛡️ Wallet Risk Scorer ML

> **An advanced Machine Learning pipeline for detecting malicious cryptocurrency wallets.**

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-red?style=for-the-badge&logo=xgboost&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active_Development-success?style=for-the-badge)

## 📌 Overview

The **Wallet Risk Scorer** is a data-driven security tool designed to classify blockchain addresses as "Safe" or "Scam" based on behavioral analysis. By ingesting transaction history and derived metrics, the system trains a high-performance **XGBoost Classifier** to predict the likelihood of malicious activity.

This project moves beyond simple blocklists by analyzing **behavioral features**—such as transaction frequency, activity duration, and token transfer patterns—to flag suspicious wallets that may not yet be reported.

## 🚀 Key Features

- **🤖 Advanced Gradient Boosting**: Utilizes **XGBoost** (Extreme Gradient Boosting) for superior performance on tabular risk data.
- **🎯 Automated Optimization**: Implements `RandomizedSearchCV` to automatically tune hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, etc.) for the best possible accuracy.
- **📉 Robust Validation**: Uses **Stratified K-Fold Cross-Validation (K=5)** to ensure the model generalizes well to unseen data and avoids overfitting.
- **📊 Detailed Analytics**: Generates comprehensive **Classification Reports** and **Confusion Matrices** to evaluate precision, recall, and F1-scores.
- **🧠 Feature Engineering**: Derives key behavioral signals like `transaction_frequency` (transactions per active day) to enhance model discriminability.

## 🛠️ Tech Stack

- **Language:** Python 3.12+
- **Machine Learning:** `XGBoost`, `Scikit-Learn`
- **Data Manipulation:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`
- **Data Source:** Etherscan / Moralis (via efficient CSV datasets)

## 📂 Project Structure

```bash
wallet-risk-scorer-ML/
├── data/                  # Source CSV datasets (Safe vs Scam wallets)
├── models/                # Serialized trained models (.joblib)
├── src/
│   ├── risk_scorer/
│   │   ├── data_collection/ # Scripts to fetch transactions & labels
│   │   ├── main.py          # 🚀 MASTER PIPELINE: Preprocessing -> Tuning -> Training -> Evaluation
│   │   └── config.py        # Configuration & Path definitions
│   └── utils/               # Helper functions for data fetching
├── pyproject.toml         # Project dependencies & configuration
└── README.md              # Documentation
```

## ⚡ Getting Started

### 1. Prerequisites

Ensure you have Python 3.9+ installed on your machine.

### 2. Installation

Clone the repository and install the dependencies.

```bash
git clone https://github.com/yourusername/wallet-risk-scorer-ML.git
cd wallet-risk-scorer-ML

# It is recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas scikit-learn xgboost matplotlib joblib python-dotenv requests
```

### 3. Usage

To run the full **Training & Evaluation Pipeline**:

```bash
python -m src.risk_scorer.main
```

**What happens when you run this?**

1.  **Data Ingestion**: Loads scam and safe wallet datasets.
2.  **Preprocessing**: Cleans data, handles missing values, and calculates `transaction_frequency`.
3.  **Hyperparameter Tuning**: Runs a Randomized Search to find the best XGBoost parameters.
4.  **Training**: Trains the model on the full dataset using the best parameters.
5.  **Evaluation**: Performs 5-Fold Cross-Validation and prints detailed accuracy metrics.
6.  **Serialization**: Saves the optimized model to `models/xgboost_optimized_v5.joblib`.

## 📊 Methodology

The core logic resides in `src/risk_scorer/main.py`. The pipeline follows these steps:

1.  **Labeling**: Assigns `1` to scam datasets and `0` to safe datasets.
2.  **Merging**: Combines base wallet data with token transfer data.
3.  **Hyperparameter Search**:
    ```python
    param_dist = {
        'n_estimators': [100, 300, 500, 700],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 8],
        # ... and more
    }
    ```
4.  **Model Serialization**: The final high-performing model is saved using `joblib` for future inference integration.

## 🔮 Future Roadmap

- [ ] **Real-time API**: Expose the model via a FastAPI/Flask endpoint.
- [ ] **Live Inference**: Script to fetch data for a _new_ address and predict immediately.
- [ ] **Deep Learning**: Explore LSTM/RNNs for sequential transaction analysis.
- [ ] **Explainability**: Integrate SHAP (SHapley Additive exPlanations) to explain individual risk scores.

---

_Disclaimer: This tool is for educational and research purposes. Cryptocurrency markets are volatile and high-risk. Always do your own research._
