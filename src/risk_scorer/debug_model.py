import pandas as pd
from src.risk_scorer.config import DATA_DIR
from sklearn.ensemble import RandomForestClassifier

# Loading exactly what user main.py loads (step 93)
scam = pd.read_csv(DATA_DIR / "scam_dataset2.csv")
safe = pd.read_csv(DATA_DIR / "safe_dataset.csv")

scam["scam"] = 1
safe["scam"] = 0

df = pd.concat([scam, safe], ignore_index=True)
X = df.drop(columns=["address", "scam", "dormancy"])
y = df["scam"]

print(f"{'Feature':<30} {'Safe (Mean)':<15} {'Scam (Mean)':<15}")
print("-" * 65)

for col in X.columns:
    safe_mean = df[df["scam"] == 0][col].mean()
    scam_mean = df[df["scam"] == 1][col].mean()
    print(f"{col:<30} {safe_mean:<15.2f} {scam_mean:<15.2f}")

clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)
res = sorted(zip(X.columns, clf.feature_importances_), key=lambda x: x[1], reverse=True)

print("\nFeature Importance:")
for k, v in res:
    print(f"{k:<30} {v:.4f}")
