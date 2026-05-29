import pandas as pd
import os

# =========================
# LOAD RESULTS
# =========================
baseline = pd.read_csv("../results/baseline_results.csv")
smote = pd.read_csv("../results/smote_results.csv")
ctgan = pd.read_csv("../results/ctgan_results.csv")
tvae = pd.read_csv("../results/tvae_results.csv")
copula = pd.read_csv("../results/copulagan_results.csv")

# =========================
# MERGE ALL
# =========================
all_results = pd.concat(
    [baseline, smote, ctgan, tvae, copula],
    ignore_index=True
)

# =========================
# SHOW RESULTS
# =========================
print("\n===== FINAL MODEL COMPARISON =====\n")
print(all_results)

# =========================
# BEST MODEL
# =========================
best = all_results.loc[all_results["F1"].idxmax()]

print("\n===== BEST MODEL =====")
print(best)

# =========================
# SAVE FINAL OUTPUT
# =========================
os.makedirs("../results", exist_ok=True)

all_results.to_csv("../results/final_comparison.csv", index=False)

print("\nSaved: final_comparison.csv")