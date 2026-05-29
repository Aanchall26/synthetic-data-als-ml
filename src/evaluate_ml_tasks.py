import pandas as pd
import os

# =========================
# LOAD ALL RESULTS
# =========================
baseline = pd.read_csv("../results/baseline_results.csv")
smote = pd.read_csv("../results/smote_results.csv")
ctgan = pd.read_csv("../results/ctgan_results.csv")
tvae = pd.read_csv("../results/tvae_results.csv")
copula = pd.read_csv("../results/copulagan_results.csv")

# =========================
# COMBINE
# =========================
all_results = pd.concat(
    [baseline, smote, ctgan, tvae, copula],
    ignore_index=True
)

# =========================
# DISPLAY TABLE
# =========================
print("\n===== FINAL MODEL COMPARISON =====\n")
print(all_results)

# =========================
# BEST MODEL (BY F1 SCORE)
# =========================
best_model = all_results.loc[all_results["F1"].idxmax()]

print("\n===== BEST MODEL =====")
print(best_model)

# =========================
# SAVE FINAL REPORT
# =========================
os.makedirs("../results", exist_ok=True)

all_results.to_csv("../results/final_comparison.csv", index=False)

print("\nFinal comparison saved to ../results/final_comparison.csv")