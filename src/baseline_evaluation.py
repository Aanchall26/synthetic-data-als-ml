import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import os

# =========================
# LOAD DATA
# =========================
train_df = pd.read_csv("../data/processed/train_reduced.csv")
test_df = pd.read_csv("../data/processed/test_reduced.csv")

X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=5,
    min_samples_split=15,
    min_samples_leaf=10,
    max_features="sqrt",
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# =========================
# METRICS
# =========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n===== Test Results (REAL) =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# =========================
# CROSS VALIDATION
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="f1"
)

print("\n===== Cross Validation (REAL) =====")
print("CV Scores:", cv_scores)
print("Mean F1:", cv_scores.mean())
print("Std Dev:", cv_scores.std())

# =========================
# LEAKAGE CHECK
# =========================
print("\n===== LEAKAGE CHECK (Shuffled Labels) =====")

y_train_shuffled = np.random.permutation(y_train)

model_shuffled = RandomForestClassifier(
    n_estimators=150,
    max_depth=5,
    min_samples_split=15,
    min_samples_leaf=10,
    max_features="sqrt",
    random_state=42
)

model_shuffled.fit(X_train, y_train_shuffled)
y_pred_shuffled = model_shuffled.predict(X_test)

print("Accuracy (Shuffled):", accuracy_score(y_test, y_pred_shuffled))
print("F1-score (Shuffled):", f1_score(y_test, y_pred_shuffled))

# =========================
# SAVE MODEL
# =========================
os.makedirs("../models", exist_ok=True)

joblib.dump(model, "../models/baseline_model.pkl")
print("\nBaseline model saved!")

# =========================
# SAVE RESULTS
# =========================
os.makedirs("../results", exist_ok=True)

df = pd.DataFrame([[
    "Baseline",
    accuracy,
    precision,
    recall,
    f1
]], columns=["Model", "Accuracy", "Precision", "Recall", "F1"])

df.to_csv("../results/baseline_results.csv", index=False)

print("Results saved!")