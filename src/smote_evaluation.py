import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score

# =========================
# LOAD DATA
# =========================
train_df = pd.read_csv("../data/processed/train_reduced.csv")
test_df = pd.read_csv("../data/processed/test_reduced.csv")

X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

print("Original Train Shape:", X_train.shape)
print("Original Class Distribution:\n", y_train.value_counts())

# =========================
# APPLY SMOTE (TRAIN ONLY)
# =========================
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE Train Shape:", X_train_smote.shape)
print("After SMOTE Class Distribution:\n", y_train_smote.value_counts())

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
model.fit(X_train_smote, y_train_smote)

# Predict
y_pred = model.predict(X_test)

# =========================
# METRICS
# =========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n===== SMOTE FINAL RESULTS =====")
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
    X_train_smote,
    y_train_smote,
    cv=cv,
    scoring="f1"
)

print("\n===== Cross Validation (SMOTE) =====")
print("CV Scores:", cv_scores)
print("Mean F1:", cv_scores.mean())
print("Std Dev:", cv_scores.std())

# =========================
# SAVE MODEL
# =========================
os.makedirs("../models", exist_ok=True)

joblib.dump(model, "../models/smote_model.pkl")
print("\nSMOTE model saved to ../models/smote_model.pkl")

# =========================
# SAVE RESULTS (OPTIONAL BUT GOOD)
# =========================
os.makedirs("../results", exist_ok=True)

df = pd.DataFrame([[
    "SMOTE",
    accuracy,
    precision,
    recall,
    f1
]], columns=["Model", "Accuracy", "Precision", "Recall", "F1"])

df.to_csv("../results/smote_results.csv", index=False)

print("Results saved to ../results/smote_results.csv")