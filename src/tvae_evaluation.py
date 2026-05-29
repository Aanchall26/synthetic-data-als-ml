import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

# =========================
# LOAD DATA
# =========================
train_df = pd.read_csv("../data/processed/train_reduced.csv")
test_df = pd.read_csv("../data/processed/test_reduced.csv")

print("Original Train Shape:", train_df.shape)

# =========================
# METADATA
# =========================
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_df)

# =========================
# TRAIN TVAE
# =========================
tvae = TVAESynthesizer(metadata, epochs=300)
tvae.fit(train_df)

# =========================
# GENERATE SYNTHETIC DATA
# =========================
synthetic_data = tvae.sample(len(train_df))

print("Synthetic Shape:", synthetic_data.shape)

# =========================
# SPLIT
# =========================
X_train = synthetic_data.drop("target", axis=1)
y_train = synthetic_data["target"]

X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

# =========================
# CLASSIFIER
# =========================
clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=5,
    min_samples_split=15,
    min_samples_leaf=10,
    max_features="sqrt",
    random_state=42
)

# Train on synthetic data
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# =========================
# METRICS
# =========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n===== TVAE Results =====")
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
    clf,
    X_train,
    y_train,
    cv=cv,
    scoring="f1"
)

print("\n===== Cross Validation (TVAE) =====")
print("CV Scores:", cv_scores)
print("Mean F1:", cv_scores.mean())
print("Std Dev:", cv_scores.std())

# =========================
# SAVE MODEL
# =========================
os.makedirs("../models", exist_ok=True)

joblib.dump(clf, "../models/tvae_model.pkl")
print("\nTVAE model saved!")

# =========================
# SAVE RESULTS
# =========================
os.makedirs("../results", exist_ok=True)

results_df = pd.DataFrame([[
    "TVAE",
    accuracy,
    precision,
    recall,
    f1
]], columns=["Model", "Accuracy", "Precision", "Recall", "F1"])

results_df.to_csv("../results/tvae_results.csv", index=False)

print("TVAE results saved!")