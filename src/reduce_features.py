import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os

# Load datasets
train_df = pd.read_csv("../data/processed/train.csv")
test_df = pd.read_csv("../data/processed/test.csv")

# Separate features and targets
X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

print("Original Features:", X_train.shape[1])

# Select top 500 features
selector = SelectKBest(
    score_func=f_classif,
    k=150
)

# Fit on training data
X_train_reduced = selector.fit_transform(X_train, y_train)

# Transform test data
X_test_reduced = selector.transform(X_test)

print("Reduced Features:", X_train_reduced.shape[1])

# Convert back to DataFrames
train_reduced = pd.DataFrame(X_train_reduced)
test_reduced = pd.DataFrame(X_test_reduced)

# Add target column back
train_reduced["target"] = y_train.values
test_reduced["target"] = y_test.values

# Save datasets
train_reduced.to_csv(
    "../data/processed/train_reduced.csv",
    index=False
)

test_reduced.to_csv(
    "../data/processed/test_reduced.csv",
    index=False
)

# Save selector
os.makedirs("../models", exist_ok=True)

joblib.dump(
    selector,
    "../models/feature_selector.pkl"
)

print("Reduced datasets saved successfully.")