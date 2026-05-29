import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

# 1. Load data
df = pd.read_csv("../data/processed/als_controls_merged.csv")

print("Original Shape:", df.shape)

# 2. Remove duplicates
df = df.drop_duplicates()

# 3. Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. SPLIT FIRST (MOST IMPORTANT STEP)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["target"],
    random_state=42
)

print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)

# 5. Split features
X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

# 6. Convert numeric
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# 7. Imputer (FIT ONLY ON TRAIN)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 8. Scaler (FIT ONLY ON TRAIN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Save datasets
train_final = pd.DataFrame(X_train)
train_final["target"] = y_train.values

test_final = pd.DataFrame(X_test)
test_final["target"] = y_test.values

os.makedirs("../data/processed", exist_ok=True)

train_final.to_csv("../data/processed/train.csv", index=False)
test_final.to_csv("../data/processed/test.csv", index=False)

# 10. Save preprocessors
os.makedirs("../models", exist_ok=True)

joblib.dump(imputer, "../models/imputer.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("DONE: Clean pipeline completed")