import pandas as pd

# Load reduced datasets
train_df = pd.read_csv("../data/processed/train_reduced.csv")
test_df = pd.read_csv("../data/processed/test_reduced.csv")

# Remove target
X_train = train_df.drop("target", axis=1)
X_test = test_df.drop("target", axis=1)

# Convert rows to tuples
train_rows = set(map(tuple, X_train.values))
test_rows = set(map(tuple, X_test.values))

# Find overlap
overlap = train_rows.intersection(test_rows)

print("Train Samples:", len(train_rows))
print("Test Samples :", len(test_rows))
print("Overlapping Samples:", len(overlap))