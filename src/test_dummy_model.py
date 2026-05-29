import pandas as pd
from sklearn.dummy import DummyClassifier

# Load data
train_df = pd.read_csv("../data/processed/train.csv")
test_df = pd.read_csv("../data/processed/test.csv")

# Split features and target
X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

# Create dummy model (baseline - no learning)
model = DummyClassifier(strategy="most_frequent")

# Train
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)

print("Dummy Model Accuracy:", score)