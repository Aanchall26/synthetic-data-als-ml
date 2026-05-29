import pandas as pd
from sdv.single_table import CopulaGANSynthesizer
import os

# Load trained model
model = CopulaGANSynthesizer.load(
    "../models/copulagan.pkl"
)

print("Model loaded successfully.")

# Generate synthetic samples
synthetic_data = model.sample(num_rows=500)

print("Synthetic Data Shape:", synthetic_data.shape)

# Create folder
os.makedirs("../data/synthetic", exist_ok=True)

# Save synthetic dataset
synthetic_data.to_csv(
    "../data/synthetic/copulagan_synthetic.csv",
    index=False
)

print("Synthetic dataset saved successfully.")