import pandas as pd
from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
import os

# Load reduced training data
train_df = pd.read_csv("../data/processed/train_reduced.csv")

print("Training Shape:", train_df.shape)

# Detect metadata
metadata = SingleTableMetadata()

metadata.detect_from_dataframe(train_df)

print("Metadata detected.")

# Initialize CopulaGAN
model = CopulaGANSynthesizer(
     metadata, 
     epochs=20
)

print("Training CopulaGAN...")

# Train model
model.fit(train_df)

# Create models folder
os.makedirs("../models", exist_ok=True)

# Save model
model.save("../models/copulagan.pkl")

print("CopulaGAN model saved successfully.")

copula_model = model
import joblib
import os

os.makedirs("models", exist_ok=True)
joblib.dump(copula_model, "models/copula.pkl")
