import pandas as pd
import numpy as np
import os

# File paths
als_file = "../data/raw/als_series_matrix.txt"
control_file = "../data/raw/control_series_matrix.txt"

# Function to extract expression table
def extract_expression_data(filepath):

    with open(filepath, "r") as f:
        lines = f.readlines()

    start = None
    end = None

    for i, line in enumerate(lines):

        if line.startswith("!series_matrix_table_begin"):
            start = i + 1

        if line.startswith("!series_matrix_table_end"):
            end = i
            break

    data = lines[start:end]

    from io import StringIO

    df = pd.read_csv(
        StringIO("".join(data)),
        sep="\t"
    )

    return df

# Extract datasets
print("Loading ALS dataset...")
als_df = extract_expression_data(als_file)

print("Loading Control dataset...")
control_df = extract_expression_data(control_file)

print("ALS Shape:", als_df.shape)
print("Control Shape:", control_df.shape)

# Transpose datasets
als_df = als_df.set_index("ID_REF").T
control_df = control_df.set_index("ID_REF").T

# Reset index
als_df.reset_index(drop=True, inplace=True)
control_df.reset_index(drop=True, inplace=True)

# Add labels
als_df["target"] = 1
control_df["target"] = 0

# Merge datasets
merged_df = pd.concat([als_df, control_df], ignore_index=True)

print("Merged Shape:", merged_df.shape)

# Save processed dataset
os.makedirs("../data/processed", exist_ok=True)

merged_df.to_csv(
    "../data/processed/als_controls_merged.csv",
    index=False
)

print("Merged dataset saved successfully.")