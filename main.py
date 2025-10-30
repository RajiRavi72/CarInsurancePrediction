# main.py
import pandas as pd
import os

# Set file paths
base_path = "car_insurance_data"
train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")

# Load CSV data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Display basic information
print("✅ Data loaded successfully!\n")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Quick look at the data
print("\n🔹 Train Data Head:")
print(train_df.head())

print("\n🔹 Missing Values (Train):")
print(train_df.isnull().sum())
