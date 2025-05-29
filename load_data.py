import pandas as pd

# Load the dataset
file_path = "data/Crop_recommendation.csv"
df = pd.read_csv(file_path)

# Display basic info
print("\nDataset Head:\n", df.head())
print("\nDataset Info:\n")
df.info()

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())