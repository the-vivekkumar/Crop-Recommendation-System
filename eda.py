import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "data/Crop_recommendation.csv"
df = pd.read_csv(file_path)

# Print summary statistics
print("\nSummary Statistics:\n", df.describe())

# Drop the categorical 'label' column before correlation
df_numeric = df.drop(columns=["label"])  # Remove 'label' column

# Plot Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()