import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving the model

# Load dataset
file_path = "data/Crop_recommendation.csv"
df = pd.read_csv(file_path)

# Splitting Features and Labels
X = df.drop(columns=["label"])  # Features (N, P, K, temperature, etc.)
y = df["label"]  # Target (Crop Type)

# Splitting dataset into training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "crop_model.pkl")
print("\n✅ Model saved as 'crop_model.pkl'")