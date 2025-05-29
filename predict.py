import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("crop_model.pkl")

# Function to predict crop
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    # Define feature names
    feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    
    # Convert input data into a DataFrame (with column names)
    features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_names)
    
    # Make prediction
    prediction = model.predict(features)
    return prediction[0]

# Example input
if __name__ == "__main__":
    print("\n🔍 Enter soil & climate details to get crop recommendation:")
    N = float(input("Nitrogen (N): "))
    P = float(input("Phosphorus (P): "))
    K = float(input("Potassium (K): "))
    temperature = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    ph = float(input("pH Level: "))
    rainfall = float(input("Rainfall (mm): "))

    crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
    print(f"\n🌱 Recommended Crop: {crop}")