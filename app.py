from flask import Flask, render_template, request
import joblib 
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("crop_model.pkl")  

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about.html")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data and convert to float
        data = [float(request.form[i]) for i in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
        
        # Ensure input is 2D for model prediction
        user_input = np.array([data])  
        
        # Predict the crop
        prediction = model.predict(user_input)[0]  # Get the first (only) prediction

        return render_template("index.html", prediction=prediction)
    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)