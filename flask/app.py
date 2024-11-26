from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import numpy as np
import logging

# Initialize the Flask app
app = Flask(__name__)

# Load the Keras model, scaler, and column list
model = load_model("nn_model.keras")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("column_list.pkl")

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

@app.route("/")
def home():
    return render_template("index.html")  # Render the input form

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect and process user input
        user_data = {
            "Height_(cm)": float(request.form.get("Height_(cm)", 0)),
            "Weight_(kg)": float(request.form.get("Weight_(kg)", 0)),
            "BMI": float(request.form.get("BMI", 0)),
            "Alcohol_Consumption": request.form["Alcohol_Consumption"],
            "Fruit_Consumption": int(request.form.get("Fruit_Consumption", 0)),
            "Green_Vegetables_Consumption": int(request.form.get("Green_Vegetables_Consumption", 0)),
            "FriedPotato_Consumption": int(request.form.get("FriedPotato_Consumption", 0)),
            "General_Health": request.form["General_Health"],
            "Exercise": request.form["Exercise"],
            "Sex": request.form["Sex"],
            "Age_Category": request.form["Age_Category"]
        }

        # Preprocessing
        user_df = pd.DataFrame([user_data])
        user_dummies = pd.get_dummies(user_df)
        user_dummies = user_dummies.reindex(columns=columns, fill_value=0)
        numerical_features = ["Height_(cm)", "Weight_(kg)", "BMI", 
                              "Fruit_Consumption", "Green_Vegetables_Consumption", 
                              "FriedPotato_Consumption"]
        user_dummies[numerical_features] = scaler.transform(user_dummies[numerical_features])
        final_input = user_dummies.values

        # Prediction
        prediction = model.predict(final_input)[0][0]
        threshold = 0.5
        if prediction >= threshold:
            result = f"At Risk for Cardiovascular Disease ({prediction * 100:.2f}%)"
        else:
            result = f"Not at Risk for Cardiovascular Disease ({prediction * 100:.2f}%)"

        return render_template("result.html", prediction=result)

    except Exception as e:
        # Pass the error message to the error page
        return render_template("error.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
