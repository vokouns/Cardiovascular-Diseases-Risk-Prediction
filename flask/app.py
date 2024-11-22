from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the Keras model, scaler, and column list
model = load_model("nn_model.keras")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("column_list.pkl")

@app.route("/")
def home():
    return render_template("index.html")  # Render the input form

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        user_data = {
            "Height_(cm)": float(request.form["Height_(cm)"]),
            "Weight_(kg)": float(request.form["Weight_(kg)"]),
            "BMI": float(request.form["BMI"]),
            "Alcohol_Consumption": request.form["Alcohol_Consumption"],  # Handle as categorical
            "Fruit_Consumption": int(request.form["Fruit_Consumption"]),
            "Green_Vegetables_Consumption": int(request.form["Green_Vegetables_Consumption"]),
            "FriedPotato_Consumption": int(request.form["FriedPotato_Consumption"]),
            "General_Health": request.form["General_Health"],  # E.g., "Excellent"
            "Exercise": request.form["Exercise"],              # E.g., "Yes"
            "Sex": request.form["Sex"],                        # E.g., "Male"
            "Age_Category": request.form["Age_Category"]       # E.g., "18-24"
        }

        # Convert to DataFrame
        user_df = pd.DataFrame([user_data])

        # One-hot encode categorical data
        user_dummies = pd.get_dummies(user_df)

        # Align columns with the training data
        for col in columns:
            if col not in user_dummies.columns:
                user_dummies[col] = 0  # Add missing columns with default value of 0
        user_dummies = user_dummies[columns]  # Ensure column order matches

        # Scale numerical data
        numerical_features = ["Height_(cm)", "Weight_(kg)", "BMI", 
                              "Fruit_Consumption", "Green_Vegetables_Consumption", 
                              "FriedPotato_Consumption"]
        user_dummies[numerical_features] = scaler.transform(user_dummies[numerical_features])

        # Prepare final input for the model
        final_input = user_dummies.values  # Convert to NumPy array

        # Make prediction
        prediction = model.predict(final_input)[0][0]  # Assuming binary classification

        # Format the result
        result = f"{prediction * 100:.2f}% risk of cardiovascular disease"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return f"Error processing input: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
