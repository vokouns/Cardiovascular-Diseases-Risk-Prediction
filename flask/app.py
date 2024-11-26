from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the regression model, scaler, and column list
model = joblib.load("regression_model.pkl")  # Replace with your saved regression model
scaler = joblib.load("scaler.pkl")         # Scaler used for numerical data
columns = joblib.load("column_list.pkl")   # Column structure used during training

@app.route("/")
def home():
    return render_template("index.html")  # Render the input form

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect user input
        user_data = {
            "Height_(cm)": float(request.form["Height_(cm)"]),
            "Weight_(kg)": float(request.form["Weight_(kg)"]),
            "BMI": float(request.form["BMI"]),
            "Alcohol_Consumption": request.form["Alcohol_Consumption"],
            "Fruit_Consumption": int(request.form["Fruit_Consumption"]),
            "Green_Vegetables_Consumption": int(request.form["Green_Vegetables_Consumption"]),
            "FriedPotato_Consumption": int(request.form["FriedPotato_Consumption"]),
            "General_Health": request.form["General_Health"],
            "Exercise": request.form["Exercise"],
            "Heart_Disease": request.form["Heart_Disease"],
            "Sex": request.form["Sex"],
            "Age_Category": request.form["Age_Category"]
        }

        # Convert input into a DataFrame
        user_df = pd.DataFrame([user_data])

        # One-hot encode categorical data
        user_dummies = pd.get_dummies(user_df)

        # Align columns with the training data
        user_dummies = user_dummies.reindex(columns=columns, fill_value=0)

        # Debugging: Check alignment and dtypes
        print("Processed Columns:", user_dummies.columns)
        print("Dtypes of Processed Data:")
        print(user_dummies.dtypes)

        # Use DataFrame directly
        final_input = user_dummies

        # Predict using the regression model
        prediction = model.predict(final_input)[0]

        # Render the result
        return render_template("result.html", prediction=f"Predicted Risk: {prediction:.2f}%")

    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template("error.html", error=str(e))




if __name__ == "__main__":
    app.run(debug=True)
