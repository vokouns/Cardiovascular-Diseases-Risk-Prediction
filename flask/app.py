from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the regression model, scaler, and column list
model = joblib.load("regression_model.pkl")  # Replace with your saved regression model
scaler = joblib.load("scaler.pkl")           # Scaler used for numerical data
columns = joblib.load("column_list.pkl")     # Column structure used during training

@app.route("/")
def home():
    return render_template("index.html")  # Render the input form

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect user input from the form
        user_data = {
            "Height_(cm)": float(request.form["Height_(cm)"]),
            "Weight_(kg)": float(request.form["Weight_(kg)"]),
            "BMI": float(request.form["BMI"]),
            "Alcohol_Consumption": int(request.form["Alcohol_Consumption"]),
            "Fruit_Consumption": int(request.form["Fruit_Consumption"]),
            "Green_Vegetables_Consumption": int(request.form["Green_Vegetables_Consumption"]),
            "FriedPotato_Consumption": int(request.form["FriedPotato_Consumption"]),
            "General_Health": request.form["General_Health"],
            "Exercise": request.form["Exercise"],
            "Skin_Cancer": request.form["Skin_Cancer"],
            "Other_Cancer": request.form["Other_Cancer"],
            "Depression": request.form["Depression"],
            "Diabetes": request.form["Diabetes"],
            "Arthritis": request.form["Arthritis"],
            "Sex": request.form["Sex"],
            "Age_Category": request.form["Age_Category"]
        }

        # Debugging: Print raw user input
        print("Raw User Data:", user_data)

        # Step 1: Convert user_data into a DataFrame
        user_df = pd.DataFrame([user_data])
        print("Initial User DataFrame:")
        print(user_df)

        # Step 2: One-hot encode categorical data
        user_dummies = pd.get_dummies(user_df)
        print("One-Hot Encoded DataFrame:")
        print(user_dummies)

        # Step 3: Align columns with the training data
        user_dummies = user_dummies.reindex(columns=columns, fill_value=0)
        print("Aligned DataFrame Columns:")
        print(user_dummies.columns)

        # Step 4: Scale numerical features
        numerical_features = ["Height_(cm)", "Weight_(kg)", "BMI", "Alcohol_Consumption",
                              "Fruit_Consumption", "Green_Vegetables_Consumption", 
                              "FriedPotato_Consumption"]
        user_dummies[numerical_features] = scaler.transform(user_dummies[numerical_features])
        print("Scaled Numerical Features:")
        print(user_dummies[numerical_features])

        # Step 5: Use DataFrame directly for prediction
        final_input = user_dummies
        print("Final Input to Model:")
        print(final_input)

        # Predict using the regression model
        prediction = model.predict(final_input)[0]

        # Debugging: Print prediction result
        print("Prediction Result:", prediction)

        # Render the result
        return render_template("result.html", prediction=f"Predicted Risk: {prediction:.2f}%")

    except Exception as e:
        # Log error for debugging
        print(f"Error occurred: {e}")
        return render_template("error.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
