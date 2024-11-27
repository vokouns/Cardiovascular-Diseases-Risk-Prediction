from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

# Define the headers and the first row of data for the dummy DataFrame
headers = [ 'Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption', 'General_Health_Excellent', 'General_Health_Fair', 'General_Health_Good', 'General_Health_Poor', 'General_Health_Very Good', 'Exercise_No', 'Exercise_Yes', 'Skin_Cancer_No', 'Skin_Cancer_Yes', 'Other_Cancer_No', 'Other_Cancer_Yes', 'Depression_No', 'Depression_Yes', 'Diabetes_No', 'Diabetes_No, pre-diabetes or borderline diabetes', 'Diabetes_Yes', 'Diabetes_Yes, but female told only during pregnancy', 'Arthritis_No', 'Arthritis_Yes', 'Sex_Female', 'Sex_Male', 'Age_Category_18-24', 'Age_Category_25-29', 'Age_Category_30-34', 'Age_Category_35-39', 'Age_Category_40-44', 'Age_Category_45-49', 'Age_Category_50-54', 'Age_Category_55-59', 'Age_Category_60-64', 'Age_Category_65-69', 'Age_Category_70-74', 'Age_Category_75-79', 'Age_Category_80+', 'Smoking_History_No', 'Smoking_History_Yes' ]
row = [ 122, 55, 14, 12, 12, 12, 12, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ]
starting_dummy_df = pd.DataFrame([row], columns=headers)

# Initialize the Flask app
app = Flask(__name__)

# Load the regression model, scaler, and column list
model = joblib.load("regression_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("column_list.pkl")

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
            "Age_Category": request.form["Age_Category"],
            "Smoking_History": request.form["Smoking_History"]
        }

        # Convert user input into a DataFrame
        user_df = pd.DataFrame([user_data])

        # One-hot encode user input
        user_dummies = pd.get_dummies(user_df)

        # Ensure all expected columns are present
        for col in columns:
            if col not in user_dummies.columns:
                user_dummies[col] = 0

        # Align column order
        user_dummies = user_dummies[columns]

        # Combine with starting dummy DataFrame
        combined_df = pd.concat([starting_dummy_df, user_dummies], ignore_index=True)

        # Save combined DataFrame as CSV for inspection
        if not os.path.exists("data"):
            os.makedirs("data")
        csv_path = os.path.join("data", "debug_output.csv")
        combined_df.to_csv(csv_path, index=False)
        print(f"CSV saved successfully at: {csv_path}")

        # Extract the second row for prediction
        prediction_input = combined_df.iloc[1:2]  # Ensure it's a DataFrame, not a Series

        # Align the columns with the training data
        prediction_input = prediction_input[columns]

        # Run the prediction directly on the DataFrame
        prediction = model.predict(prediction_input)[0]


        # Interpret the prediction
        if prediction == 1:
            result = "At Risk for Cardiovascular Disease"
        else:
            result = "Not at Risk for Cardiovascular Disease"

        return render_template("result.html", prediction=result)

    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template("error.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
