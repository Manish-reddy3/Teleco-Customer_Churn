from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
with open("Teleco.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    customer = pd.DataFrame([{
        "gender": request.form["gender"],
        "SeniorCitizen": int(request.form["SeniorCitizen"]),
        "Partner": request.form["Partner"],
        "Dependents": request.form["Dependents"],
        "tenure": int(request.form["tenure"]),
        "PhoneService": request.form["PhoneService"],
        "MultipleLines": request.form["MultipleLines"],
        "InternetService": request.form["InternetService"],
        "OnlineSecurity": request.form["OnlineSecurity"],
        "OnlineBackup": request.form["OnlineBackup"],
        "DeviceProtection": request.form["DeviceProtection"],
        "TechSupport": request.form["TechSupport"],
        "StreamingTV": request.form["StreamingTV"],
        "StreamingMovies": request.form["StreamingMovies"],
        "Contract": request.form["Contract"],
        "PaperlessBilling": request.form["PaperlessBilling"],
        "PaymentMethod": request.form["PaymentMethod"],
        "MonthlyCharges": float(request.form["MonthlyCharges"]),
        "TotalCharges": float(request.form["TotalCharges"])
    }])

    prediction = model.predict(customer)[0]

    # Probability of churn
    probability = model.predict_proba(customer)[0][1]

    # Your chosen threshold
    threshold = 0.30

    if probability >= threshold:
        result = "Customer is likely to CHURN"
        status = "churn"
    else:
        result = "Customer is likely to STAY"
        status = "stay"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability * 100, 2),
        status=status
    )


if __name__ == "__main__":
    app.run(debug=True)