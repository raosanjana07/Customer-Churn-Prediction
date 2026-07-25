import pandas as pd

from utils.model_loader import (
    load_feature_columns,
    load_model,
    load_scaler,
)


def predict_customer(customer_data):
    model = load_model()
    scaler = load_scaler()
    feature_columns = load_feature_columns()

    input_df = pd.DataFrame(
        data=[[0.0] * len(feature_columns)],
        columns=feature_columns,
        dtype="float64",
    )

    # Numerical features
    input_df.loc[0, "SeniorCitizen"] = float(customer_data["SeniorCitizen"])
    input_df.loc[0, "tenure"] = float(customer_data["tenure"])
    input_df.loc[0, "MonthlyCharges"] = float(customer_data["MonthlyCharges"])
    input_df.loc[0, "TotalCharges"] = float(customer_data["TotalCharges"])

    # One-hot encoded categorical features
    mappings = {
        "gender": ("Male", "gender_Male"),
        "Partner": ("Yes", "Partner_Yes"),
        "Dependents": ("Yes", "Dependents_Yes"),
        "PhoneService": ("Yes", "PhoneService_Yes"),
        "PaperlessBilling": ("Yes", "PaperlessBilling_Yes"),
    }

    for key, (value, column) in mappings.items():
        if customer_data[key] == value:
            input_df.loc[0, column] = 1.0

    category_mappings = {
        "MultipleLines": {
            "No phone service": "MultipleLines_No phone service",
            "Yes": "MultipleLines_Yes",
        },
        "InternetService": {
            "Fiber optic": "InternetService_Fiber optic",
            "No": "InternetService_No",
        },
        "OnlineSecurity": {
            "No internet service": "OnlineSecurity_No internet service",
            "Yes": "OnlineSecurity_Yes",
        },
        "OnlineBackup": {
            "No internet service": "OnlineBackup_No internet service",
            "Yes": "OnlineBackup_Yes",
        },
        "DeviceProtection": {
            "No internet service": "DeviceProtection_No internet service",
            "Yes": "DeviceProtection_Yes",
        },
        "TechSupport": {
            "No internet service": "TechSupport_No internet service",
            "Yes": "TechSupport_Yes",
        },
        "StreamingTV": {
            "No internet service": "StreamingTV_No internet service",
            "Yes": "StreamingTV_Yes",
        },
        "StreamingMovies": {
            "No internet service": "StreamingMovies_No internet service",
            "Yes": "StreamingMovies_Yes",
        },
        "Contract": {
            "One year": "Contract_One year",
            "Two year": "Contract_Two year",
        },
        "PaymentMethod": {
            "Credit card (automatic)": "PaymentMethod_Credit card (automatic)",
            "Electronic check": "PaymentMethod_Electronic check",
            "Mailed check": "PaymentMethod_Mailed check",
        },
    }

    for feature, options in category_mappings.items():
        value = customer_data[feature]
        if value in options:
            input_df.loc[0, options[value]] = 1.0

    scaled_input = scaler.transform(input_df)

    prediction = int(model.predict(scaled_input)[0])
    probability = float(model.predict_proba(scaled_input)[0][1])

    return prediction, probability