import pandas as pd

from utils.model_loader import (
    load_model,
    load_scaler,
    load_feature_columns
)


def predict_customer(customer_data):
    """
    Predict whether a customer will churn.
    """

    model = load_model()
    scaler = load_scaler()
    feature_columns = load_feature_columns()

    # Create a DataFrame with all required columns
    input_df = pd.DataFrame(columns=feature_columns)

    # Add one row of zeros
    input_df.loc[0] = 0

    # Fill in the values provided
    for key, value in customer_data.items():
        if key in input_df.columns:
            input_df.loc[0, key] = value

    # Scale the input
    scaled_data = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_data)[0]

    # Prediction probability
    probability = model.predict_proba(scaled_data)[0][1]

    return prediction, probability