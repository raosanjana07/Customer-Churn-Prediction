import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("churn_model.pkl")

st.title("Customer Churn Prediction")

st.header("Enter Customer Details")

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=800.0
)

if st.button("Predict Churn"):

    # Create input data
    customer = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    st.write("Customer Data")
    st.write(customer)

    st.info("In the next step we'll connect this data to the AI model.")