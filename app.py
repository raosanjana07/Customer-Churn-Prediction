import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from utils.prediction import predict_customer
from utils.model_loader import load_model, load_feature_columns

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# -----------------------------
# Header
# -----------------------------
st.title("📊 Customer Churn Prediction Dashboard")

st.markdown("""
Predict whether a telecom customer is likely to leave the company using a
Machine Learning model trained on historical customer data.
""")

# -----------------------------
# Dashboard Metrics
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Accuracy", "77.9%")

with col2:
    st.metric("Algorithm", "Random Forest")

with col3:
    st.metric("Features Used", len(feature_columns))

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Customer Details")

tenure = st.sidebar.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=800.0
)

predict_button = st.sidebar.button("🔍 Predict Churn")

# -----------------------------
# Prediction
# -----------------------------
if predict_button:

    customer = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    prediction, probability = predict_customer(customer)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )

    st.progress(float(probability))

st.divider()

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("📈 Top 10 Important Features")

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

fig, ax = plt.subplots(figsize=(8,5))

ax.barh(
    importance["Feature"],
    importance["Importance"]
)

ax.invert_yaxis()

st.pyplot(fig)