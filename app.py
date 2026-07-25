import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import plotly.express as px
import plotly.figure_factory as ff

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
from utils.model_loader import (
    load_model,
    load_scaler,
    load_feature_columns
)

model = load_model()
scaler = load_scaler()
feature_columns = load_feature_columns()
model_metrics = joblib.load("models/model_metrics.pkl")
# Load dataset
df = pd.read_csv("data/cleaned_churn.csv")

# -----------------------------
# Header
# -----------------------------
st.title("📊 Customer Churn Prediction Dashboard")

st.markdown("""
Predict whether a telecom customer is likely to leave the company using a
Machine Learning model trained on historical customer data.
""")

# -----------------------------
# Dashboard Overview
# -----------------------------
st.subheader("📊 Dashboard Overview")

total_customers = len(df)
avg_monthly = df["MonthlyCharges"].mean()
avg_tenure = df["tenure"].mean()

# Calculate churn rate
if "Churn_Yes" in df.columns:
    churn_rate = df["Churn_Yes"].mean() * 100
elif "Churn" in df.columns:
    churn_rate = (df["Churn"] == "Yes").mean() * 100
else:
    churn_rate = 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("👥 Total Customers", f"{total_customers:,}")

with col2:
    st.metric("📉 Churn Rate", f"{churn_rate:.1f}%")

with col3:
    st.metric("💰 Avg Monthly Charges", f"${avg_monthly:.2f}")

with col4:
    st.metric("📅 Avg Tenure", f"{avg_tenure:.1f} Months")

st.divider()

# Model Information
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎯 Model Accuracy", "77.9%")

with col2:
    st.metric("🤖 Algorithm", "Random Forest")

with col3:
    st.metric("📌 Features Used", len(feature_columns))

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("👤 Customer Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

partner = st.sidebar.selectbox(
    "Partner",
    ["No", "Yes"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["No", "Yes"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📞 Services")

phone = st.sidebar.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

multiple = st.sidebar.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

security = st.sidebar.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

backup = st.sidebar.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device = st.sidebar.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

support = st.sidebar.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

tv = st.sidebar.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Contract & Billing")

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Charges")

tenure = st.sidebar.slider(
    "Tenure (Months)",
    0,
    72,
    12
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

predict_button = st.sidebar.button(
    "🔍 Predict Churn",
    use_container_width=True
)

# -----------------------------
# Prediction
# -----------------------------
if predict_button:

    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    prediction, probability = predict_customer(customer)

    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )

    st.progress(float(probability))

    if probability < 0.30:

        risk_level = "Low Risk"

        st.success("🟢 Low Churn Risk")

        st.info("""
**Customer is likely to stay.**

### Recommended Action
- Maintain current service quality
- Offer loyalty rewards
- Suggest premium plans
""")

    elif probability < 0.70:

        risk_level = "Medium Risk"

        st.warning("🟡 Medium Churn Risk")

        st.info("""
**Customer may churn.**

### Recommended Action
- Send promotional offers
- Check customer satisfaction
- Offer contract renewal discounts
""")

    else:

        risk_level = "High Risk"

        st.error("🔴 High Churn Risk")

        st.info("""
**Customer is likely to churn.**

### Recommended Action
- Contact customer immediately
- Offer personalized discounts
- Escalate to retention team
""")

    report_data = {
        "Prediction": [
            "Churn" if prediction == 1 else "Stay"
        ],
        "Churn Probability": [
            f"{probability:.2%}"
        ],
        "Risk Level": [
            risk_level
        ],
        "Gender": [
            gender
        ],
        "Senior Citizen": [
            senior
        ],
        "Partner": [
            partner
        ],
        "Dependents": [
            dependents
        ],
        "Tenure": [
            tenure
        ],
        "Monthly Charges": [
            monthly_charges
        ],
        "Total Charges": [
            total_charges
        ],
        "Contract": [
            contract
        ],
        "Internet Service": [
            internet
        ],
        "Payment Method": [
            payment
        ]
    }

    report_df = pd.DataFrame(report_data)

    csv_report = report_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📄 Download Prediction Report",
        data=csv_report,
        file_name="customer_churn_prediction.csv",
        mime="text/csv"
    )
st.divider()
# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("📈 Top 10 Important Features")

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance = (
    importance
    .sort_values(by="Importance", ascending=False)
    .head(10)
)

fig = px.bar(
    importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 10 Most Important Features",
    text="Importance"
)

fig.update_layout(
    yaxis=dict(categoryorder="total ascending"),
    height=500,
    template="plotly_dark"
)

fig.update_traces(texttemplate="%{text:.3f}")

st.plotly_chart(
    fig,
    use_container_width=True,
    key="feature_importance_chart"
)


# -----------------------------
# Customer Analytics
# -----------------------------
st.divider()

st.header("📊 Customer Analytics")
col1, col2 = st.columns(2)

# -----------------------------
# Churn Distribution
# -----------------------------
with col1:

    churn_counts = (
        df["Churn"]
        .value_counts()
        .reset_index()
    )

    churn_counts.columns = ["Status", "Count"]

    fig1 = px.pie(
        churn_counts,
        names="Status",
        values="Count",
        hole=0.55,
        title="Customer Churn Distribution",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True,
        key="pie_chart"
    )

# -----------------------------
# Internet Service Distribution
# -----------------------------
with col2:

    internet_counts = (
        df["InternetService"]
        .value_counts()
        .reset_index()
    )

    internet_counts.columns = ["Service", "Count"]

    fig2 = px.bar(
        internet_counts,
        x="Service",
        y="Count",
        color="Service",
        title="Internet Service Distribution",
        template="plotly_dark",
        text="Count"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True,
        key="internet_chart"
    )
    st.divider()

st.header("📈 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Accuracy",
        f"{model_metrics['accuracy']:.2%}"
    )

    st.metric(
        "Precision",
        f"{model_metrics['precision']:.2%}"
    )

with col2:
    st.metric(
        "Recall",
        f"{model_metrics['recall']:.2%}"
    )

    st.metric(
        "F1 Score",
        f"{model_metrics['f1_score']:.2%}"
    )

with col3:
    st.metric(
        "ROC-AUC",
        f"{model_metrics['roc_auc']:.2%}"
    )
    st.divider()

st.header("📊 Confusion Matrix")

cm = model_metrics["confusion_matrix"]

fig = ff.create_annotated_heatmap(
    z=cm,
    x=["Predicted: Stay", "Predicted: Churn"],
    y=["Actual: Stay", "Actual: Churn"],
    colorscale="Blues",
    showscale=True
)

fig.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True,
    key="confusion_matrix"
)
st.divider()

st.header("📈 ROC Curve")

fpr = model_metrics["fpr"]
tpr = model_metrics["tpr"]
roc_auc = model_metrics["roc_auc"]

roc_df = pd.DataFrame({
    "False Positive Rate": fpr,
    "True Positive Rate": tpr
})

roc_fig = px.line(
    roc_df,
    x="False Positive Rate",
    y="True Positive Rate",
    title=f"ROC Curve (AUC = {roc_auc:.3f})"
)

roc_fig.add_shape(
    type="line",
    x0=0,
    y0=0,
    x1=1,
    y1=1,
    line=dict(dash="dash")
)

roc_fig.update_layout(
    template="plotly_dark",
    height=500,
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate"
)

st.plotly_chart(
    roc_fig,
    use_container_width=True,
    key="roc_curve"
)