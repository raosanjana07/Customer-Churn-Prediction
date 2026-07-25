import joblib
import streamlit as st


@st.cache_resource
def load_model():
    """Load the trained model only once."""
    return joblib.load("models/churn_model.pkl")


@st.cache_resource
def load_scaler():
    """Load the scaler only once."""
    return joblib.load("models/scaler.pkl")


@st.cache_resource
def load_feature_columns():
    """Load feature names only once."""
    return joblib.load("models/feature_columns.pkl")