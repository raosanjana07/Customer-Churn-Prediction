import joblib


def load_model():
    """Load the trained model."""
    return joblib.load("models/churn_model.pkl")


def load_scaler():
    """Load the saved scaler."""
    return joblib.load("models/scaler.pkl")


def load_feature_columns():
    """Load the feature column names."""
    return joblib.load("models/feature_columns.pkl")