import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Load and prepare dataset
# -----------------------------
df = pd.read_csv("data/cleaned_churn.csv")

# Remove the unique identifier
df = df.drop("customerID", axis=1)

# Convert categorical columns into numeric features
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

feature_columns = X.columns.tolist()


# -----------------------------
# Split the data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# -----------------------------
# Scale the features
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# Train the model
# -----------------------------
model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train_scaled, y_train)


# -----------------------------
# Evaluate the model
# -----------------------------
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

confusion = confusion_matrix(y_test, predictions)

fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = roc_auc_score(y_test, probabilities)


# -----------------------------
# Save evaluation results
# -----------------------------
model_metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "confusion_matrix": confusion,
    "fpr": fpr,
    "tpr": tpr,
    "thresholds": thresholds,
    "roc_auc": roc_auc,
}


# -----------------------------
# Save model artifacts
# -----------------------------
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")
joblib.dump(model_metrics, "models/model_metrics.pkl")


# -----------------------------
# Display results
# -----------------------------
print("Training completed successfully!")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")