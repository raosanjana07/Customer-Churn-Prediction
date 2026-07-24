import joblib

# Load the saved feature names
feature_columns = joblib.load("models/feature_columns.pkl")

print("Total Features:", len(feature_columns))

print("\nFirst 20 Features:\n")

for feature in feature_columns[:20]:
    print(feature)