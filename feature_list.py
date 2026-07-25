import joblib

features = joblib.load("models/feature_columns.pkl")

print(f"Total Features: {len(features)}\n")

for feature in features:
    print(feature)
    