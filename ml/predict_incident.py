from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR /
    "trained_models" /
    "incident_risk_model.pkl"
)


model = joblib.load(MODEL_PATH)


# Example incident

incident = pd.DataFrame([
    {
        "type": "Chemical Leak",
        "severity": 8,
        "people_affected": 30,
        "chemical": "Chlorine",
        "fire": 0,
        "explosion": 0,
        "incident_impact_score": 80
    }
])


prediction = model.predict(incident)[0]

probabilities = model.predict_proba(incident)[0]

classes = model.classes_


print("\n================================")
print("INCIDENT RISK PREDICTION")
print("================================")

print("Risk Level:", prediction)

print("\nProbabilities:")

for class_name, probability in zip(
    classes,
    probabilities
):
    print(
        f"{class_name}: {probability:.3f}"
    )