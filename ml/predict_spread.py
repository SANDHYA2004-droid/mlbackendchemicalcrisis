from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "trained_models"
    / "spread_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# SAMPLE INCIDENT
# ============================================================

incident = pd.DataFrame(
    [
        {
            "chemical": "Chlorine",
            "release_type": "Gas",
            "release_amount_kg": 500,
            "release_duration_min": 30,
            "wind_speed_kmh": 15,
            "temperature_c": 30,
            "source_height_m": 10,
            "atmospheric_stability": "Moderate"
        }
    ]
)


# ============================================================
# PREDICT
# ============================================================

predicted_red_zone = model.predict(
    incident
)[0]


predicted_red_zone = max(
    0,
    round(predicted_red_zone, 2)
)


# ============================================================
# DISPLAY
# ============================================================

print("\n================================")
print("HAZARD SPREAD PREDICTION")
print("================================")

print(
    f"Predicted high-risk zone: "
    f"{predicted_red_zone} meters"
)

print("\nIncident:")
print(
    f"Chemical: {incident.iloc[0]['chemical']}"
)

print(
    f"Wind speed: "
    f"{incident.iloc[0]['wind_speed_kmh']} km/h"
)

print(
    f"Release amount: "
    f"{incident.iloc[0]['release_amount_kg']} kg"
)

print("\nDONE.")