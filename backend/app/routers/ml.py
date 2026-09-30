from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path


router = APIRouter(
    prefix="/ml",
    tags=["ML"]
)


# ============================================================
# PROJECT PATHS
# ============================================================

# Current file:
# D:\IndustrialCrisisAI\backend\app\routers\ml.py
#
# parents[0] -> routers
# parents[1] -> app
# parents[2] -> backend
# parents[3] -> IndustrialCrisisAI

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = PROJECT_ROOT / "trained_models"

RISK_MODEL_PATH = MODEL_DIR / "incident_risk_model.pkl"
SPREAD_MODEL_PATH = MODEL_DIR / "spread_model.pkl"


# ============================================================
# LOAD ML MODELS
# ============================================================

risk_model = None
spread_model = None


# ---------- Incident Risk Model ----------

try:
    if RISK_MODEL_PATH.exists():

        risk_model = joblib.load(RISK_MODEL_PATH)

        print("================================")
        print("INCIDENT RISK MODEL LOADED")
        print("================================")
        print(f"Model path: {RISK_MODEL_PATH}")

    else:

        print("WARNING: Incident risk model not found")
        print(f"Expected path: {RISK_MODEL_PATH}")

except Exception as e:

    print("ERROR loading incident risk model:")
    print(e)


# ---------- Hazard Spread Model ----------

try:
    if SPREAD_MODEL_PATH.exists():

        spread_model = joblib.load(SPREAD_MODEL_PATH)

        print("================================")
        print("SPREAD MODEL LOADED")
        print("================================")
        print(f"Model path: {SPREAD_MODEL_PATH}")

    else:

        print("WARNING: Spread model not found")
        print(f"Expected path: {SPREAD_MODEL_PATH}")

except Exception as e:

    print("ERROR loading spread model:")
    print(e)


# ============================================================
# INCIDENT RISK REQUEST
# ============================================================

class IncidentRiskRequest(BaseModel):

    type: str

    severity: int

    people_affected: int

    chemical: str | None = None

    fire: bool = False

    explosion: bool = False

    incident_impact_score: float


# ============================================================
# HAZARD SPREAD REQUEST
# ============================================================

class SpreadRequest(BaseModel):

    chemical: str

    release_type: str

    release_amount_kg: float

    release_duration_min: float

    wind_speed_kmh: float

    temperature_c: float

    source_height_m: float

    atmospheric_stability: str


# ============================================================
# INCIDENT RISK PREDICTION
# ============================================================

@router.post("/predict-risk")
def predict_risk(data: IncidentRiskRequest):

    # Check model
    if risk_model is None:

        raise HTTPException(
            status_code=500,
            detail="Incident risk model is not loaded"
        )

    try:

        # ----------------------------------------------------
        # IMPORTANT:
        # The model was trained using these exact columns.
        # Therefore we MUST send a pandas DataFrame.
        # ----------------------------------------------------

        input_data = pd.DataFrame([{

            "type": data.type,

            "severity": data.severity,

            "people_affected": data.people_affected,

            "chemical": data.chemical,

            "fire": data.fire,

            "explosion": data.explosion,

            "incident_impact_score": data.incident_impact_score

        }])

        print("Risk prediction input:")
        print(input_data)

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = risk_model.predict(input_data)[0]

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        probability_dict = {}

        if hasattr(risk_model, "predict_proba"):

            probabilities = risk_model.predict_proba(input_data)[0]

            classes = risk_model.classes_

            probability_dict = {

                str(cls): round(float(prob), 3)

                for cls, prob in zip(classes, probabilities)

            }

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            "status": "SUCCESS",

            "risk_level": str(prediction),

            "probabilities": probability_dict

        }

    except Exception as e:

        print("Risk prediction error:")
        print(e)

        raise HTTPException(

            status_code=500,

            detail=f"Risk prediction failed: {str(e)}"

        )


# ============================================================
# HAZARD SPREAD PREDICTION
# ============================================================

@router.post("/predict-spread")
def predict_spread(data: SpreadRequest):

    if spread_model is None:

        raise HTTPException(

            status_code=500,

            detail="Spread model is not loaded"

        )

    try:

        # ----------------------------------------------------
        # IMPORTANT:
        # This DataFrame should match the columns used
        # when training your spread model.
        # ----------------------------------------------------

        input_data = pd.DataFrame([{

            "chemical": data.chemical,

            "release_type": data.release_type,

            "release_amount_kg": data.release_amount_kg,

            "release_duration_min": data.release_duration_min,

            "wind_speed_kmh": data.wind_speed_kmh,

            "temperature_c": data.temperature_c,

            "source_height_m": data.source_height_m,

            "atmospheric_stability": data.atmospheric_stability

        }])

        print("Spread prediction input:")
        print(input_data)

        prediction = spread_model.predict(input_data)[0]

        return {

            "status": "SUCCESS",

            "predicted_high_risk_zone_m": round(
                float(prediction),
                2
            ),

            "input": {

                "chemical": data.chemical,

                "release_type": data.release_type,

                "release_amount_kg": data.release_amount_kg,

                "release_duration_min": data.release_duration_min,

                "wind_speed_kmh": data.wind_speed_kmh,

                "temperature_c": data.temperature_c,

                "source_height_m": data.source_height_m,

                "atmospheric_stability":
                    data.atmospheric_stability

            }

        }

    except Exception as e:

        print("Spread prediction error:")
        print(e)

        raise HTTPException(

            status_code=500,

            detail=f"Spread prediction failed: {str(e)}"

        )


# ============================================================
# MODEL STATUS
# ============================================================

@router.get("/status")
def model_status():

    return {

        "risk_model_loaded":
            risk_model is not None,

        "risk_model_path":
            str(RISK_MODEL_PATH),

        "spread_model_loaded":
            spread_model is not None,

        "spread_model_path":
            str(SPREAD_MODEL_PATH)

    }


