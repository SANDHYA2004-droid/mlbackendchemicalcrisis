from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "incidents_clean.csv"

MODEL_DIR = BASE_DIR / "trained_models"

MODEL_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("Loading incident dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ---------------------------------------------------------
# CREATE TARGET
# ---------------------------------------------------------
#
# The dataset has severity.
# We convert severity into operational risk classes.
#
# 1-2  -> LOW
# 3-4  -> MEDIUM
# 5-7  -> HIGH
# 8-10 -> CRITICAL
#
# ---------------------------------------------------------

def create_risk_level(severity):

    if severity >= 8:
        return "CRITICAL"

    elif severity >= 5:
        return "HIGH"

    elif severity >= 3:
        return "MEDIUM"

    else:
        return "LOW"


df["risk_level"] = df["severity"].apply(create_risk_level)


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

features = [
    "type",
    "severity",
    "people_affected",
    "chemical",
    "fire",
    "explosion",
    "incident_impact_score"
]

target = "risk_level"


# Keep only columns that actually exist
features = [
    column for column in features
    if column in df.columns
]

print("\nFeatures used:")
print(features)


# ---------------------------------------------------------
# REMOVE ROWS WITHOUT TARGET
# ---------------------------------------------------------

df = df.dropna(subset=[target])


X = df[features]
y = df[target]


# ---------------------------------------------------------
# IDENTIFY DATA TYPES
# ---------------------------------------------------------

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nCategorical features:")
print(categorical_features)

print("\nNumeric features:")
print(numeric_features)


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)


# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n================================")
print("INCIDENT RISK MODEL")
print("================================")

print(
    f"Accuracy: {accuracy:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

model_path = (
    MODEL_DIR /
    "incident_risk_model.pkl"
)

joblib.dump(
    pipeline,
    model_path
)


# Save feature names too

columns_path = (
    MODEL_DIR /
    "incident_risk_columns.pkl"
)

joblib.dump(
    features,
    columns_path
)


print("\nModel saved to:")

print(model_path)

print("\nDONE.")