from pathlib import Path

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "chemicals_clean (1).csv"
MODEL_DIR = BASE_DIR / "trained_models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "spread_model.pkl"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading chemical dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# FEATURES
# ============================================================

features = [
    "chemical",
    "release_type",
    "release_amount_kg",
    "release_duration_min",
    "wind_speed_kmh",
    "temperature_c",
    "source_height_m",
    "atmospheric_stability"
]

target = "red_zone_m"


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = features + [target]

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns in dataset: {missing}"
    )


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

df = df.dropna(
    subset=required_columns
).copy()


X = df[features]
y = df[target]


print("\nFeatures used:")
print(features)

print("\nTarget:")
print(target)

print("\nRows available:")
print(len(df))


# ============================================================
# CATEGORICAL / NUMERICAL FEATURES
# ============================================================

categorical_features = [
    "chemical",
    "release_type",
    "atmospheric_stability"
]

numeric_features = [
    "release_amount_kg",
    "release_duration_min",
    "wind_speed_kmh",
    "temperature_c",
    "source_height_m"
]


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ============================================================
# MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=12
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# USE DATASET'S OWN TRAIN / TEST SPLIT
# ============================================================

train_df = df[df["dataset_split"] == "train"].copy()
test_df = df[df["dataset_split"] == "test"].copy()

print("\nDataset split:")
print("Training:", len(train_df))
print("Testing :", len(test_df))


X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


# ============================================================
# TRAIN
# ============================================================

print("\nTraining spread prediction model...")

pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# EVALUATION
# ============================================================

predictions = pipeline.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n================================")
print("HAZARD SPREAD MODEL")
print("================================")

print(f"MAE: {mae:.2f} meters")
print(f"R2 Score: {r2:.4f}")


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nDONE.")