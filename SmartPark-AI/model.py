"""
model.py
--------
Location-aware Machine Learning training, evaluation, persistence, inference,
and leak-free multi-hour future forecasting for SmartPark AI 2.0.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from data_generator import save_sample_dataset
from data_processor import VALID_DAYS, validate_and_clean_dataset

FEATURE_COLUMNS: List[str] = [
    "Location",
    "Hour",
    "Day",
    "Previous_Occupied",
    "Total_Capacity",
    "Occupancy_Percentage",
]
NUMERIC_FEATURES: List[str] = [
    "Hour",
    "Previous_Occupied",
    "Total_Capacity",
    "Occupancy_Percentage",
]
CATEGORICAL_FEATURES: List[str] = ["Location", "Day"]
TARGET_COLUMN: str = "Availability_Status"
CLASS_LABELS: List[str] = ["HIGH", "LOW"]


class ModelTrainingError(ValueError):
    """Raised when the dataset cannot support reliable binary classification."""


def _build_preprocessor(locations: List[str]) -> ColumnTransformer:
    """Create an unfitted ColumnTransformer for numeric and categorical features."""
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    categories=[sorted(locations), VALID_DAYS],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def _evaluate_classifier(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    algorithm_name: str,
) -> Dict[str, Any]:
    """Compute real classification metrics on the held-out test set."""
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
    report_dict = classification_report(
        y_test, y_pred, labels=CLASS_LABELS, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_test, y_pred, labels=CLASS_LABELS, zero_division=0
    )

    return {
        "algorithm": algorithm_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "confusion_matrix": cm.tolist(),
        "labels": CLASS_LABELS,
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
    }


def _extract_feature_importances(
    pipeline: Pipeline,
    num_locations: int,
) -> Dict[str, float]:
    """
    Aggregate tree feature importances back to the 6 human-readable features:
    Location, Hour, Day, Previous_Occupied, Total_Capacity, Occupancy_Percentage.
    """
    classifier = pipeline.named_steps["classifier"]
    importances = classifier.feature_importances_

    num_count = len(NUMERIC_FEATURES)
    aggregated: Dict[str, float] = {}
    for idx, feat in enumerate(NUMERIC_FEATURES):
        aggregated[feat] = float(importances[idx])

    loc_start = num_count
    loc_end = num_count + num_locations
    aggregated["Location"] = float(np.sum(importances[loc_start:loc_end]))
    aggregated["Day"] = float(np.sum(importances[loc_end:]))

    total = sum(aggregated.values())
    if total > 0:
        aggregated = {k: round((v / total) * 100.0, 2) for k, v in aggregated.items()}
    return dict(sorted(aggregated.items(), key=lambda item: item[1], reverse=True))


def _build_cohort_baselines(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute historical cohort averages (by Location+Day+Hour, Location+Hour, Hour)
    strictly for contextual space estimation and explanations.
    """
    df_copy = df.copy()
    df_copy["Actual_Occupied"] = df_copy["Total_Capacity"] - df_copy["Available_Spaces"]
    df_copy["Actual_Occ_Ratio"] = df_copy["Actual_Occupied"] / df_copy["Total_Capacity"]

    by_loc_day_hour = (
        df_copy.groupby(["Location", "Day", "Hour"])["Actual_Occ_Ratio"].mean().to_dict()
    )
    by_loc_hour = df_copy.groupby(["Location", "Hour"])["Actual_Occ_Ratio"].mean().to_dict()
    by_hour = df_copy.groupby("Hour")["Actual_Occ_Ratio"].mean().to_dict()
    overall_mean = float(df_copy["Actual_Occ_Ratio"].mean())

    return {
        "by_loc_day_hour": {
            f"{loc}|{d}|{int(h)}": float(v) for (loc, d, h), v in by_loc_day_hour.items()
        },
        "by_loc_hour": {f"{loc}|{int(h)}": float(v) for (loc, h), v in by_loc_hour.items()},
        "by_hour": {int(h): float(v) for h, v in by_hour.items()},
        "overall_mean": overall_mean,
    }


def train_and_evaluate_models(
    df: pd.DataFrame,
    availability_threshold_pct: float = 30.0,
    test_size: float = 0.2,
    random_state: int = 42,
    model_save_path: Optional[str] = os.path.join("models", "parking_model.pkl"),
) -> Dict[str, Any]:
    """
    Train RandomForestClassifier and DecisionTreeClassifier on multi-location
    historical data, evaluate both on a stratified test split, and save bundle.
    """
    if df is None or len(df) < 20:
        raise ModelTrainingError(
            "Dataset has fewer than 20 valid records. At least 20 records are required "
            "to split into training and testing sets reliably."
        )

    unique_classes = sorted(df[TARGET_COLUMN].dropna().unique().tolist())
    if len(unique_classes) < 2:
        only_class = unique_classes[0] if unique_classes else "None"
        raise ModelTrainingError(
            f"The dataset contains only one target class ('{only_class}') at the "
            f"{availability_threshold_pct:.0f}% availability threshold. Both HIGH and LOW "
            "availability records are required to train the classifier."
        )

    locations = sorted(df["Location"].unique().tolist())
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    class_counts = y.value_counts()
    if int(class_counts.min()) < 2:
        raise ModelTrainingError(
            "One of the availability classes has fewer than 2 samples, which is too few "
            "for a stratified train/test split."
        )

    # Split BEFORE fitting any preprocessing steps to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(locations)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_split=4,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    rf_metrics = _evaluate_classifier(rf_pipeline, X_test, y_test, "Random Forest Classifier")
    rf_importances = _extract_feature_importances(rf_pipeline, len(locations))

    dt_pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(locations)),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=8,
                    min_samples_split=6,
                    random_state=random_state,
                ),
            ),
        ]
    )
    dt_pipeline.fit(X_train, y_train)
    dt_metrics = _evaluate_classifier(dt_pipeline, X_test, y_test, "Decision Tree Classifier")
    dt_importances = _extract_feature_importances(dt_pipeline, len(locations))

    bundle: Dict[str, Any] = {
        "primary_model": rf_pipeline,
        "comparison_model": dt_pipeline,
        "primary_metrics": rf_metrics,
        "comparison_metrics": dt_metrics,
        "feature_importances": rf_importances,
        "dt_feature_importances": dt_importances,
        "locations": locations,
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "total_records": int(len(df)),
        "threshold_pct": float(availability_threshold_pct),
        "random_state": random_state,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cohort_baselines": _build_cohort_baselines(df),
        "classes": list(rf_pipeline.named_steps["classifier"].classes_),
    }

    if model_save_path:
        parent_dir = os.path.dirname(model_save_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        joblib.dump(bundle, model_save_path)

    return bundle


def load_or_train_model(
    data_path: str = os.path.join("data", "sample_parking_data.csv"),
    model_path: str = os.path.join("models", "parking_model.pkl"),
    availability_threshold_pct: float = 30.0,
    force_retrain: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """Load an existing model bundle and dataset, or auto-generate & train if missing."""
    if not os.path.exists(data_path):
        save_sample_dataset(
            output_path=data_path,
            availability_threshold_pct=availability_threshold_pct,
        )

    raw_df = pd.read_csv(data_path)
    cleaned_df, validation_summary = validate_and_clean_dataset(
        raw_df, availability_threshold_pct=availability_threshold_pct
    )

    if not force_retrain and os.path.exists(model_path):
        try:
            bundle = joblib.load(model_path)
            if (
                isinstance(bundle, dict)
                and "primary_model" in bundle
                and "locations" in bundle
                and abs(float(bundle.get("threshold_pct", -1)) - float(availability_threshold_pct)) < 1e-6
                and int(bundle.get("total_records", -1)) == len(cleaned_df)
            ):
                return bundle, cleaned_df, validation_summary
        except Exception:
            pass

    bundle = train_and_evaluate_models(
        cleaned_df,
        availability_threshold_pct=availability_threshold_pct,
        model_save_path=model_path,
    )
    return bundle, cleaned_df, validation_summary


def predict_parking_availability(
    model_bundle: Dict[str, Any],
    day: str,
    hour: int,
    previous_occupied: int,
    total_capacity: int,
    location: str = "Main Gate Lot",
) -> Dict[str, Any]:
    """Run location-aware inference using the trained Random Forest pipeline."""
    pipeline: Pipeline = model_bundle["primary_model"]
    clean_location = str(location).strip() or "Main Gate Lot"
    clean_day = str(day).strip().title()
    occupancy_pct = round((previous_occupied / total_capacity) * 100.0, 2)

    input_df = pd.DataFrame(
        [
            {
                "Location": clean_location,
                "Hour": int(hour),
                "Day": clean_day,
                "Previous_Occupied": int(previous_occupied),
                "Total_Capacity": int(total_capacity),
                "Occupancy_Percentage": occupancy_pct,
            }
        ]
    )

    predicted_label = str(pipeline.predict(input_df)[0])
    probabilities = pipeline.predict_proba(input_df)[0]
    classes = list(pipeline.named_steps["classifier"].classes_)
    prob_map = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}

    prob_high = prob_map.get("HIGH", 0.0)
    prob_low = prob_map.get("LOW", 0.0)
    confidence_pct = round(max(prob_high, prob_low) * 100.0, 2)

    baselines = model_bundle.get("cohort_baselines", {})
    by_loc_day_hour = baselines.get("by_loc_day_hour", {})
    by_loc_hour = baselines.get("by_loc_hour", {})
    by_hour = baselines.get("by_hour", {})
    overall_mean = float(baselines.get("overall_mean", occupancy_pct / 100.0))

    cohort_ratio = by_loc_day_hour.get(
        f"{clean_location}|{clean_day}|{int(hour)}",
        by_loc_hour.get(
            f"{clean_location}|{int(hour)}",
            by_hour.get(int(hour), overall_mean),
        ),
    )
    prev_ratio = previous_occupied / total_capacity
    estimated_ratio = max(0.02, min(0.99, 0.55 * cohort_ratio + 0.45 * prev_ratio))
    estimated_occupied = int(round(estimated_ratio * total_capacity))

    threshold_pct = float(model_bundle.get("threshold_pct", 30.0))
    max_occ_for_high = max(0, int(np.floor(total_capacity * (1.0 - threshold_pct / 100.0))) - 1)
    min_occ_for_low = min(total_capacity, int(np.ceil(total_capacity * (1.0 - threshold_pct / 100.0))))

    if predicted_label == "HIGH" and estimated_occupied > max_occ_for_high:
        estimated_occupied = max_occ_for_high
    elif predicted_label == "LOW" and estimated_occupied < min_occ_for_low:
        estimated_occupied = min_occ_for_low

    estimated_available = max(0, total_capacity - estimated_occupied)
    estimated_occ_pct = round((estimated_occupied / total_capacity) * 100.0, 2)
    estimated_avail_pct = round((estimated_available / total_capacity) * 100.0, 2)

    if predicted_label == "HIGH":
        recommendation = (
            f"Good news! Parking availability at {clean_location} is expected to be high. "
            "You can proceed towards the parking area."
        )
    else:
        recommendation = (
            f"{clean_location} may be crowded. Consider arriving earlier, checking another zone "
            "in Parking Finder, or trying again later."
        )

    importances = model_bundle.get("feature_importances", {})
    top_feature = next(iter(importances), "Occupancy_Percentage")
    top_weight = importances.get(top_feature, 0.0)
    explanations: List[str] = [
        f"Top learned model signal: '{top_feature.replace('_', ' ')}' (importance: {top_weight:.1f}%).",
        f"Prior utilization at {clean_location}: {occupancy_pct:.1f}% ({previous_occupied} of {total_capacity} spaces occupied).",
        f"Historical {clean_location} occupancy on {clean_day} around {int(hour):02d}:00 averages {cohort_ratio * 100:.1f}%.",
    ]

    return {
        "location": clean_location,
        "predicted_status": predicted_label,
        "confidence_pct": confidence_pct,
        "prob_high_pct": round(prob_high * 100.0, 2),
        "prob_low_pct": round(prob_low * 100.0, 2),
        "estimated_occupied": estimated_occupied,
        "estimated_available": estimated_available,
        "estimated_occupancy_pct": estimated_occ_pct,
        "estimated_availability_pct": estimated_avail_pct,
        "previous_occupancy_pct": occupancy_pct,
        "recommendation": recommendation,
        "explanations": explanations,
        "threshold_pct": threshold_pct,
    }


def forecast_next_hours(
    model_bundle: Dict[str, Any],
    location: str,
    day: str,
    start_hour: int,
    initial_occupied: int,
    total_capacity: int,
    horizon_hours: int = 4,
) -> List[Dict[str, Any]]:
    """
    Generate a leak-free multi-hour future availability forecast.

    At step k=0, uses the caller's known `initial_occupied` count.
    At step k>0, chains ONLY the previous step's model-estimated `estimated_occupied`
    as `previous_occupied` — never reading future actual occupancy from the dataset.
    """
    forecasts: List[Dict[str, Any]] = []
    running_occupied = int(initial_occupied)
    day_idx = VALID_DAYS.index(day) if day in VALID_DAYS else 0

    for offset in range(horizon_hours):
        raw_hour = int(start_hour) + offset
        target_hour = raw_hour % 24
        target_day = VALID_DAYS[(day_idx + (raw_hour // 24)) % 7]

        step_pred = predict_parking_availability(
            model_bundle=model_bundle,
            day=target_day,
            hour=target_hour,
            previous_occupied=running_occupied,
            total_capacity=total_capacity,
            location=location,
        )
        forecasts.append(
            {
                "step_offset": offset,
                "day": target_day,
                "hour": target_hour,
                "time_label": f"{target_hour:02d}:00",
                "input_prev_occupied": running_occupied,
                **step_pred,
            }
        )
        # Autoregressive chaining: next hour receives only this step's estimated occupancy
        running_occupied = int(step_pred["estimated_occupied"])

    return forecasts
