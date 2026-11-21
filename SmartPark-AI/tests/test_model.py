"""
tests/test_model.py
-------------------
Comprehensive QA verification suite for SmartPark AI 2.0 covering:
1. Multi-location generation & separate capacities per location
2. Location filtering & isolation
3. Model training & evaluation with Location feature
4. Real ML prediction & leak-free chained future forecasting
5. Invalid input handling & CSV validation
6. Disk persistence across restarts (config & prediction history)
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analytics import compute_dataset_summary, filter_dataset_by_location, get_location_snapshots
from data_generator import CAMPUS_LOCATIONS, generate_synthetic_records, save_sample_dataset
from data_processor import (
    DatasetValidationError,
    append_prediction_history,
    load_app_config,
    load_prediction_history,
    save_app_config,
    validate_and_clean_dataset,
    validate_prediction_input,
)
from model import (
    ModelTrainingError,
    forecast_next_hours,
    load_or_train_model,
    predict_parking_availability,
    train_and_evaluate_models,
)


def test_four_locations_have_separate_capacities(tmp_path):
    records = generate_synthetic_records(num_days=40, seed=42)
    df = pd.DataFrame(records)
    assert set(df["Location"].unique()) == set(CAMPUS_LOCATIONS.keys())

    snapshots = get_location_snapshots(df)
    assert len(snapshots) == 4
    for snap in snapshots:
        expected_cap = int(CAMPUS_LOCATIONS[snap["location"]]["capacity"])
        assert snap["total_capacity"] == expected_cap
        assert snap["latest_occupied"] + snap["latest_available"] == snap["total_capacity"]


def test_location_filtering_isolates_records():
    df = pd.DataFrame(generate_synthetic_records(num_days=30, seed=42))
    eng_df = filter_dataset_by_location(df, "Engineering Block Lot")
    assert set(eng_df["Location"].unique()) == {"Engineering Block Lot"}
    summary = compute_dataset_summary(eng_df)
    assert summary["total_capacity"] == 250

    all_summary = compute_dataset_summary(df)
    assert all_summary["total_capacity"] == sum(int(v["capacity"]) for v in CAMPUS_LOCATIONS.values())


def test_model_training_prediction_and_leak_free_forecast(tmp_path):
    csv_path = save_sample_dataset(str(tmp_path / "sample.csv"), num_days=40, seed=42)
    raw_df = pd.read_csv(csv_path)
    cleaned_df, _ = validate_and_clean_dataset(raw_df, availability_threshold_pct=30.0)
    model_path = str(tmp_path / "parking_model.pkl")

    bundle = train_and_evaluate_models(cleaned_df, availability_threshold_pct=30.0, model_save_path=model_path)
    assert os.path.exists(model_path)
    assert "Location" in bundle["feature_importances"]

    pred = predict_parking_availability(
        bundle,
        location="Central Library Lot",
        day="Monday",
        hour=10,
        previous_occupied=175,
        total_capacity=200,
    )
    assert pred["predicted_status"] in ("HIGH", "LOW")
    assert abs((pred["prob_high_pct"] + pred["prob_low_pct"]) - 100.0) < 0.2

    forecasts = forecast_next_hours(
        bundle,
        location="Central Library Lot",
        day="Monday",
        start_hour=9,
        initial_occupied=150,
        total_capacity=200,
        horizon_hours=4,
    )
    assert len(forecasts) == 4
    # Verify chained input uses only prior step's estimated_occupied
    assert forecasts[0]["input_prev_occupied"] == 150
    for idx in range(1, 4):
        assert forecasts[idx]["input_prev_occupied"] == forecasts[idx - 1]["estimated_occupied"]


def test_legacy_csv_without_location_and_invalid_inputs():
    with pytest.raises(DatasetValidationError):
        validate_and_clean_dataset(pd.DataFrame())

    legacy_df = pd.DataFrame(
        [
            {"Date": "2026-10-01", "Time": "09:00", "Day": "Monday", "Hour": 9, "Total_Capacity": 200, "Previous_Occupied": 160, "Available_Spaces": 35},
            {"Date": "2026-10-01", "Time": "10:00", "Day": "Monday", "Hour": 10, "Total_Capacity": 0, "Previous_Occupied": 10, "Available_Spaces": 0},
            {"Date": "2026-10-01", "Time": "11:00", "Day": "Monday", "Hour": 11, "Total_Capacity": 100, "Previous_Occupied": 150, "Available_Spaces": 0},
        ]
    )
    cleaned, summary = validate_and_clean_dataset(legacy_df, availability_threshold_pct=30.0)
    assert summary["valid_rows"] == 1
    assert cleaned.iloc[0]["Location"] == "Main Gate Lot"

    ok, _, _ = validate_prediction_input("Tuesday", "09:30", 100, 120, location="Main Gate Lot")
    assert not ok


def test_persistence_across_restart(tmp_path):
    cfg_path = str(tmp_path / "app_config.json")
    save_app_config(
        {"threshold_pct": 35.0, "data_source_mode": "CUSTOM", "selected_location": "Main Gate Lot"},
        cfg_path,
    )
    loaded = load_app_config(cfg_path)
    assert loaded["threshold_pct"] == 35.0
    assert loaded["data_source_mode"] == "CUSTOM"

    history_path = str(tmp_path / "predictions.csv")
    append_prediction_history(
        {
            "Timestamp": "2026-10-15 10:00:00",
            "Location": "Main Gate Lot",
            "Day": "Thursday",
            "Entry_Time": "10:00",
            "Hour": 10,
            "Total_Capacity": 300,
            "Previous_Occupied": 240,
            "Previous_Occupancy_Pct": 80.0,
            "Predicted_Status": "LOW",
            "Confidence_Pct": 91.5,
            "Estimated_Occupied": 245,
            "Estimated_Available": 55,
            "Threshold_Pct": 35.0,
        },
        history_path=history_path,
    )
    hist = load_prediction_history(history_path=history_path)
    assert len(hist) == 1
    assert hist.iloc[0]["Location"] == "Main Gate Lot"
