"""
data_processor.py
-----------------
Handles CSV loading, schema validation, missing-value cleaning, location
normalization, threshold-based labeling, prediction input checks, and disk
persistence for configuration, active datasets, and prediction logs.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

VALID_DAYS: List[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

DEFAULT_LOCATIONS: List[str] = [
    "Main Gate Lot",
    "Engineering Block Lot",
    "Central Library Lot",
    "Hostel & Sports Lot",
]

REQUIRED_COLUMNS: List[str] = [
    "Day",
    "Total_Capacity",
    "Previous_Occupied",
]

STANDARD_COLUMNS: List[str] = [
    "Date",
    "Time",
    "Location",
    "Day",
    "Hour",
    "Total_Capacity",
    "Previous_Occupied",
    "Occupancy_Percentage",
    "Available_Spaces",
    "Availability_Status",
]

PREDICTION_HISTORY_COLUMNS: List[str] = [
    "Timestamp",
    "Location",
    "Day",
    "Entry_Time",
    "Hour",
    "Total_Capacity",
    "Previous_Occupied",
    "Previous_Occupancy_Pct",
    "Predicted_Status",
    "Confidence_Pct",
    "Estimated_Occupied",
    "Estimated_Available",
    "Threshold_Pct",
]


class DatasetValidationError(ValueError):
    """Raised when an uploaded or loaded CSV dataset cannot be processed safely."""


def get_csv_template() -> str:
    """Return a clean sample CSV template string with Location column for user uploads."""
    header = ",".join(STANDARD_COLUMNS)
    rows = [
        "2026-10-12,09:00,Main Gate Lot,Monday,9,300,240,80.0,45,LOW",
        "2026-10-12,17:30,Engineering Block Lot,Monday,17,250,95,38.0,158,HIGH",
        "2026-10-13,11:00,Central Library Lot,Tuesday,11,200,170,85.0,22,LOW",
        "2026-10-17,10:00,Hostel & Sports Lot,Saturday,10,150,60,40.0,88,HIGH",
    ]
    return "\n".join([header] + rows) + "\n"


def _parse_hour_from_time(time_val: Any) -> Optional[int]:
    """Extract an integer hour (0..23) from a time string like '09:30' or '14:00:00'."""
    if pd.isna(time_val):
        return None
    text = str(time_val).strip()
    if not text:
        return None
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p"):
        try:
            return datetime.strptime(text, fmt).hour
        except ValueError:
            continue
    try:
        numeric_hour = int(float(text))
        if 0 <= numeric_hour <= 23:
            return numeric_hour
    except ValueError:
        pass
    return None


def validate_and_clean_dataset(
    raw_df: pd.DataFrame,
    availability_threshold_pct: float = 30.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate, clean, and enrich a campus parking dataset (multi-location or legacy single-lot).
    """
    if raw_df is None or raw_df.empty:
        raise DatasetValidationError(
            "The dataset is empty. Please upload a CSV file with historical parking records."
        )

    df = raw_df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_required:
        raise DatasetValidationError(
            f"Missing required column(s): {', '.join(missing_required)}. "
            f"Expected at least: {', '.join(REQUIRED_COLUMNS)} plus Hour or Time."
        )

    if "Hour" not in df.columns and "Time" not in df.columns:
        raise DatasetValidationError(
            "Dataset must contain either an 'Hour' column (0-23) or a 'Time' column (HH:MM)."
        )

    initial_rows = len(df)
    issues: List[str] = []

    if "Date" not in df.columns:
        df["Date"] = datetime.now().strftime("%Y-%m-%d")
    else:
        df["Date"] = df["Date"].fillna(datetime.now().strftime("%Y-%m-%d")).astype(str).str.strip()

    if "Location" not in df.columns:
        df["Location"] = "Main Gate Lot"
        issues.append("Column 'Location' was not provided; assigned default 'Main Gate Lot'.")
    else:
        df["Location"] = (
            df["Location"]
            .fillna("Main Gate Lot")
            .astype(str)
            .str.strip()
            .replace({"": "Main Gate Lot", "nan": "Main Gate Lot"})
        )

    if "Hour" in df.columns:
        df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
        if "Time" in df.columns:
            missing_hour_mask = df["Hour"].isna()
            if missing_hour_mask.any():
                df.loc[missing_hour_mask, "Hour"] = df.loc[missing_hour_mask, "Time"].apply(
                    _parse_hour_from_time
                )
    else:
        df["Hour"] = df["Time"].apply(_parse_hour_from_time)

    if "Time" not in df.columns:
        df["Time"] = df["Hour"].apply(
            lambda h: f"{int(h):02d}:00" if pd.notna(h) and 0 <= h <= 23 else None
        )
    else:
        df["Time"] = df["Time"].astype(str).str.strip()

    df["Day"] = df["Day"].astype(str).str.strip().str.title()
    invalid_day_mask = ~df["Day"].isin(VALID_DAYS)
    if invalid_day_mask.any():
        issues.append(f"Removed {int(invalid_day_mask.sum())} row(s) with invalid Day values.")

    df["Total_Capacity"] = pd.to_numeric(df["Total_Capacity"], errors="coerce")
    df["Previous_Occupied"] = pd.to_numeric(df["Previous_Occupied"], errors="coerce")

    if "Available_Spaces" in df.columns:
        df["Available_Spaces"] = pd.to_numeric(df["Available_Spaces"], errors="coerce")

    missing_core_mask = (
        df["Hour"].isna()
        | df["Total_Capacity"].isna()
        | df["Previous_Occupied"].isna()
        | invalid_day_mask
    )
    invalid_hour_mask = (df["Hour"] < 0) | (df["Hour"] > 23)
    invalid_capacity_mask = df["Total_Capacity"] <= 0
    invalid_occ_mask = (df["Previous_Occupied"] < 0) | (
        df["Previous_Occupied"] > df["Total_Capacity"]
    )

    drop_mask = (
        missing_core_mask | invalid_hour_mask | invalid_capacity_mask | invalid_occ_mask
    )
    dropped_rows = int(drop_mask.sum())
    if dropped_rows > 0:
        if invalid_capacity_mask.any():
            issues.append(
                f"Removed {int(invalid_capacity_mask.sum())} row(s) with non-positive Total_Capacity."
            )
        if invalid_occ_mask.any():
            issues.append(
                f"Removed {int(invalid_occ_mask.sum())} row(s) where Previous_Occupied was < 0 or exceeded Total_Capacity."
            )
        if missing_core_mask.any():
            issues.append(
                f"Removed {int(missing_core_mask.sum())} row(s) with missing critical values."
            )

    cleaned = df.loc[~drop_mask].copy()
    if cleaned.empty:
        raise DatasetValidationError(
            "No valid parking records remained after cleaning invalid or missing rows."
        )

    cleaned["Hour"] = cleaned["Hour"].astype(int)
    cleaned["Total_Capacity"] = cleaned["Total_Capacity"].astype(int)
    cleaned["Previous_Occupied"] = cleaned["Previous_Occupied"].astype(int)

    cleaned["Occupancy_Percentage"] = np.round(
        (cleaned["Previous_Occupied"] / cleaned["Total_Capacity"]) * 100.0, 2
    )

    if "Available_Spaces" not in cleaned.columns:
        cleaned["Available_Spaces"] = cleaned["Total_Capacity"] - cleaned["Previous_Occupied"]
        issues.append(
            "Column 'Available_Spaces' was not provided; derived from Total_Capacity - Previous_Occupied."
        )
    else:
        invalid_avail = (
            cleaned["Available_Spaces"].isna()
            | (cleaned["Available_Spaces"] < 0)
            | (cleaned["Available_Spaces"] > cleaned["Total_Capacity"])
        )
        if invalid_avail.any():
            cleaned.loc[invalid_avail, "Available_Spaces"] = (
                cleaned.loc[invalid_avail, "Total_Capacity"]
                - cleaned.loc[invalid_avail, "Previous_Occupied"]
            )
            issues.append(
                f"Imputed {int(invalid_avail.sum())} missing/out-of-range Available_Spaces values."
            )
        cleaned["Available_Spaces"] = cleaned["Available_Spaces"].astype(int)

    available_pct = (cleaned["Available_Spaces"] / cleaned["Total_Capacity"]) * 100.0
    cleaned["Availability_Status"] = np.where(
        available_pct > float(availability_threshold_pct), "HIGH", "LOW"
    )

    ordered_cols = [c for c in STANDARD_COLUMNS if c in cleaned.columns]
    extra_cols = [c for c in cleaned.columns if c not in ordered_cols]
    cleaned = cleaned[ordered_cols + extra_cols].reset_index(drop=True)

    summary: Dict[str, Any] = {
        "initial_rows": initial_rows,
        "valid_rows": len(cleaned),
        "dropped_rows": dropped_rows,
        "threshold_pct": float(availability_threshold_pct),
        "high_count": int((cleaned["Availability_Status"] == "HIGH").sum()),
        "low_count": int((cleaned["Availability_Status"] == "LOW").sum()),
        "locations": sorted(cleaned["Location"].unique().tolist()),
        "issues": issues,
    }
    return cleaned, summary


def validate_prediction_input(
    day: str,
    entry_time: str,
    total_capacity: Any,
    previous_occupied: Any,
    location: str = "Main Gate Lot",
    valid_locations: Optional[List[str]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate user-supplied parking prediction inputs including location."""
    clean_location = str(location).strip()
    if not clean_location:
        return False, "Please select a valid parking location.", {}
    if valid_locations and clean_location not in valid_locations:
        return False, f"Unknown parking location '{clean_location}'.", {}

    clean_day = str(day).strip().title()
    if clean_day not in VALID_DAYS:
        return False, f"Day must be one of {', '.join(VALID_DAYS)}.", {}

    hour = _parse_hour_from_time(entry_time)
    if hour is None:
        return False, "Entry time must be a valid time in HH:MM format (e.g., 09:30).", {}

    try:
        capacity_num = int(total_capacity)
    except (TypeError, ValueError):
        return False, "Total parking capacity must be a positive whole number.", {}

    if capacity_num <= 0:
        return False, "Total parking capacity must be greater than 0.", {}

    try:
        occupied_num = int(previous_occupied)
    except (TypeError, ValueError):
        return False, "Previous occupied spaces must be a non-negative whole number.", {}

    if occupied_num < 0:
        return False, "Previous occupied spaces cannot be negative.", {}

    if occupied_num > capacity_num:
        return (
            False,
            f"Previous occupied spaces ({occupied_num}) cannot exceed total capacity ({capacity_num}).",
            {},
        )

    occupancy_pct = round((occupied_num / capacity_num) * 100.0, 2)
    return (
        True,
        "",
        {
            "Location": clean_location,
            "Day": clean_day,
            "Entry_Time": str(entry_time).strip(),
            "Hour": int(hour),
            "Total_Capacity": capacity_num,
            "Previous_Occupied": occupied_num,
            "Occupancy_Percentage": occupancy_pct,
        },
    )


def load_prediction_history(
    history_path: str = os.path.join("data", "predictions.csv"),
) -> pd.DataFrame:
    """Load saved prediction records from CSV, upgrading legacy headers if needed."""
    if not os.path.exists(history_path):
        return pd.DataFrame(columns=PREDICTION_HISTORY_COLUMNS)
    try:
        df = pd.read_csv(history_path)
        if df.empty:
            return pd.DataFrame(columns=PREDICTION_HISTORY_COLUMNS)
        if "Location" not in df.columns:
            df["Location"] = "Main Gate Lot"
        return df[[c for c in PREDICTION_HISTORY_COLUMNS if c in df.columns]]
    except Exception:
        return pd.DataFrame(columns=PREDICTION_HISTORY_COLUMNS)


def append_prediction_history(
    record: Dict[str, Any],
    history_path: str = os.path.join("data", "predictions.csv"),
) -> pd.DataFrame:
    """Append a new prediction record to local CSV storage and return updated history."""
    parent_dir = os.path.dirname(history_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    existing = load_prediction_history(history_path)
    new_row = pd.DataFrame([{col: record.get(col, "") for col in PREDICTION_HISTORY_COLUMNS}])
    updated = pd.concat([new_row, existing], ignore_index=True)
    updated.to_csv(history_path, index=False)
    return updated


def load_app_config(config_path: str = os.path.join("data", "app_config.json")) -> Dict[str, Any]:
    """Load persisted app configuration (threshold, active data source mode, selected location)."""
    default_cfg: Dict[str, Any] = {
        "threshold_pct": 30.0,
        "data_source_mode": "SYNTHETIC",
        "selected_location": "All Campus Zones",
    }
    if not os.path.exists(config_path):
        return default_cfg
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            default_cfg.update(loaded)
        return default_cfg
    except Exception:
        return default_cfg


def save_app_config(
    config: Dict[str, Any],
    config_path: str = os.path.join("data", "app_config.json"),
) -> None:
    """Persist app configuration to disk."""
    parent_dir = os.path.dirname(config_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
