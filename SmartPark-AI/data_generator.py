"""
data_generator.py
-----------------
Generates realistic DEMONSTRATION / SYNTHETIC historical campus parking
occupancy data across 4 distinct campus parking locations for SmartPark AI 2.0.

IMPORTANT NOTICE:
All records created by this module are DEMONSTRATION / SYNTHETIC DATA
modeled around typical college campus schedules. They do NOT represent live
CCTV, IoT sensor, or real college parking telemetry.
"""

from __future__ import annotations

import csv
import math
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List

DATA_SOURCE_DISCLAIMER = (
    "DEMONSTRATION / SYNTHETIC DATA: Generated for academic ML training and "
    "simulation across 4 campus parking zones. Does not represent live sensors, "
    "CCTV, or real campus telemetry."
)

DAYS_OF_WEEK: List[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# 4 distinct campus parking zones with separate capacities and demand profiles
CAMPUS_LOCATIONS: Dict[str, Dict[str, float]] = {
    "Main Gate Lot": {
        "capacity": 300,
        "weekday_mult": 1.02,
        "weekend_mult": 0.90,
        "evening_shift": 0.02,
    },
    "Engineering Block Lot": {
        "capacity": 250,
        "weekday_mult": 1.06,
        "weekend_mult": 0.65,
        "evening_shift": -0.04,
    },
    "Central Library Lot": {
        "capacity": 200,
        "weekday_mult": 0.94,
        "weekend_mult": 1.28,
        "evening_shift": 0.08,
    },
    "Hostel & Sports Lot": {
        "capacity": 150,
        "weekday_mult": 0.78,
        "weekend_mult": 1.35,
        "evening_shift": 0.16,
    },
}

LOCATION_NAMES: List[str] = list(CAMPUS_LOCATIONS.keys())
OPERATING_HOURS: List[int] = list(range(7, 21))


def _base_occupancy_ratio(location: str, day_name: str, hour: int) -> float:
    """Return expected occupancy ratio (0.0 to 1.0) for a specific zone, day, and hour."""
    profile = CAMPUS_LOCATIONS.get(location, CAMPUS_LOCATIONS["Main Gate Lot"])

    if day_name in ("Monday", "Tuesday", "Wednesday", "Thursday"):
        day_multiplier = 1.00 if day_name in ("Tuesday", "Wednesday") else 0.96
        curve: Dict[int, float] = {
            7: 0.24,
            8: 0.62,
            9: 0.82,
            10: 0.91,
            11: 0.93,
            12: 0.88,
            13: 0.86,
            14: 0.81,
            15: 0.69,
            16: 0.54,
            17: 0.40,
            18: 0.30,
            19: 0.22,
            20: 0.15,
        }
        base = curve.get(hour, 0.25) * day_multiplier * profile["weekday_mult"]
    elif day_name == "Friday":
        curve_fri: Dict[int, float] = {
            7: 0.20,
            8: 0.55,
            9: 0.76,
            10: 0.84,
            11: 0.85,
            12: 0.78,
            13: 0.72,
            14: 0.63,
            15: 0.48,
            16: 0.35,
            17: 0.24,
            18: 0.18,
            19: 0.14,
            20: 0.10,
        }
        base = curve_fri.get(hour, 0.20) * profile["weekday_mult"]
    elif day_name == "Saturday":
        curve_sat: Dict[int, float] = {
            7: 0.10,
            8: 0.22,
            9: 0.38,
            10: 0.50,
            11: 0.54,
            12: 0.52,
            13: 0.46,
            14: 0.41,
            15: 0.34,
            16: 0.28,
            17: 0.22,
            18: 0.18,
            19: 0.14,
            20: 0.10,
        }
        base = curve_sat.get(hour, 0.20) * profile["weekend_mult"]
    else:
        curve_sun: Dict[int, float] = {
            7: 0.08,
            8: 0.12,
            9: 0.18,
            10: 0.25,
            11: 0.28,
            12: 0.30,
            13: 0.29,
            14: 0.26,
            15: 0.24,
            16: 0.21,
            17: 0.18,
            18: 0.15,
            19: 0.12,
            20: 0.09,
        }
        base = curve_sun.get(hour, 0.15) * profile["weekend_mult"]

    if hour >= 17:
        base += profile["evening_shift"]

    return max(0.05, min(0.97, base))


def generate_synthetic_records(
    num_days: int = 90,
    availability_threshold_pct: float = 30.0,
    seed: int = 42,
) -> List[Dict[str, object]]:
    """
    Generate synthetic historical records across all 4 campus parking locations.

    With num_days=90, 4 locations, and 14 hourly slots/day, this produces
    5,040 independent location records (1,260 per location).
    """
    rng = random.Random(seed)
    end_date = datetime(2026, 10, 31)
    start_date = end_date - timedelta(days=num_days - 1)

    records: List[Dict[str, object]] = []

    for day_offset in range(num_days):
        current_date = start_date + timedelta(days=day_offset)
        day_name = DAYS_OF_WEEK[current_date.weekday()]
        date_str = current_date.strftime("%Y-%m-%d")

        for loc_idx, location_name in enumerate(LOCATION_NAMES):
            total_capacity = int(CAMPUS_LOCATIONS[location_name]["capacity"])
            daily_event_shift = rng.gauss(0.0, 0.045)
            prev_ratio = max(
                0.05,
                min(0.95, _base_occupancy_ratio(location_name, day_name, 6) + rng.gauss(0.0, 0.04)),
            )

            for hour in OPERATING_HOURS:
                minute = rng.choice([0, 15, 30, 45])
                time_str = f"{hour:02d}:{minute:02d}"

                expected_ratio = _base_occupancy_ratio(location_name, day_name, hour) + daily_event_shift

                prev_occupied = int(round(max(0.0, min(1.0, prev_ratio)) * total_capacity))
                prev_occupied = max(0, min(total_capacity, prev_occupied))
                prev_occupancy_pct = round((prev_occupied / total_capacity) * 100.0, 2)

                arrival_pressure = 0.68 * expected_ratio + 0.32 * (prev_occupied / total_capacity)
                noise = rng.gauss(0.0, 0.042)
                semester_wave = 0.03 * math.sin((day_offset + loc_idx * 3) / 14.0)

                current_ratio = max(0.03, min(0.99, arrival_pressure + noise + semester_wave))
                current_occupied = int(round(current_ratio * total_capacity))
                current_occupied = max(0, min(total_capacity, current_occupied))

                available_spaces = total_capacity - current_occupied
                available_pct = (available_spaces / total_capacity) * 100.0
                status = "HIGH" if available_pct > availability_threshold_pct else "LOW"

                records.append(
                    {
                        "Date": date_str,
                        "Time": time_str,
                        "Location": location_name,
                        "Day": day_name,
                        "Hour": hour,
                        "Total_Capacity": total_capacity,
                        "Previous_Occupied": prev_occupied,
                        "Occupancy_Percentage": prev_occupancy_pct,
                        "Available_Spaces": available_spaces,
                        "Availability_Status": status,
                    }
                )

                prev_ratio = current_ratio

    return records


def save_sample_dataset(
    output_path: str = os.path.join("data", "sample_parking_data.csv"),
    num_days: int = 90,
    availability_threshold_pct: float = 30.0,
    seed: int = 42,
) -> str:
    """Generate and write synthetic multi-location parking records to CSV."""
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    records = generate_synthetic_records(
        num_days=num_days,
        availability_threshold_pct=availability_threshold_pct,
        seed=seed,
    )

    fieldnames = [
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

    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return output_path


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(base_dir, "data", "sample_parking_data.csv")
    saved_path = save_sample_dataset(output_path=default_csv)
    print(f"Generated synthetic multi-location dataset at: {saved_path}")
    print(DATA_SOURCE_DISCLAIMER)
