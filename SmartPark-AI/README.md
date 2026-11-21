# SmartPark AI 2.0 – Intelligent Campus Parking Availability Predictor

**Predict smarter. Park faster. Save time.**

SmartPark AI 2.0 is a multi-zone Machine Learning web application built with **Python**, **Streamlit**, **Scikit-learn**, **Pandas**, and **Plotly** to help students, faculty members, and campus visitors check zone availability and predict **HIGH** vs **LOW** parking availability across 4 campus parking locations.

> **Transparency Notice:** The default dataset bundled with this repository (`data/sample_parking_data.csv`) is **DEMONSTRATION / SYNTHETIC DATA** generated across 4 campus zones for academic ML training. SmartPark AI does **not** connect to live CCTV cameras, physical IoT sensors, or real-time GPS feeds.

---

## Campus Parking Locations (4 Independent Zones)

| Location | Capacity | Typical Demand Pattern |
| :--- | :---: | :--- |
| **Main Gate Lot** | 300 spaces | High commuter and visitor turnover throughout weekdays |
| **Engineering Block Lot** | 250 spaces | Steepest morning and midday lecture congestion |
| **Central Library Lot** | 200 spaces | Steady weekday and elevated Saturday study occupancy |
| **Hostel & Sports Lot** | 150 spaces | Lighter morning traffic with higher evening and weekend usage |

---

## Key Features

- **Dedicated Parking Finder:** Commuter-facing lot comparison cards across all 4 zones plus a **4-Hour Chained ML Forecast** that chains only prior model estimates (never future actuals).
- **Location-Scoped Analytics Dashboard:** Filter by `All Campus Zones` (accurately summing per-zone capacities = `900`) or inspect any single zone independently across KPIs and 5 interactive charts.
- **Predict Availability Workbench:** Location-aware single-slot inference using `RandomForestClassifier` (`predict_proba` confidence), visual gauge, factor explanation, and disk-persisted prediction history.
- **Parking Analytics & Historical Data:** Location-filterable utilization rankings, search, date/day/status filters, sortable tables, custom CSV upload with validation preview, and template/report exports.
- **Disk Persistence Across Restarts:** Uploaded custom CSVs (`data/active_dataset.csv`), active threshold/location settings (`data/app_config.json`), prediction logs (`data/predictions.csv`), and trained models (`models/parking_model.pkl`) persist across server restarts.

---

## Installation & Run Commands

```bash
cd SmartPark-AI
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run automated tests:

```bash
pytest -v
```

---

## Project Folder Structure

```text
SmartPark-AI/
├── app.py
├── model.py
├── data_generator.py
├── data_processor.py
├── analytics.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── sample_parking_data.csv
│   └── predictions.csv
├── models/
│   └── .gitkeep
├── assets/
│   └── logo.png
└── tests/
    └── test_model.py
```
