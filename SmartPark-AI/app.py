"""
app.py
------
Main Streamlit web application for SmartPark AI 2.0 – Intelligent Campus Parking System.
Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import datetime, time as dt_time
from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    compute_dataset_summary,
    filter_dataset_by_location,
    get_location_snapshots,
    plot_availability_by_day,
    plot_availability_split,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_occupancy_by_hour,
    plot_occupancy_distribution,
    plot_weekday_occupancy_comparison,
    plot_weekly_trends,
)
from data_generator import DATA_SOURCE_DISCLAIMER, save_sample_dataset
from data_processor import (
    DatasetValidationError,
    VALID_DAYS,
    append_prediction_history,
    get_csv_template,
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "sample_parking_data.csv")
ACTIVE_DATA_PATH = os.path.join(BASE_DIR, "data", "active_dataset.csv")
CONFIG_PATH = os.path.join(BASE_DIR, "data", "app_config.json")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "predictions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "parking_model.pkl")
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")

st.set_page_config(
    page_title="SmartPark AI 2.0 – Intelligent Campus Parking System",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }
    section[data-testid="stSidebar"] {
        background-color: #0B192C;
        color: #FFFFFF;
    }
    section[data-testid="stSidebar"] * {
        color: #F8FAFC;
    }
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stRadio label {
        color: #E2E8F0 !important;
    }
    .smartpark-header {
        background: linear-gradient(135deg, #0B192C 0%, #152C4E 100%);
        padding: 1.5rem 2rem;
        border-radius: 14px;
        color: #FFFFFF;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 16px rgba(11, 25, 44, 0.14);
    }
    .smartpark-header h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .smartpark-header p {
        margin: 0.35rem 0 0 0;
        font-size: 1rem;
        color: #BAE6FD;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0B192C;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #475569;
        margin-top: 0.25rem;
    }
    .status-banner-high {
        background-color: #ECFDF5;
        border: 1px solid #86EFAC;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        color: #14532D;
    }
    .status-banner-low {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        color: #7F1D1D;
    }
    .disclaimer-box {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: #1E3A8A;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _persist_config(threshold_pct: float, mode: str, selected_location: str) -> None:
    save_app_config(
        {
            "threshold_pct": float(threshold_pct),
            "data_source_mode": mode,
            "selected_location": selected_location,
        },
        CONFIG_PATH,
    )


def _init_session_state(threshold_pct: float) -> None:
    """Initialize dataset and trained ML model from persisted disk state."""
    needs_init = (
        "active_df" not in st.session_state
        or "model_bundle" not in st.session_state
        or st.session_state.get("active_threshold") != threshold_pct
    )
    if not needs_init:
        return

    cfg = load_app_config(CONFIG_PATH)
    use_custom = cfg.get("data_source_mode") == "CUSTOM" and os.path.exists(ACTIVE_DATA_PATH)
    target_data_path = ACTIVE_DATA_PATH if use_custom else DATA_PATH

    try:
        bundle, cleaned_df, summary = load_or_train_model(
            data_path=target_data_path,
            model_path=MODEL_PATH,
            availability_threshold_pct=threshold_pct,
            force_retrain=(st.session_state.get("active_threshold") != threshold_pct),
        )
    except (DatasetValidationError, ModelTrainingError) as exc:
        st.error(f"Dataset/Model error: {exc}. Falling back to synthetic sample dataset.")
        bundle, cleaned_df, summary = load_or_train_model(
            data_path=DATA_PATH,
            model_path=MODEL_PATH,
            availability_threshold_pct=threshold_pct,
            force_retrain=True,
        )
        use_custom = False

    st.session_state["active_df"] = cleaned_df
    st.session_state["validation_summary"] = summary
    st.session_state["model_bundle"] = bundle
    st.session_state["data_source_mode"] = "CUSTOM" if use_custom else "SYNTHETIC"
    st.session_state["active_threshold"] = threshold_pct


def _render_metric_card(label: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="smartpark-header">
            <h1>SmartPark AI – Intelligent Campus Parking System</h1>
            <p>Predict smarter. Park faster. Save time.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_availability_gauge(available_pct: float, status: str) -> go.Figure:
    bar_color = "#16A34A" if status == "HIGH" else "#DC2626"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=available_pct,
            number={"suffix": "% Open", "font": {"color": "#0B192C", "size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
                "bar": {"color": bar_color},
                "bgcolor": "#F1F5F9",
                "steps": [
                    {"range": [0, 30], "color": "#FEE2E2"},
                    {"range": [30, 100], "color": "#DCFCE7"},
                ],
            },
            title={"text": "Estimated Availability", "font": {"size": 14, "color": "#475569"}},
        )
    )
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render_parking_finder(df: pd.DataFrame, model_bundle: dict, threshold_pct: float) -> None:
    """Dedicated Commuter Parking Finder experience (separate from Analytics Dashboard)."""
    st.subheader("Campus Parking Finder")
    st.write(
        "Compare capacity across all campus parking zones, select a zone, and generate a "
        "4-hour leak-free ML forecast using only your current observed occupancy and the trained model."
    )

    snapshots = get_location_snapshots(df)
    if not snapshots:
        st.warning("No parking location records found.")
        return

    cols = st.columns(len(snapshots))
    for idx, snap in enumerate(snapshots):
        with cols[idx]:
            status_color = "#16A34A" if snap["latest_status"] == "HIGH" else "#DC2626"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{snap['location']}</div>
                    <div class="metric-value" style="color:{status_color}">{snap['latest_status']}</div>
                    <div class="metric-sub">
                        Available: <strong>{snap['latest_available']}</strong> / {snap['total_capacity']} spaces ({snap['latest_availability_pct']:.1f}% open)<br/>
                        Occupied: {snap['latest_occupied']} ({snap['latest_occupancy_pct']:.1f}%)<br/>
                        Latest log: {snap['latest_day']} {snap['latest_time']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    loc_names = [s["location"] for s in snapshots]
    selected_loc = st.selectbox("Select Parking Location to Inspect & Forecast", loc_names)
    selected_snap = next(s for s in snapshots if s["location"] == selected_loc)

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        forecast_day = st.selectbox("Arrival Day", VALID_DAYS, index=0, key="finder_day")
    with f_col2:
        forecast_hour = st.slider("Starting Hour (24h)", min_value=6, max_value=22, value=9, key="finder_hour")
    with f_col3:
        current_occ_input = st.number_input(
            f"Current Observed Occupied (Max {selected_snap['total_capacity']})",
            min_value=0,
            max_value=selected_snap["total_capacity"],
            value=min(selected_snap["latest_occupied"], selected_snap["total_capacity"]),
            step=5,
            key="finder_occ",
        )

    forecast_rows = forecast_next_hours(
        model_bundle=model_bundle,
        location=selected_loc,
        day=forecast_day,
        start_hour=forecast_hour,
        initial_occupied=int(current_occ_input),
        total_capacity=selected_snap["total_capacity"],
        horizon_hours=4,
    )

    st.markdown(f"#### 4-Hour Chained ML Forecast for {selected_loc}")
    st.caption(
        "Each subsequent hour chains only the previous hour's ML-estimated occupancy—never future actual records."
    )
    forecast_df = pd.DataFrame(
        [
            {
                "Day": r["day"],
                "Hour": r["time_label"],
                "Predicted Status": r["predicted_status"],
                "Confidence (%)": r["confidence_pct"],
                "Est. Occupied": r["estimated_occupied"],
                "Est. Available": r["estimated_available"],
                "Est. Open (%)": r["estimated_availability_pct"],
            }
            for r in forecast_rows
        ]
    )
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)


def render_dashboard(df: pd.DataFrame, model_bundle: dict, threshold_pct: float) -> None:
    """Section 1: Location-Scoped Analytics Dashboard."""
    locations = ["All Campus Zones"] + sorted(df["Location"].unique().tolist())
    cfg = load_app_config(CONFIG_PATH)
    default_loc = cfg.get("selected_location", "All Campus Zones")
    default_idx = locations.index(default_loc) if default_loc in locations else 0

    selected_loc = st.selectbox("Filter Dashboard by Parking Location", locations, index=default_idx, key="dash_loc")
    _persist_config(threshold_pct, st.session_state.get("data_source_mode", "SYNTHETIC"), selected_loc)

    scoped_df = filter_dataset_by_location(df, selected_loc)
    summary = compute_dataset_summary(scoped_df)
    rf_accuracy = model_bundle["primary_metrics"]["accuracy"] * 100.0

    snapshots = get_location_snapshots(scoped_df)
    rep_snap = snapshots[0] if snapshots else None
    scenario_pred = (
        predict_parking_availability(
            model_bundle=model_bundle,
            location=rep_snap["location"],
            day=rep_snap["latest_day"],
            hour=rep_snap["latest_hour"],
            previous_occupied=rep_snap["latest_occupied"],
            total_capacity=rep_snap["total_capacity"],
        )
        if rep_snap
        else None
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_metric_card(
            "Total Parking Capacity",
            f"{summary['total_capacity']:,} spaces",
            f"Scope: {selected_loc}",
        )
    with c2:
        _render_metric_card(
            "Currently Occupied (Latest)",
            f"{summary['latest_occupied']:,} spaces",
            f"Avg Occupied: {summary['avg_occupied']} / zone",
        )
    with c3:
        _render_metric_card(
            "Available Parking Spaces",
            f"{summary['latest_available']:,} spaces",
            f"Dataset Avg: {summary['avg_available']} open / zone",
        )
    with c4:
        _render_metric_card(
            "Current Availability %",
            f"{summary['latest_availability_pct']:.1f}%",
            f"Historical Avg: {summary['avg_availability_pct']:.1f}% open",
        )

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

    c5, c6, c7 = st.columns(3)
    with c5:
        pred_label = scenario_pred["predicted_status"] if scenario_pred else "N/A"
        conf_val = f"Confidence: {scenario_pred['confidence_pct']:.1f}%" if scenario_pred else ""
        _render_metric_card("Predicted Availability Status", f"{pred_label} AVAILABILITY", conf_val)
    with c6:
        _render_metric_card(
            "Total Historical Records",
            f"{summary['total_records']:,}",
            f"Threshold: >{threshold_pct:.0f}% open = HIGH",
        )
    with c7:
        _render_metric_card(
            "Model Accuracy (Random Forest)",
            f"{rf_accuracy:.2f}%",
            f"Test Set: {model_bundle['test_records']} records",
        )

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    peak_str = ", ".join([f"{item['hour']:02d}:00 ({item['avg_occ_pct']}%)" for item in summary["busiest_hours"]])
    st.info(f"**Peak Hours ({selected_loc}):** {peak_str}  |  **Recommendation:** {summary['best_time_recommendation']}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(plot_occupancy_distribution(scoped_df), use_container_width=True)
    with col_right:
        st.plotly_chart(plot_availability_by_day(scoped_df), use_container_width=True)

    col_left2, col_right2 = st.columns(2)
    with col_left2:
        st.plotly_chart(plot_occupancy_by_hour(scoped_df), use_container_width=True)
    with col_right2:
        st.plotly_chart(plot_weekly_trends(scoped_df), use_container_width=True)

    st.plotly_chart(plot_availability_split(scoped_df), use_container_width=True)


def render_predict_page(df: pd.DataFrame, model_bundle: dict, threshold_pct: float) -> None:
    """Section 2: Location-Aware Prediction Workbench."""
    st.subheader("Predict Campus Parking Availability")
    snapshots = get_location_snapshots(df)
    loc_cap_map = {s["location"]: s["total_capacity"] for s in snapshots}
    loc_names = list(loc_cap_map.keys()) or ["Main Gate Lot"]

    selected_location = st.selectbox("Parking Location", loc_names, index=0, key="pred_loc")
    default_cap = loc_cap_map.get(selected_location, 250)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            selected_day = st.selectbox("Day of the Week", VALID_DAYS, index=0)
            entry_time = st.time_input("Time of Entry", value=dt_time(10, 0))
        with col2:
            total_capacity = st.number_input(
                "Total Parking Capacity (spaces)",
                min_value=1,
                max_value=10000,
                value=int(default_cap),
                step=10,
            )
            previous_occupied = st.number_input(
                "Previous Occupied Parking Spaces",
                min_value=0,
                max_value=10000,
                value=min(175, int(default_cap)),
                step=5,
            )

        auto_occ_pct = (
            round((previous_occupied / total_capacity) * 100.0, 2)
            if total_capacity > 0 and previous_occupied <= total_capacity
            else None
        )
        if auto_occ_pct is not None:
            st.caption(f"Calculated Previous Occupancy Percentage: **{auto_occ_pct:.2f}%**")
        else:
            st.caption("Ensure Previous Occupied Spaces does not exceed Total Parking Capacity.")

        submitted = st.form_submit_button("Predict Parking Availability", type="primary")

    if submitted:
        entry_time_str = entry_time.strftime("%H:%M")
        is_valid, err_msg, parsed = validate_prediction_input(
            day=selected_day,
            entry_time=entry_time_str,
            total_capacity=total_capacity,
            previous_occupied=previous_occupied,
            location=selected_location,
            valid_locations=loc_names,
        )
        if not is_valid:
            st.error(err_msg)
        else:
            result = predict_parking_availability(
                model_bundle=model_bundle,
                location=parsed["Location"],
                day=parsed["Day"],
                hour=parsed["Hour"],
                previous_occupied=parsed["Previous_Occupied"],
                total_capacity=parsed["Total_Capacity"],
            )

            history_row = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Location": parsed["Location"],
                "Day": parsed["Day"],
                "Entry_Time": parsed["Entry_Time"],
                "Hour": parsed["Hour"],
                "Total_Capacity": parsed["Total_Capacity"],
                "Previous_Occupied": parsed["Previous_Occupied"],
                "Previous_Occupancy_Pct": result["previous_occupancy_pct"],
                "Predicted_Status": result["predicted_status"],
                "Confidence_Pct": result["confidence_pct"],
                "Estimated_Occupied": result["estimated_occupied"],
                "Estimated_Available": result["estimated_available"],
                "Threshold_Pct": threshold_pct,
            }
            append_prediction_history(history_row, HISTORY_PATH)

            status = result["predicted_status"]
            banner_cls = "status-banner-high" if status == "HIGH" else "status-banner-low"
            st.markdown(
                f"""
                <div class="{banner_cls}">
                    <h3 style="margin:0;">{parsed['Location']} — {status} AVAILABILITY</h3>
                    <p style="margin:0.4rem 0 0 0; font-size:1rem;">{result['recommendation']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

            res_col1, res_col2 = st.columns([1.35, 1])
            with res_col1:
                m1, m2 = st.columns(2)
                with m1:
                    _render_metric_card(
                        "Prediction Confidence",
                        f"{result['confidence_pct']:.1f}%",
                        f"P(HIGH)={result['prob_high_pct']:.1f}% · P(LOW)={result['prob_low_pct']:.1f}%",
                    )
                    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
                    _render_metric_card(
                        "Estimated Occupied Spaces",
                        f"{result['estimated_occupied']} / {parsed['Total_Capacity']}",
                        f"Estimated Occupancy: {result['estimated_occupancy_pct']:.1f}%",
                    )
                with m2:
                    _render_metric_card(
                        "Estimated Available Spaces",
                        f"{result['estimated_available']} spaces",
                        f"Threshold: >{threshold_pct:.0f}% open = HIGH",
                    )
                    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
                    _render_metric_card(
                        "Previous Occupancy %",
                        f"{result['previous_occupancy_pct']:.1f}%",
                        f"Input: {parsed['Previous_Occupied']} occupied",
                    )

            with res_col2:
                st.plotly_chart(
                    _build_availability_gauge(result["estimated_availability_pct"], status),
                    use_container_width=True,
                )

            st.markdown("#### Prediction Explanation")
            for bullet in result["explanations"]:
                st.markdown(f"- {bullet}")

    st.divider()
    st.subheader("Recent Prediction History")
    history_df = load_prediction_history(HISTORY_PATH)
    if history_df.empty:
        st.info("No predictions have been logged yet. Submit the form above to record your first prediction.")
    else:
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Prediction History Report (CSV)",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="smartpark_prediction_history.csv",
            mime="text/csv",
        )


def render_analytics_page(df: pd.DataFrame) -> None:
    """Section 3: Location-Filtered Parking Analytics."""
    st.subheader("Campus Parking Analytics & Utilization Insights")
    locations = ["All Campus Zones"] + sorted(df["Location"].unique().tolist())
    selected_loc = st.selectbox("Filter Analytics by Parking Location", locations, key="analytics_loc")
    scoped_df = filter_dataset_by_location(df, selected_loc)
    summary = compute_dataset_summary(scoped_df)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _render_metric_card("Average Occupancy %", f"{summary['avg_occupancy_pct']:.1f}%", f"Scope: {selected_loc}")
    with k2:
        _render_metric_card("Average Available Spaces", f"{summary['avg_available']} spaces", f"Avg Cap: {summary['avg_capacity']}")
    with k3:
        _render_metric_card("HIGH Availability Frequency", f"{summary['high_pct']:.1f}%", f"{summary['high_count']} intervals")
    with k4:
        _render_metric_card("LOW Availability Frequency", f"{summary['low_pct']:.1f}%", f"{summary['low_count']} intervals")

    st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)

    col_peak, col_quiet, col_day = st.columns(3)
    with col_peak:
        st.markdown("#### Busiest Parking Hours")
        for item in summary["busiest_hours"]:
            st.write(f"- **{item['hour']:02d}:00** — {item['avg_occ_pct']:.1f}% avg occupancy")
    with col_quiet:
        st.markdown("#### Least Busy Parking Hours")
        for item in summary["least_busy_hours"]:
            st.write(f"- **{item['hour']:02d}:00** — {item['avg_occ_pct']:.1f}% avg occupancy")
    with col_day:
        st.markdown("#### Busiest Weekdays")
        for item in summary["busiest_days"][:3]:
            st.write(f"- **{item['day']}** — {item['avg_occ_pct']:.1f}% avg occupancy")

    st.success(summary["best_time_recommendation"])

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_weekday_occupancy_comparison(scoped_df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_occupancy_by_hour(scoped_df), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_availability_by_day(scoped_df), use_container_width=True)
    with c4:
        st.plotly_chart(plot_weekly_trends(scoped_df), use_container_width=True)


def render_historical_data_page(df: pd.DataFrame, threshold_pct: float) -> None:
    """Section 4: Historical Data Explorer & Persistent Custom CSV Upload."""
    st.subheader("Historical Parking Records & Dataset Management")

    with st.expander("Upload Custom CSV Dataset or Download Templates", expanded=False):
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button(
                "Download CSV Upload Template",
                data=get_csv_template().encode("utf-8"),
                file_name="smartpark_upload_template.csv",
                mime="text/csv",
            )
        with dl2:
            if os.path.exists(DATA_PATH):
                with open(DATA_PATH, "rb") as f:
                    st.download_button(
                        "Download Sample Synthetic Dataset",
                        data=f.read(),
                        file_name="sample_parking_data.csv",
                        mime="text/csv",
                    )
        with dl3:
            st.download_button(
                "Download Processed Active Dataset",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="processed_parking_data.csv",
                mime="text/csv",
            )

        uploaded_file = st.file_uploader("Upload Custom Campus Parking CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                raw_upload = pd.read_csv(uploaded_file)
                st.markdown("**Uploaded Data Preview (First 10 Rows):**")
                st.dataframe(raw_upload.head(10), use_container_width=True)

                cleaned_upload, val_summary = validate_and_clean_dataset(
                    raw_upload, availability_threshold_pct=threshold_pct
                )
                st.success(
                    f"Validation passed: {val_summary['valid_rows']} valid records "
                    f"({val_summary['dropped_rows']} invalid/missing rows removed)."
                )
                for issue in val_summary["issues"]:
                    st.warning(issue)

                if st.button("Train Model on Uploaded Dataset", type="primary"):
                    new_bundle = train_and_evaluate_models(
                        cleaned_upload,
                        availability_threshold_pct=threshold_pct,
                        model_save_path=MODEL_PATH,
                    )
                    os.makedirs(os.path.dirname(ACTIVE_DATA_PATH), exist_ok=True)
                    raw_upload.to_csv(ACTIVE_DATA_PATH, index=False)
                    st.session_state["active_df"] = cleaned_upload
                    st.session_state["validation_summary"] = val_summary
                    st.session_state["model_bundle"] = new_bundle
                    st.session_state["data_source_mode"] = "CUSTOM"
                    _persist_config(threshold_pct, "CUSTOM", "All Campus Zones")
                    st.success("Model retrained and custom dataset saved to disk across restarts.")
                    st.rerun()
            except (DatasetValidationError, ModelTrainingError, Exception) as exc:
                st.error(f"Dataset upload error: {exc}")

        if st.session_state.get("data_source_mode") == "CUSTOM":
            if st.button("Reset to Synthetic Sample Dataset"):
                if os.path.exists(ACTIVE_DATA_PATH):
                    os.remove(ACTIVE_DATA_PATH)
                if not os.path.exists(DATA_PATH):
                    save_sample_dataset(DATA_PATH, availability_threshold_pct=threshold_pct)
                bundle, cleaned_df, summary = load_or_train_model(
                    DATA_PATH, MODEL_PATH, threshold_pct, force_retrain=True
                )
                st.session_state["active_df"] = cleaned_df
                st.session_state["validation_summary"] = summary
                st.session_state["model_bundle"] = bundle
                st.session_state["data_source_mode"] = "SYNTHETIC"
                _persist_config(threshold_pct, "SYNTHETIC", "All Campus Zones")
                st.rerun()

    st.markdown("#### Filter Historical Records")
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        search_query = st.text_input("Search", "")
    with f2:
        loc_filter = st.multiselect("Location", sorted(df["Location"].unique().tolist()), default=[])
    with f3:
        day_filter = st.multiselect("Day", VALID_DAYS, default=[])
    with f4:
        status_filter = st.selectbox("Status", ["All", "HIGH", "LOW"])
    with f5:
        available_dates = sorted(df["Date"].astype(str).unique().tolist())
        date_filter = st.selectbox("Date", ["All Dates"] + available_dates)

    filtered_df = df.copy()
    if loc_filter:
        filtered_df = filtered_df[filtered_df["Location"].isin(loc_filter)]
    if day_filter:
        filtered_df = filtered_df[filtered_df["Day"].isin(day_filter)]
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df["Availability_Status"] == status_filter]
    if date_filter != "All Dates":
        filtered_df = filtered_df[filtered_df["Date"].astype(str) == date_filter]
    if search_query.strip():
        q = search_query.strip().lower()
        mask = (
            filtered_df["Date"].astype(str).str.lower().str.contains(q)
            | filtered_df["Time"].astype(str).str.lower().str.contains(q)
            | filtered_df["Location"].astype(str).str.lower().str.contains(q)
            | filtered_df["Day"].astype(str).str.lower().str.contains(q)
            | filtered_df["Availability_Status"].astype(str).str.lower().str.contains(q)
        )
        filtered_df = filtered_df[mask]

    st.caption(f"Showing **{len(filtered_df):,}** of **{len(df):,}** records.")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download Filtered Records (CSV)",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="smartpark_filtered_records.csv",
        mime="text/csv",
    )


def render_model_performance_page(model_bundle: dict) -> None:
    """Section 5: Model Performance & Comparison Page."""
    st.subheader("Machine Learning Model Performance & Evaluation")

    rf = model_bundle["primary_metrics"]
    dt = model_bundle["comparison_metrics"]

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        _render_metric_card("Primary Algorithm", "Random Forest", f"Trained {model_bundle['trained_at']}")
    with m2:
        _render_metric_card("Training Records", f"{model_bundle['train_records']:,}", "80% Stratified Split")
    with m3:
        _render_metric_card("Testing Records", f"{model_bundle['test_records']:,}", "20% Held-Out Set")
    with m4:
        _render_metric_card("Accuracy", f"{rf['accuracy']*100:.2f}%", f"DT: {dt['accuracy']*100:.2f}%")
    with m5:
        _render_metric_card("Precision / Recall", f"{rf['precision']*100:.1f}% / {rf['recall']*100:.1f}%", "Weighted Avg")
    with m6:
        _render_metric_card("F1-Score", f"{rf['f1_score']*100:.2f}%", f"DT: {dt['f1_score']*100:.2f}%")

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_model_comparison(rf, dt), use_container_width=True)
    with c2:
        st.plotly_chart(plot_feature_importance(model_bundle["feature_importances"]), use_container_width=True)

    cm1, cm2 = st.columns(2)
    with cm1:
        st.plotly_chart(
            plot_confusion_matrix(rf["confusion_matrix"], rf["labels"], "Random Forest Confusion Matrix"),
            use_container_width=True,
        )
    with cm2:
        st.plotly_chart(
            plot_confusion_matrix(dt["confusion_matrix"], dt["labels"], "Decision Tree Confusion Matrix"),
            use_container_width=True,
        )

    st.markdown("#### Why Random Forest Performs Better Than a Single Decision Tree")
    st.write(
        "A single Decision Tree learns one sequential hierarchy of splits, which can overfit minor fluctuations "
        "in historical campus occupancy. Random Forest builds 150 independent trees using bootstrapped samples "
        "and random feature subsets, averaging their predicted class probabilities across locations and hours."
    )


def render_about_page() -> None:
    """Section 6: About Project."""
    st.subheader("About SmartPark AI 2.0")
    st.write(
        "SmartPark AI 2.0 expands the campus parking availability predictor across **4 distinct campus zones** "
        "with a dedicated commuter **Parking Finder**, location-isolated **Analytics Dashboard**, leak-free "
        "chained future forecasts, and persistent on-disk storage."
    )
    st.markdown(f"**Transparency Notice:** {DATA_SOURCE_DISCLAIMER}")


def main() -> None:
    cfg = load_app_config(CONFIG_PATH)
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=96)
        st.markdown("### SMARTPARK AI 2.0")
        st.caption("Intelligent Campus Parking Predictor")

        page = st.radio(
            "Navigation",
            [
                "Parking Finder",
                "Dashboard",
                "Predict Availability",
                "Parking Analytics",
                "Historical Data",
                "Model Performance",
                "About Project",
            ],
        )

        st.divider()
        threshold_pct = float(
            st.slider(
                "HIGH Availability Threshold (% open)",
                min_value=10,
                max_value=60,
                value=int(cfg.get("threshold_pct", 30)),
                step=5,
                help="HIGH if available spaces > threshold%; LOW if <= threshold%.",
            )
        )

        mode_label = (
            "CUSTOM UPLOADED CSV"
            if st.session_state.get("data_source_mode") == "CUSTOM"
            else "DEMONSTRATION / SYNTHETIC DATA"
        )
        st.caption(f"Active Dataset: **{mode_label}**")

    _init_session_state(threshold_pct)
    _render_header()

    if st.session_state.get("data_source_mode", "SYNTHETIC") == "SYNTHETIC":
        st.markdown(
            f'<div class="disclaimer-box"><strong>Dataset Notice:</strong> {DATA_SOURCE_DISCLAIMER}</div>',
            unsafe_allow_html=True,
        )

    active_df: pd.DataFrame = st.session_state["active_df"]
    model_bundle: dict = st.session_state["model_bundle"]

    if page == "Parking Finder":
        render_parking_finder(active_df, model_bundle, threshold_pct)
    elif page == "Dashboard":
        render_dashboard(active_df, model_bundle, threshold_pct)
    elif page == "Predict Availability":
        render_predict_page(active_df, model_bundle, threshold_pct)
    elif page == "Parking Analytics":
        render_analytics_page(active_df)
    elif page == "Historical Data":
        render_historical_data_page(active_df, threshold_pct)
    elif page == "Model Performance":
        render_model_performance_page(model_bundle)
    elif page == "About Project":
        render_about_page()


if __name__ == "__main__":
    main()
