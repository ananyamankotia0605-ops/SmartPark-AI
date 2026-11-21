"""
analytics.py
------------
Computes location-scoped historical parking statistics, accurate single-lot and
multi-zone capacity math, and interactive Plotly visualizations for SmartPark AI 2.0.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_processor import VALID_DAYS

NAVY_PRIMARY = "#0B192C"
NAVY_SECONDARY = "#1E3A8A"
SKY_ACCENT = "#0284C7"
HIGH_GREEN = "#16A34A"
LOW_RED = "#DC2626"


def filter_dataset_by_location(df: pd.DataFrame, location: Optional[str]) -> pd.DataFrame:
    """Return records strictly belonging to `location`, or all records if 'All Campus Zones'."""
    if df is None or df.empty or not location or location == "All Campus Zones":
        return df
    return df[df["Location"] == location].copy()


def get_location_snapshots(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Compute each parking location's independent capacity and latest historical
    occupancy snapshot strictly from its own records.
    """
    if df is None or df.empty:
        return []

    snapshots: List[Dict[str, Any]] = []
    for loc in sorted(df["Location"].unique().tolist()):
        loc_df = df[df["Location"] == loc].sort_values(["Date", "Time", "Hour"])
        latest = loc_df.iloc[-1]
        cap = int(latest["Total_Capacity"])
        avail = int(latest["Available_Spaces"])
        occ = max(0, cap - avail)
        avail_pct = round((avail / cap) * 100.0, 1) if cap > 0 else 0.0
        occ_pct = round((occ / cap) * 100.0, 1) if cap > 0 else 0.0

        snapshots.append(
            {
                "location": loc,
                "total_capacity": cap,
                "latest_occupied": occ,
                "latest_available": avail,
                "latest_availability_pct": avail_pct,
                "latest_occupancy_pct": occ_pct,
                "latest_status": str(latest["Availability_Status"]),
                "latest_date": str(latest["Date"]),
                "latest_time": str(latest["Time"]),
                "latest_day": str(latest["Day"]),
                "latest_hour": int(latest["Hour"]),
                "record_count": int(len(loc_df)),
            }
        )
    return snapshots


def compute_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate core KPI metrics and accurate multi-zone or single-zone capacity totals."""
    if df is None or df.empty:
        return {
            "total_records": 0,
            "total_capacity": 0,
            "latest_occupied": 0,
            "latest_available": 0,
            "latest_availability_pct": 0.0,
            "avg_capacity": 0,
            "avg_occupied": 0,
            "avg_available": 0,
            "avg_occupancy_pct": 0.0,
            "avg_availability_pct": 0.0,
            "high_count": 0,
            "low_count": 0,
            "high_pct": 0.0,
            "low_pct": 0.0,
            "busiest_hours": [],
            "least_busy_hours": [],
            "busiest_days": [],
            "best_time_recommendation": "No historical data available.",
        }

    work_df = df.copy()
    work_df["Actual_Occupied"] = work_df["Total_Capacity"] - work_df["Available_Spaces"]
    work_df["Actual_Occupancy_Pct"] = (
        work_df["Actual_Occupied"] / work_df["Total_Capacity"]
    ) * 100.0

    snapshots = get_location_snapshots(work_df)
    total_capacity = sum(s["total_capacity"] for s in snapshots)
    latest_occupied = sum(s["latest_occupied"] for s in snapshots)
    latest_available = sum(s["latest_available"] for s in snapshots)
    latest_availability_pct = (
        round((latest_available / total_capacity) * 100.0, 1) if total_capacity > 0 else 0.0
    )

    total_records = int(len(work_df))
    avg_capacity = int(round(work_df["Total_Capacity"].mean()))
    avg_available = int(round(work_df["Available_Spaces"].mean()))
    avg_occupied = max(0, avg_capacity - avg_available)
    avg_occupancy_pct = round(float(work_df["Actual_Occupancy_Pct"].mean()), 1)
    avg_availability_pct = round(100.0 - avg_occupancy_pct, 1)

    high_count = int((work_df["Availability_Status"] == "HIGH").sum())
    low_count = int((work_df["Availability_Status"] == "LOW").sum())
    high_pct = round((high_count / total_records) * 100.0, 1)
    low_pct = round((low_count / total_records) * 100.0, 1)

    hourly_stats = (
        work_df.groupby("Hour")["Actual_Occupancy_Pct"]
        .mean()
        .reset_index()
        .sort_values("Actual_Occupancy_Pct", ascending=False)
    )
    busiest_hours = [
        {"hour": int(row["Hour"]), "avg_occ_pct": round(float(row["Actual_Occupancy_Pct"]), 1)}
        for _, row in hourly_stats.head(3).iterrows()
    ]
    least_busy_hours = [
        {"hour": int(row["Hour"]), "avg_occ_pct": round(float(row["Actual_Occupancy_Pct"]), 1)}
        for _, row in hourly_stats.tail(3).sort_values("Actual_Occupancy_Pct").iterrows()
    ]

    day_stats = (
        work_df.groupby("Day")["Actual_Occupancy_Pct"]
        .mean()
        .reset_index()
        .sort_values("Actual_Occupancy_Pct", ascending=False)
    )
    busiest_days = [
        {"day": str(row["Day"]), "avg_occ_pct": round(float(row["Actual_Occupancy_Pct"]), 1)}
        for _, row in day_stats.iterrows()
    ]

    if least_busy_hours:
        best_h = least_busy_hours[0]
        quiet_str = ", ".join([f"{item['hour']:02d}:00" for item in least_busy_hours[:3]])
        best_time_recommendation = (
            f"Best arrival windows based on historical occupancy: {quiet_str} "
            f"(lowest average occupancy at {best_h['hour']:02d}:00 with {best_h['avg_occ_pct']:.1f}% filled)."
        )
    else:
        best_time_recommendation = "Insufficient hourly diversity for time recommendations."

    return {
        "total_records": total_records,
        "total_capacity": total_capacity,
        "latest_occupied": latest_occupied,
        "latest_available": latest_available,
        "latest_availability_pct": latest_availability_pct,
        "avg_capacity": avg_capacity,
        "avg_occupied": avg_occupied,
        "avg_available": avg_available,
        "avg_occupancy_pct": avg_occupancy_pct,
        "avg_availability_pct": avg_availability_pct,
        "high_count": high_count,
        "low_count": low_count,
        "high_pct": high_pct,
        "low_pct": low_pct,
        "busiest_hours": busiest_hours,
        "least_busy_hours": least_busy_hours,
        "busiest_days": busiest_days,
        "best_time_recommendation": best_time_recommendation,
    }


def _apply_clean_layout(fig: go.Figure, title: str, xaxis_title: str = "", yaxis_title: str = "") -> go.Figure:
    """Apply consistent SmartPark AI Navy & White styling to Plotly figures."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=NAVY_PRIMARY)),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(248,250,252,1)",
        font=dict(color="#1E293B", size=12),
        xaxis=dict(title=xaxis_title, gridcolor="#E2E8F0", zeroline=False),
        yaxis=dict(title=yaxis_title, gridcolor="#E2E8F0", zeroline=False),
        margin=dict(l=40, r=24, t=48, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_occupancy_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="Occupancy_Percentage",
        nbins=20,
        color="Availability_Status",
        color_discrete_map={"HIGH": HIGH_GREEN, "LOW": LOW_RED},
        opacity=0.88,
    )
    return _apply_clean_layout(
        fig,
        "Parking Occupancy Distribution",
        xaxis_title="Occupancy Percentage (%)",
        yaxis_title="Record Count",
    )


def plot_availability_by_day(df: pd.DataFrame) -> go.Figure:
    grouped = df.groupby(["Day", "Availability_Status"]).size().reset_index(name="Records")
    fig = px.bar(
        grouped,
        x="Day",
        y="Records",
        color="Availability_Status",
        barmode="group",
        category_orders={"Day": VALID_DAYS},
        color_discrete_map={"HIGH": HIGH_GREEN, "LOW": LOW_RED},
    )
    return _apply_clean_layout(
        fig,
        "Parking Availability by Day of Week",
        xaxis_title="Day",
        yaxis_title="Number of Intervals",
    )


def plot_occupancy_by_hour(df: pd.DataFrame) -> go.Figure:
    work_df = df.copy()
    work_df["Actual_Occupancy_Pct"] = (
        (work_df["Total_Capacity"] - work_df["Available_Spaces"])
        / work_df["Total_Capacity"]
    ) * 100.0
    hourly = work_df.groupby("Hour")["Actual_Occupancy_Pct"].mean().reset_index().sort_values("Hour")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hourly["Hour"],
            y=hourly["Actual_Occupancy_Pct"].round(1),
            mode="lines+markers",
            name="Avg Occupancy %",
            line=dict(color=NAVY_SECONDARY, width=3),
            marker=dict(size=7, color=SKY_ACCENT),
            fill="tozeroy",
            fillcolor="rgba(2, 132, 199, 0.12)",
        )
    )
    fig.update_xaxes(dtick=1)
    return _apply_clean_layout(
        fig,
        "Average Parking Occupancy by Hour",
        xaxis_title="Hour of Day (24h)",
        yaxis_title="Average Occupancy (%)",
    )


def plot_weekly_trends(df: pd.DataFrame) -> go.Figure:
    work_df = df.copy()
    work_df["Parsed_Date"] = pd.to_datetime(work_df["Date"], errors="coerce")
    work_df = work_df.dropna(subset=["Parsed_Date"])
    if work_df.empty:
        return _apply_clean_layout(go.Figure(), "Weekly Parking Trends", "Week", "Occupancy (%)")

    work_df["Week_Start"] = work_df["Parsed_Date"].dt.to_period("W").apply(
        lambda p: p.start_time.strftime("%Y-%m-%d")
    )
    work_df["Actual_Occupancy_Pct"] = (
        (work_df["Total_Capacity"] - work_df["Available_Spaces"])
        / work_df["Total_Capacity"]
    ) * 100.0
    weekly = work_df.groupby("Week_Start")["Actual_Occupancy_Pct"].mean().reset_index().sort_values("Week_Start")

    fig = px.line(
        weekly,
        x="Week_Start",
        y="Actual_Occupancy_Pct",
        markers=True,
        color_discrete_sequence=[NAVY_PRIMARY],
    )
    fig.update_traces(line=dict(width=2.5), marker=dict(size=7, color=SKY_ACCENT))
    return _apply_clean_layout(
        fig,
        "Weekly Parking Occupancy Trend",
        xaxis_title="Week Starting",
        yaxis_title="Average Occupancy (%)",
    )


def plot_availability_split(df: pd.DataFrame) -> go.Figure:
    counts = df["Availability_Status"].value_counts().reset_index()
    counts.columns = ["Availability_Status", "Count"]
    fig = px.pie(
        counts,
        names="Availability_Status",
        values="Count",
        hole=0.52,
        color="Availability_Status",
        color_discrete_map={"HIGH": HIGH_GREEN, "LOW": LOW_RED},
    )
    fig.update_traces(textinfo="label+percent+value")
    return _apply_clean_layout(fig, "Low vs High Availability Distribution")


def plot_weekday_occupancy_comparison(df: pd.DataFrame) -> go.Figure:
    work_df = df.copy()
    work_df["Actual_Occupancy_Pct"] = (
        (work_df["Total_Capacity"] - work_df["Available_Spaces"])
        / work_df["Total_Capacity"]
    ) * 100.0
    day_avg = (
        work_df.groupby("Day")["Actual_Occupancy_Pct"]
        .mean()
        .reindex(VALID_DAYS)
        .dropna()
        .reset_index()
    )
    fig = px.bar(
        day_avg,
        x="Day",
        y="Actual_Occupancy_Pct",
        text=day_avg["Actual_Occupancy_Pct"].round(1),
        color="Actual_Occupancy_Pct",
        color_continuous_scale=["#16A34A", "#F59E0B", "#DC2626"],
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_coloraxes(showscale=False)
    return _apply_clean_layout(
        fig,
        "Average Occupancy Percentage by Weekday",
        xaxis_title="Day of Week",
        yaxis_title="Average Occupancy (%)",
    )


def plot_feature_importance(importances: Dict[str, float]) -> go.Figure:
    items = [{"Feature": k.replace("_", " "), "Importance": v} for k, v in importances.items()]
    imp_df = pd.DataFrame(items).sort_values("Importance", ascending=True)
    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text=imp_df["Importance"].round(1),
        color_discrete_sequence=[NAVY_SECONDARY],
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    return _apply_clean_layout(
        fig,
        "Trained Model Feature Importance (%)",
        xaxis_title="Relative Importance (%)",
        yaxis_title="",
    )


def plot_confusion_matrix(cm: List[List[int]], labels: List[str], title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Predicted {l}" for l in labels],
            y=[f"Actual {l}" for l in labels],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _apply_clean_layout(fig, title)


def plot_model_comparison(rf_metrics: Dict[str, Any], dt_metrics: Dict[str, Any]) -> go.Figure:
    metric_keys = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1-Score"),
    ]
    rows = []
    for key, label in metric_keys:
        rows.append({"Metric": label, "Model": "Random Forest", "Score": round(rf_metrics[key] * 100, 2)})
        rows.append({"Metric": label, "Model": "Decision Tree", "Score": round(dt_metrics[key] * 100, 2)})
    comp_df = pd.DataFrame(rows)

    fig = px.bar(
        comp_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text=comp_df["Score"].round(1),
        color_discrete_map={"Random Forest": NAVY_PRIMARY, "Decision Tree": SKY_ACCENT},
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_yaxes(range=[0, 110])
    return _apply_clean_layout(
        fig,
        "Random Forest vs Decision Tree Evaluation (%)",
        xaxis_title="Evaluation Metric",
        yaxis_title="Score (%)",
    )
