from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd


CORE_COLUMN_ALIASES: dict[str, list[str]] = {
    "animal_id": ["animal_id", "animal", "cow_id", "cow", "id", "tag_id", "tag"],
    "date": ["date", "datetime", "timestamp", "event_date", "record_date"],
    "rumination_min": ["rumination_min", "rumination", "rumination_minutes", "rumination_mins"],
    "activity_rate": ["activity_rate", "activity", "activity_index", "activity_score"],
    "standing_min": ["standing_min", "standing", "standing_minutes"],
    "eating_min": ["eating_min", "eating", "eating_minutes", "feeding_min"],
}

REQUIRED_COLUMNS = ["animal_id", "date", "rumination_min", "activity_rate"]
GROUPING_CANDIDATES = ["pen_id", "group_id", "pen", "group", "cohort"]


def _normalize_column_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")


def read_uploaded_dataset(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    lower_name = file_name.lower()
    buffer = BytesIO(file_bytes)

    if lower_name.endswith(".csv"):
        return pd.read_csv(buffer)
    if lower_name.endswith(".xlsx"):
        return pd.read_excel(buffer)

    raise ValueError("Unsupported file format. Upload a CSV or XLSX file.")


def standardize_behavioural_columns(df: pd.DataFrame) -> pd.DataFrame:
    standardized = df.copy()
    normalized_lookup = {_normalize_column_name(col): col for col in standardized.columns}

    rename_map: dict[str, str] = {}
    for canonical, aliases in CORE_COLUMN_ALIASES.items():
        if canonical in standardized.columns:
            continue

        for alias in aliases:
            match = normalized_lookup.get(_normalize_column_name(alias))
            if match and match not in rename_map:
                rename_map[match] = canonical
                break

    if rename_map:
        standardized = standardized.rename(columns=rename_map)

    return standardized


def parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df.copy()
    if "date" in parsed.columns:
        parsed["date"] = pd.to_datetime(parsed["date"], errors="coerce")
    return parsed


def filter_dataset_by_period(
    df: pd.DataFrame,
    period_option: str,
    custom_start: pd.Timestamp | None = None,
    custom_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if "date" not in df.columns:
        return df.copy()

    base = df.dropna(subset=["date"]).copy()
    if base.empty:
        return base

    end_date = base["date"].max().normalize()

    if period_option == "7 days":
        start_date = end_date - pd.Timedelta(days=6)
    elif period_option == "30 days":
        start_date = end_date - pd.Timedelta(days=29)
    elif period_option == "custom range" and custom_start is not None and custom_end is not None:
        start_date = pd.Timestamp(custom_start).normalize()
        end_date = pd.Timestamp(custom_end).normalize()
    else:
        return base

    return base[(base["date"] >= start_date) & (base["date"] <= end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))].copy()


def build_validation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for col in REQUIRED_COLUMNS:
        rows.append(
            {
                "check": f"column:{col}",
                "status": "OK" if col in df.columns else "Missing",
                "detail": "available" if col in df.columns else "not found",
            }
        )

    if "date" in df.columns:
        missing_date = int(df["date"].isna().sum())
        rows.append(
            {
                "check": "date_parse",
                "status": "OK" if missing_date == 0 else "Warning",
                "detail": f"rows with invalid date: {missing_date}",
            }
        )

    if "animal_id" in df.columns:
        missing_animal = int(df["animal_id"].isna().sum())
        rows.append(
            {
                "check": "animal_id_missing",
                "status": "OK" if missing_animal == 0 else "Warning",
                "detail": f"rows with missing animal_id: {missing_animal}",
            }
        )

    for metric in ["rumination_min", "activity_rate"]:
        if metric in df.columns:
            missing_metric = int(df[metric].isna().sum())
            rows.append(
                {
                    "check": f"missing:{metric}",
                    "status": "OK" if missing_metric == 0 else "Warning",
                    "detail": f"rows with missing {metric}: {missing_metric}",
                }
            )

    return pd.DataFrame(rows)


def compute_data_completeness(df: pd.DataFrame) -> float:
    required_present = [col for col in REQUIRED_COLUMNS if col in df.columns]
    if not required_present or df.empty:
        return 0.0

    completeness = 1 - float(df[required_present].isna().sum().sum()) / float(len(df) * len(required_present))
    return max(0.0, min(1.0, completeness))


def compute_herd_metrics(df: pd.DataFrame) -> dict[str, Any]:
    animals_detected = int(df["animal_id"].nunique()) if "animal_id" in df.columns else 0
    records_loaded = int(len(df))

    date_range = "N/A"
    if "date" in df.columns:
        valid_dates = df["date"].dropna()
        if not valid_dates.empty:
            date_range = f"{valid_dates.min().date()} to {valid_dates.max().date()}"

    return {
        "animals_detected": animals_detected,
        "records_loaded": records_loaded,
        "date_range": date_range,
        "variables_available": int(len(df.columns)),
    }


def compute_herd_timeseries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "date" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    base = df.dropna(subset=["date"]).copy()
    if base.empty:
        return pd.DataFrame(), pd.DataFrame()

    base["date_day"] = base["date"].dt.floor("D")

    activity_ts = pd.DataFrame()
    if "activity_rate" in base.columns:
        activity_ts = (
            base.groupby("date_day", as_index=False)["activity_rate"]
            .mean()
            .rename(columns={"date_day": "date", "activity_rate": "avg_activity_rate"})
        )

    rumination_ts = pd.DataFrame()
    if "rumination_min" in base.columns:
        rumination_ts = (
            base.groupby("date_day", as_index=False)["rumination_min"]
            .mean()
            .rename(columns={"date_day": "date", "rumination_min": "avg_rumination_min"})
        )

    return activity_ts, rumination_ts


def compute_animal_coverage(df: pd.DataFrame) -> pd.DataFrame:
    if "animal_id" not in df.columns:
        return pd.DataFrame(columns=["animal_id", "record_count", "completeness_pct", "avg_rumination_min", "avg_activity_rate", "rumination_gap_pct"])

    required_present = [col for col in REQUIRED_COLUMNS if col in df.columns]
    herd_rumination = float(df["rumination_min"].mean()) if "rumination_min" in df.columns else float("nan")

    rows: list[dict[str, Any]] = []
    for animal_id, group in df.groupby(df["animal_id"].fillna("UNKNOWN").astype(str)):
        record_count = int(len(group))
        completeness_pct = 0.0
        if required_present and record_count > 0:
            non_null = int(group[required_present].notna().sum().sum())
            completeness_pct = 100.0 * non_null / float(record_count * len(required_present))

        avg_rum = float(group["rumination_min"].mean()) if "rumination_min" in group.columns else float("nan")
        avg_act = float(group["activity_rate"].mean()) if "activity_rate" in group.columns else float("nan")

        rum_gap = 0.0
        if pd.notna(avg_rum) and pd.notna(herd_rumination) and herd_rumination != 0:
            rum_gap = ((avg_rum - herd_rumination) / herd_rumination) * 100.0

        rows.append(
            {
                "animal_id": animal_id,
                "record_count": record_count,
                "completeness_pct": round(completeness_pct, 1),
                "avg_rumination_min": round(avg_rum, 2) if pd.notna(avg_rum) else None,
                "avg_activity_rate": round(avg_act, 2) if pd.notna(avg_act) else None,
                "rumination_gap_pct": round(rum_gap, 2),
            }
        )

    return pd.DataFrame(rows).sort_values("record_count", ascending=False)


def build_animals_to_review_table(df: pd.DataFrame) -> pd.DataFrame:
    coverage = compute_animal_coverage(df)
    if coverage.empty:
        return coverage

    review = coverage.copy()
    review["coverage_risk"] = 100.0 - review["completeness_pct"]
    review["deviation_risk"] = review["rumination_gap_pct"].abs()
    review["review_score"] = review["coverage_risk"] * 0.6 + review["deviation_risk"] * 0.4

    cols = [
        "animal_id",
        "record_count",
        "completeness_pct",
        "rumination_gap_pct",
        "review_score",
    ]
    return review[cols].sort_values(["review_score", "record_count"], ascending=[False, False]).head(20)


def compute_animal_record_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "animal_id" not in df.columns:
        return pd.DataFrame(columns=["animal_id", "record_count"])

    counts = (
        df["animal_id"]
        .fillna("UNKNOWN")
        .astype(str)
        .value_counts()
        .rename_axis("animal_id")
        .reset_index(name="record_count")
    )
    return counts


def detect_group_column(df: pd.DataFrame) -> str | None:
    for col in GROUPING_CANDIDATES:
        if col in df.columns:
            return col
    return None


def compute_group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()

    grouped = df.copy()
    grouped[group_col] = grouped[group_col].fillna("UNKNOWN").astype(str)

    agg = grouped.groupby(group_col, as_index=False).agg(
        animals_monitored=("animal_id", "nunique") if "animal_id" in grouped.columns else (group_col, "count"),
        records_loaded=(group_col, "count"),
    )

    if "rumination_min" in grouped.columns:
        agg["avg_rumination_min"] = grouped.groupby(group_col)["rumination_min"].mean().values
    if "activity_rate" in grouped.columns:
        agg["avg_activity_rate"] = grouped.groupby(group_col)["activity_rate"].mean().values

    return agg.sort_values("records_loaded", ascending=False)


def compute_cow_metrics(df: pd.DataFrame, animal_id: str) -> tuple[dict[str, Any], pd.DataFrame]:
    if "animal_id" not in df.columns:
        return {}, pd.DataFrame()

    cow_df = df[df["animal_id"].astype(str) == str(animal_id)].copy()
    if cow_df.empty:
        return {}, cow_df

    mean_rumination = float(cow_df["rumination_min"].mean()) if "rumination_min" in cow_df.columns else float("nan")
    baseline_threshold = mean_rumination * 0.8 if pd.notna(mean_rumination) else float("nan")

    if "rumination_min" in cow_df.columns and pd.notna(baseline_threshold):
        cow_df["rumination_anomaly"] = cow_df["rumination_min"] < baseline_threshold
    else:
        cow_df["rumination_anomaly"] = False

    metrics = {
        "animal_id": animal_id,
        "record_count": int(len(cow_df)),
        "avg_rumination_min": round(mean_rumination, 2) if pd.notna(mean_rumination) else None,
        "avg_activity_rate": round(float(cow_df["activity_rate"].mean()), 2) if "activity_rate" in cow_df.columns else None,
        "anomaly_count": int(cow_df["rumination_anomaly"].sum()),
    }

    return metrics, cow_df


def compute_cow_vs_herd_comparison(df: pd.DataFrame, cow_df: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    for col in ["rumination_min", "activity_rate", "standing_min", "eating_min"]:
        if col not in df.columns or col not in cow_df.columns:
            continue

        herd_avg = float(df[col].mean())
        cow_avg = float(cow_df[col].mean())
        diff_pct = 0.0
        if pd.notna(herd_avg) and herd_avg != 0:
            diff_pct = ((cow_avg - herd_avg) / herd_avg) * 100.0

        metrics.append(
            {
                "metric": col,
                "cow_average": round(cow_avg, 2) if pd.notna(cow_avg) else None,
                "herd_average": round(herd_avg, 2) if pd.notna(herd_avg) else None,
                "difference_pct": round(diff_pct, 2),
            }
        )

    return pd.DataFrame(metrics)


def build_cow_timeline(cow_df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in cow_df.columns:
        return pd.DataFrame()

    base = cow_df.dropna(subset=["date"]).copy()
    if base.empty:
        return pd.DataFrame()

    base["date_day"] = base["date"].dt.floor("D")

    agg_map: dict[str, str] = {}
    for col in ["rumination_min", "activity_rate", "standing_min", "eating_min"]:
        if col in base.columns:
            agg_map[col] = "mean"

    if not agg_map:
        return pd.DataFrame()

    timeline = base.groupby("date_day", as_index=False).agg(agg_map).rename(columns={"date_day": "date"})
    return timeline


def build_cow_event_table(cow_df: pd.DataFrame, comparison_df: pd.DataFrame) -> pd.DataFrame:
    if cow_df.empty:
        return pd.DataFrame()

    events = cow_df.copy()

    herd_baseline: dict[str, float] = {}
    if not comparison_df.empty:
        for _, row in comparison_df.iterrows():
            if row["metric"] and pd.notna(row["herd_average"]):
                herd_baseline[str(row["metric"])] = float(row["herd_average"])

    if "rumination_min" in events.columns:
        events["low_rumination_vs_herd"] = events["rumination_min"] < (0.85 * herd_baseline.get("rumination_min", events["rumination_min"].mean()))
    if "activity_rate" in events.columns:
        events["low_activity_vs_herd"] = events["activity_rate"] < (0.85 * herd_baseline.get("activity_rate", events["activity_rate"].mean()))

    if "rumination_anomaly" not in events.columns:
        events["rumination_anomaly"] = False

    keep_cols = [col for col in ["date", "animal_id", "rumination_min", "activity_rate", "standing_min", "eating_min", "rumination_anomaly", "low_rumination_vs_herd", "low_activity_vs_herd"] if col in events.columns]

    event_df = events[keep_cols].copy()

    flag_cols = [col for col in ["rumination_anomaly", "low_rumination_vs_herd", "low_activity_vs_herd"] if col in event_df.columns]
    if flag_cols:
        event_df = event_df[event_df[flag_cols].any(axis=1)]

    return event_df.sort_values("date", ascending=False).head(50)


def process_uploaded_dataset(file_name: str, file_bytes: bytes) -> dict[str, Any]:
    raw_df = read_uploaded_dataset(file_name, file_bytes)
    standardized_df = standardize_behavioural_columns(raw_df)
    parsed_df = parse_date_column(standardized_df)

    herd_metrics = compute_herd_metrics(parsed_df)
    activity_ts, rumination_ts = compute_herd_timeseries(parsed_df)
    animal_counts = compute_animal_record_counts(parsed_df)
    validation_summary = build_validation_summary(parsed_df)

    return {
        "processed_df": parsed_df,
        "herd_metrics": herd_metrics,
        "activity_timeseries": activity_ts,
        "rumination_timeseries": rumination_ts,
        "animal_counts": animal_counts,
        "validation_summary": validation_summary,
    }
