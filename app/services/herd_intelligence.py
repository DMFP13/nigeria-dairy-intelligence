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
