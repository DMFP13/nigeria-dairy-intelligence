from __future__ import annotations

import pandas as pd


def build_demo_milk_production_trend(periods: int = 12) -> pd.DataFrame:
    months = pd.date_range(end=pd.Timestamp.today().normalize(), periods=periods, freq="M")
    return pd.DataFrame(
        {
            "month": months,
            "milk_production_liters_demo": [21000 + i * 450 + (i % 3) * 120 for i in range(periods)],
        }
    )


def build_demo_reproductive_trend(periods: int = 12) -> pd.DataFrame:
    months = pd.date_range(end=pd.Timestamp.today().normalize(), periods=periods, freq="M")
    return pd.DataFrame(
        {
            "month": months,
            "heat_detection_rate_demo": [58 + (i % 5) * 2 for i in range(periods)],
            "pregnancy_confirmation_rate_demo": [42 + (i % 4) * 3 for i in range(periods)],
        }
    )


def ensure_entity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "cow_id" not in out.columns and "animal_id" in out.columns:
        out["cow_id"] = out["animal_id"].astype(str)
    if "animal_id" not in out.columns and "cow_id" in out.columns:
        out["animal_id"] = out["cow_id"].astype(str)

    if "farm_id" not in out.columns:
        out["farm_id"] = "FARM-UPLOADED"
    if "farm_name" not in out.columns:
        out["farm_name"] = out["farm_id"].astype(str)
    if "farm_type" not in out.columns:
        out["farm_type"] = "Uploaded"

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    return out


def _mean(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns or df.empty:
        return None
    value = float(df[col].mean())
    return round(value, 2) if pd.notna(value) else None


def _subscore(value: float | None, target: float, tolerance: float) -> float:
    if value is None:
        return 0.0
    return max(0.0, 100.0 - (abs(value - target) / tolerance) * 100.0)


def assign_behavioral_rating(
    rumination: float | None,
    activity: float | None,
    eating: float | None,
    standing: float | None,
    completeness_pct: float,
) -> tuple[str, float]:
    score = (
        0.30 * _subscore(rumination, 300.0, 95.0)
        + 0.20 * _subscore(activity, 58.0, 24.0)
        + 0.20 * _subscore(eating, 250.0, 95.0)
        + 0.15 * _subscore(standing, 320.0, 130.0)
        + 0.15 * max(0.0, min(100.0, completeness_pct))
    )

    if score >= 85:
        return "A", round(score, 1)
    if score >= 75:
        return "B", round(score, 1)
    if score >= 65:
        return "C", round(score, 1)
    if score >= 55:
        return "D", round(score, 1)
    return "Review", round(score, 1)


def compute_data_completeness(df: pd.DataFrame) -> float:
    required = [c for c in ["date", "animal_id", "rumination_min", "activity_rate", "milk_yield_l"] if c in df.columns]
    if not required or df.empty:
        return 0.0
    completeness = 1.0 - float(df[required].isna().sum().sum()) / float(len(df) * len(required))
    return max(0.0, min(1.0, completeness))


def compute_network_kpis(df: pd.DataFrame) -> dict[str, float | int | None]:
    if df.empty:
        return {
            "farms": 0,
            "cows": 0,
            "total_milk_per_day": None,
            "avg_milk_per_cow_per_day": None,
            "avg_milk_per_farm_per_day": None,
        }

    farms = int(df["farm_id"].nunique()) if "farm_id" in df.columns else 0
    cows = int(df["animal_id"].nunique()) if "animal_id" in df.columns else 0

    if "milk_yield_l" in df.columns and "date" in df.columns:
        daily_total = df.dropna(subset=["date"]).groupby(df["date"].dt.floor("D"))["milk_yield_l"].sum()
        total_milk_per_day = float(daily_total.mean()) if not daily_total.empty else None
    else:
        total_milk_per_day = None

    avg_milk_per_cow = None
    if "milk_yield_l" in df.columns and "animal_id" in df.columns:
        by_cow = df.groupby("animal_id")["milk_yield_l"].mean()
        avg_milk_per_cow = float(by_cow.mean()) if not by_cow.empty else None

    avg_milk_per_farm = None
    if "milk_yield_l" in df.columns and "farm_id" in df.columns:
        by_farm = df.groupby("farm_id")["milk_yield_l"].mean()
        avg_milk_per_farm = float(by_farm.mean()) if not by_farm.empty else None

    return {
        "farms": farms,
        "cows": cows,
        "total_milk_per_day": round(total_milk_per_day, 2) if total_milk_per_day is not None else None,
        "avg_milk_per_cow_per_day": round(avg_milk_per_cow, 2) if avg_milk_per_cow is not None else None,
        "avg_milk_per_farm_per_day": round(avg_milk_per_farm, 2) if avg_milk_per_farm is not None else None,
    }


def compute_network_behaviour(df: pd.DataFrame) -> dict[str, float | None]:
    return {
        "avg_rumination": _mean(df, "rumination_min"),
        "avg_activity": _mean(df, "activity_rate"),
        "avg_eating": _mean(df, "eating_min"),
        "avg_standing": _mean(df, "standing_min"),
    }


def compute_farm_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    req = [c for c in ["date", "animal_id", "rumination_min", "activity_rate", "milk_yield_l"] if c in df.columns]

    for farm_id, fdf in df.groupby("farm_id"):
        record_count = int(len(fdf))
        completeness = 0.0
        if req and record_count > 0:
            non_null = int(fdf[req].notna().sum().sum())
            completeness = 100.0 * non_null / float(record_count * len(req))

        avg_rum = _mean(fdf, "rumination_min")
        avg_act = _mean(fdf, "activity_rate")
        avg_eat = _mean(fdf, "eating_min")
        avg_stand = _mean(fdf, "standing_min")
        rating, score = assign_behavioral_rating(avg_rum, avg_act, avg_eat, avg_stand, completeness)

        rows.append(
            {
                "farm_id": farm_id,
                "farm_name": str(fdf["farm_name"].iloc[0]) if "farm_name" in fdf.columns else farm_id,
                "farm_type": str(fdf["farm_type"].iloc[0]) if "farm_type" in fdf.columns else "Unknown",
                "cows": int(fdf["animal_id"].nunique()) if "animal_id" in fdf.columns else 0,
                "records_loaded": record_count,
                "avg_milk_yield_l": _mean(fdf, "milk_yield_l"),
                "data_completeness_pct": round(completeness, 1),
                "behavioral_rating": rating,
                "behavioral_score": score,
                "avg_rumination_min": avg_rum,
                "avg_activity_rate": avg_act,
            }
        )

    return pd.DataFrame(rows).sort_values(["behavioral_score", "avg_milk_yield_l"], ascending=[False, False])


def compute_farm_daily_trends(df: pd.DataFrame, farm_id: str) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()

    fdf = df[df["farm_id"].astype(str) == str(farm_id)].dropna(subset=["date"]).copy()
    if fdf.empty:
        return pd.DataFrame()

    fdf["date_day"] = fdf["date"].dt.floor("D")
    agg = fdf.groupby("date_day", as_index=False).agg(
        milk_yield_l=("milk_yield_l", "mean") if "milk_yield_l" in fdf.columns else ("date_day", "count"),
        rumination_min=("rumination_min", "mean") if "rumination_min" in fdf.columns else ("date_day", "count"),
        activity_rate=("activity_rate", "mean") if "activity_rate" in fdf.columns else ("date_day", "count"),
        eating_min=("eating_min", "mean") if "eating_min" in fdf.columns else ("date_day", "count"),
        standing_min=("standing_min", "mean") if "standing_min" in fdf.columns else ("date_day", "count"),
    )
    return agg.rename(columns={"date_day": "date"})


def compute_cow_ranking_table(df: pd.DataFrame, farm_id: str) -> pd.DataFrame:
    fdf = df[df["farm_id"].astype(str) == str(farm_id)].copy()
    if fdf.empty:
        return pd.DataFrame()

    rows = []
    req = [c for c in ["date", "rumination_min", "activity_rate", "milk_yield_l"] if c in fdf.columns]

    for cow_id, cdf in fdf.groupby(fdf["animal_id"].astype(str)):
        records = int(len(cdf))
        completeness = 0.0
        if req and records > 0:
            non_null = int(cdf[req].notna().sum().sum())
            completeness = 100.0 * non_null / float(records * len(req))

        avg_rum = _mean(cdf, "rumination_min")
        avg_act = _mean(cdf, "activity_rate")
        avg_eat = _mean(cdf, "eating_min")
        avg_stand = _mean(cdf, "standing_min")
        rating, score = assign_behavioral_rating(avg_rum, avg_act, avg_eat, avg_stand, completeness)

        rows.append(
            {
                "cow_id": cow_id,
                "avg_milk_yield_l": _mean(cdf, "milk_yield_l"),
                "records_loaded": records,
                "data_completeness_pct": round(completeness, 1),
                "behavioral_rating": rating,
                "behavioral_score": score,
                "avg_rumination_min": avg_rum,
                "avg_activity_rate": avg_act,
            }
        )

    return pd.DataFrame(rows).sort_values(["behavioral_score", "avg_milk_yield_l"], ascending=[False, False])


def build_farms_cows_to_review(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    req = [c for c in ["date", "rumination_min", "activity_rate", "milk_yield_l"] if c in df.columns]
    net_rum = _mean(df, "rumination_min")

    for cow_id, cdf in df.groupby(df["animal_id"].astype(str)):
        records = int(len(cdf))
        completeness = 0.0
        if req and records > 0:
            non_null = int(cdf[req].notna().sum().sum())
            completeness = 100.0 * non_null / float(records * len(req))

        cow_rum = _mean(cdf, "rumination_min")
        rum_gap = 0.0
        if net_rum is not None and net_rum != 0 and cow_rum is not None:
            rum_gap = ((cow_rum - net_rum) / net_rum) * 100.0

        review_score = (100.0 - completeness) * 0.55 + abs(rum_gap) * 0.45
        rows.append(
            {
                "farm_id": str(cdf["farm_id"].iloc[0]),
                "farm_name": str(cdf["farm_name"].iloc[0]) if "farm_name" in cdf.columns else str(cdf["farm_id"].iloc[0]),
                "cow_id": cow_id,
                "records_loaded": records,
                "data_completeness_pct": round(completeness, 1),
                "rumination_gap_pct": round(rum_gap, 2),
                "review_score": round(review_score, 2),
            }
        )

    return pd.DataFrame(rows).sort_values("review_score", ascending=False).head(top_n)


def compute_cow_profile(df: pd.DataFrame, cow_id: str) -> dict[str, float | str | None]:
    cdf = df[df["animal_id"].astype(str) == str(cow_id)].copy()
    if cdf.empty:
        return {}

    req = [c for c in ["date", "rumination_min", "activity_rate", "milk_yield_l"] if c in cdf.columns]
    records = int(len(cdf))
    non_null = int(cdf[req].notna().sum().sum()) if req else 0
    completeness = 100.0 * non_null / float(records * len(req)) if req and records > 0 else 0.0

    avg_rum = _mean(cdf, "rumination_min")
    avg_act = _mean(cdf, "activity_rate")
    avg_eat = _mean(cdf, "eating_min")
    avg_stand = _mean(cdf, "standing_min")
    rating, score = assign_behavioral_rating(avg_rum, avg_act, avg_eat, avg_stand, completeness)

    return {
        "cow_id": str(cow_id),
        "farm_id": str(cdf["farm_id"].iloc[0]),
        "farm_name": str(cdf["farm_name"].iloc[0]) if "farm_name" in cdf.columns else str(cdf["farm_id"].iloc[0]),
        "lpd": _mean(cdf, "milk_yield_l"),
        "avg_rumination": avg_rum,
        "avg_activity": avg_act,
        "avg_eating": avg_eat,
        "avg_standing": avg_stand,
        "data_completeness_pct": round(completeness, 1),
        "behavioral_rating": rating,
        "behavioral_score": score,
    }


def compute_cow_vs_context(df: pd.DataFrame, cow_id: str) -> pd.DataFrame:
    profile = compute_cow_profile(df, cow_id)
    if not profile:
        return pd.DataFrame()

    cdf = df[df["animal_id"].astype(str) == str(cow_id)]
    fdf = df[df["farm_id"].astype(str) == str(profile["farm_id"])]

    rows = []
    for metric in ["milk_yield_l", "rumination_min", "activity_rate", "eating_min", "standing_min"]:
        if metric not in df.columns:
            continue
        cow_avg = float(cdf[metric].mean())
        farm_avg = float(fdf[metric].mean())
        net_avg = float(df[metric].mean())

        def _pct(a: float, b: float) -> float:
            return ((a - b) / b * 100.0) if pd.notna(a) and pd.notna(b) and b != 0 else 0.0

        rows.append(
            {
                "metric": metric,
                "cow_average": round(cow_avg, 2),
                "farm_average": round(farm_avg, 2),
                "network_average": round(net_avg, 2),
                "vs_farm_pct": round(_pct(cow_avg, farm_avg), 2),
                "vs_network_pct": round(_pct(cow_avg, net_avg), 2),
            }
        )

    return pd.DataFrame(rows)


def compute_cow_daily_trend(df: pd.DataFrame, cow_id: str) -> pd.DataFrame:
    cdf = df[df["animal_id"].astype(str) == str(cow_id)].dropna(subset=["date"]).copy()
    if cdf.empty:
        return pd.DataFrame()

    cdf["date_day"] = cdf["date"].dt.floor("D")
    agg = cdf.groupby("date_day", as_index=False).agg(
        milk_yield_l=("milk_yield_l", "mean") if "milk_yield_l" in cdf.columns else ("date_day", "count"),
        rumination_min=("rumination_min", "mean") if "rumination_min" in cdf.columns else ("date_day", "count"),
        activity_rate=("activity_rate", "mean") if "activity_rate" in cdf.columns else ("date_day", "count"),
        eating_min=("eating_min", "mean") if "eating_min" in cdf.columns else ("date_day", "count"),
        standing_min=("standing_min", "mean") if "standing_min" in cdf.columns else ("date_day", "count"),
    )
    return agg.rename(columns={"date_day": "date"})


def build_cow_event_table(df: pd.DataFrame, cow_id: str) -> pd.DataFrame:
    cdf = df[df["animal_id"].astype(str) == str(cow_id)].copy()
    if cdf.empty:
        return pd.DataFrame()

    net_rum = _mean(df, "rumination_min")
    net_act = _mean(df, "activity_rate")

    if "rumination_min" in cdf.columns and net_rum is not None:
        cdf["low_rumination_flag"] = cdf["rumination_min"] < (0.85 * net_rum)
    if "activity_rate" in cdf.columns and net_act is not None:
        cdf["low_activity_flag"] = cdf["activity_rate"] < (0.85 * net_act)

    if "milk_yield_l" in cdf.columns:
        cow_milk = float(cdf["milk_yield_l"].mean()) if pd.notna(cdf["milk_yield_l"].mean()) else None
        if cow_milk is not None:
            cdf["milk_drop_flag"] = cdf["milk_yield_l"] < (0.85 * cow_milk)

    flags = [c for c in ["low_rumination_flag", "low_activity_flag", "milk_drop_flag"] if c in cdf.columns]
    if flags:
        cdf = cdf[cdf[flags].any(axis=1)]

    keep = [
        c
        for c in [
            "date",
            "farm_id",
            "animal_id",
            "milk_yield_l",
            "rumination_min",
            "activity_rate",
            "low_rumination_flag",
            "low_activity_flag",
            "milk_drop_flag",
            "data_collection_rate_pct",
        ]
        if c in cdf.columns
    ]
    return cdf[keep].sort_values("date", ascending=False).head(50)


def metric_drilldown(
    df: pd.DataFrame,
    metric: str,
    level: str,
    farm_id: str | None = None,
    cow_id: str | None = None,
) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    base = df.dropna(subset=["date"]).copy()
    if base.empty:
        return pd.DataFrame()
    base["date_day"] = base["date"].dt.floor("D")

    if level == "Network":
        out = base.groupby("date_day", as_index=False)[metric].mean()
        return out.rename(columns={"date_day": "date", metric: f"avg_{metric}"})

    if level == "Farm" and farm_id is not None:
        fdf = base[base["farm_id"].astype(str) == str(farm_id)]
        out = fdf.groupby("date_day", as_index=False)[metric].mean()
        return out.rename(columns={"date_day": "date", metric: f"avg_{metric}"})

    if level == "Cow" and cow_id is not None:
        cdf = base[base["animal_id"].astype(str) == str(cow_id)]
        out = cdf.groupby("date_day", as_index=False)[metric].mean()
        return out.rename(columns={"date_day": "date", metric: f"avg_{metric}"})

    return pd.DataFrame()
