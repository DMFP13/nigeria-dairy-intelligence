from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BOOTSTRAP_SOURCE_LABEL = "bootstrap/demo"
BOOTSTRAP_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "bootstrap" / "sensor_bootstrap.csv"

FARM_PROFILES = [
    {
        "farm_id": "FARM-01",
        "farm_name": "Danone Anchor Farm",
        "farm_type": "Anchor",
        "milk_shift": 1.0,
        "rumination_shift": 8.0,
        "activity_shift": 2.5,
        "eating_shift": 10.0,
        "standing_shift": -10.0,
        "stability": 0.8,
        "data_quality": 96.0,
        "preg_rate": 0.60,
    },
    {
        "farm_id": "FARM-02",
        "farm_name": "High Output Commercial Farm",
        "farm_type": "Commercial High Output",
        "milk_shift": 2.2,
        "rumination_shift": 3.0,
        "activity_shift": 1.5,
        "eating_shift": 14.0,
        "standing_shift": -8.0,
        "stability": 1.0,
        "data_quality": 94.0,
        "preg_rate": 0.55,
    },
    {
        "farm_id": "FARM-03",
        "farm_name": "Emerging Supply Farm",
        "farm_type": "Emerging Supplier",
        "milk_shift": -0.8,
        "rumination_shift": -4.0,
        "activity_shift": -1.5,
        "eating_shift": -4.0,
        "standing_shift": 8.0,
        "stability": 1.25,
        "data_quality": 89.0,
        "preg_rate": 0.46,
    },
    {
        "farm_id": "FARM-04",
        "farm_name": "Reproduction Challenge Farm",
        "farm_type": "Repro Challenge",
        "milk_shift": -1.1,
        "rumination_shift": -9.0,
        "activity_shift": -3.0,
        "eating_shift": -8.0,
        "standing_shift": 18.0,
        "stability": 1.4,
        "data_quality": 90.0,
        "preg_rate": 0.30,
    },
    {
        "farm_id": "FARM-05",
        "farm_name": "Low Reliability Farm",
        "farm_type": "Low Reliability",
        "milk_shift": -1.4,
        "rumination_shift": -6.5,
        "activity_shift": -2.5,
        "eating_shift": -7.5,
        "standing_shift": 14.0,
        "stability": 1.65,
        "data_quality": 78.0,
        "preg_rate": 0.40,
    },
    {
        "farm_id": "FARM-06",
        "farm_name": "Benchmark Farm",
        "farm_type": "Benchmark",
        "milk_shift": 0.2,
        "rumination_shift": 5.0,
        "activity_shift": 1.0,
        "eating_shift": 6.0,
        "standing_shift": -4.0,
        "stability": 0.95,
        "data_quality": 97.0,
        "preg_rate": 0.64,
    },
]


def _cow_productivity_class(rng: np.random.Generator) -> tuple[str, float]:
    draw = float(rng.random())
    if draw < 0.22:
        return "high", 1.8
    if draw < 0.74:
        return "medium", 0.0
    return "low", -1.8


def generate_bootstrap_sensor_data(
    num_days: int = 105,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=num_days, freq="D")

    rows = []
    for profile in FARM_PROFILES:
        for cow_idx in range(20):
            cow_id = f"{profile['farm_id']}-COW-{cow_idx + 1:03d}"
            group_id = f"{profile['farm_id']}-PEN-{int(rng.integers(1, 5)):02d}"
            parity = int(rng.integers(1, 5))
            prod_class, class_milk_shift = _cow_productivity_class(rng)

            # Cow-level baselines shape long-run behavior and output.
            base_dim = int(rng.integers(8, 240))
            cow_rum_shift = rng.normal(0, 8)
            cow_act_shift = rng.normal(0, 3)
            cow_eat_shift = rng.normal(0, 10)
            cow_stand_shift = rng.normal(0, 10)
            cow_quality_shift = rng.normal(0, 2.5)

            # Event windows: mild stress and recovery for realism.
            stress_start = int(rng.integers(15, 75))
            stress_len = int(rng.integers(6, 14))
            stress_end = stress_start + stress_len

            for day_idx, dt in enumerate(dates):
                days_in_milk = min(base_dim + day_idx, 360)
                dim_curve = -0.015 * max(days_in_milk - 140, 0)
                lactation_phase = 1.2 if days_in_milk < 110 else (0.7 if days_in_milk < 220 else 0.0)
                weekly_wave = 1.0 * np.sin((2 * np.pi * day_idx) / 14)

                in_stress = stress_start <= day_idx <= stress_end
                stress_factor = -1.7 if in_stress else 0.0
                recovery_factor = 0.8 if (stress_end < day_idx <= stress_end + 10) else 0.0

                rumination_min = np.clip(
                    300
                    + profile["rumination_shift"]
                    + cow_rum_shift
                    + weekly_wave * 4
                    + stress_factor * 8
                    + recovery_factor * 4
                    + rng.normal(0, 14 * profile["stability"]),
                    120,
                    540,
                )

                activity_rate = np.clip(
                    56
                    + profile["activity_shift"]
                    + cow_act_shift
                    + weekly_wave * 1.5
                    + stress_factor * 2
                    + rng.normal(0, 5 * profile["stability"]),
                    18,
                    100,
                )

                eating_min = np.clip(
                    250
                    + profile["eating_shift"]
                    + cow_eat_shift
                    + weekly_wave * 5
                    + stress_factor * 5
                    + rng.normal(0, 18 * profile["stability"]),
                    90,
                    440,
                )

                standing_min = np.clip(
                    320
                    + profile["standing_shift"]
                    + cow_stand_shift
                    - weekly_wave * 4
                    - recovery_factor * 3
                    + rng.normal(0, 22 * profile["stability"]),
                    120,
                    640,
                )

                resting_min = np.clip(1440 - eating_min - standing_min + rng.normal(0, 18), 150, 900)

                # Milk is tied to behavior + lactation stage, not random.
                behavior_score = (
                    0.018 * rumination_min
                    + 0.012 * eating_min
                    + 0.006 * activity_rate
                    - 0.005 * standing_min
                )
                milk_yield_l = np.clip(
                    12.5
                    + behavior_score
                    + profile["milk_shift"]
                    + class_milk_shift
                    + parity * 0.55
                    + lactation_phase
                    + dim_curve
                    + stress_factor * 0.45
                    + rng.normal(0, 1.1 * profile["stability"]),
                    8,
                    36,
                )

                insemination_flag = 1 if (55 <= days_in_milk <= 180 and rng.random() < 0.009) else 0
                pregnancy_status = "pregnant" if (days_in_milk > 145 and rng.random() < profile["preg_rate"]) else "open"

                data_collection_rate = np.clip(
                    profile["data_quality"] + cow_quality_shift + rng.normal(0, 3.5 * profile["stability"]),
                    62,
                    100,
                )

                # Lower reliability farms intentionally have more missing values.
                if data_collection_rate < 84 and rng.random() < 0.30:
                    rumination_min = np.nan
                if data_collection_rate < 82 and rng.random() < 0.25:
                    activity_rate = np.nan
                if data_collection_rate < 78 and rng.random() < 0.20:
                    milk_yield_l = np.nan

                rows.append(
                    {
                        "farm_id": profile["farm_id"],
                        "farm_name": profile["farm_name"],
                        "farm_type": profile["farm_type"],
                        "cow_id": cow_id,
                        "animal_id": cow_id,
                        "date": dt.date().isoformat(),
                        "milk_yield_l": round(float(milk_yield_l), 2) if pd.notna(milk_yield_l) else np.nan,
                        "rumination_min": round(float(rumination_min), 2) if pd.notna(rumination_min) else np.nan,
                        "activity_rate": round(float(activity_rate), 2) if pd.notna(activity_rate) else np.nan,
                        "eating_min": round(float(eating_min), 2),
                        "standing_min": round(float(standing_min), 2),
                        "resting_min": round(float(resting_min), 2),
                        "data_collection_rate_pct": round(float(data_collection_rate), 2),
                        "group_id": group_id,
                        "parity": parity,
                        "days_in_milk": days_in_milk,
                        "insemination_flag": insemination_flag,
                        "pregnancy_status": pregnancy_status,
                        "data_source": BOOTSTRAP_SOURCE_LABEL,
                        "productivity_class_demo": prod_class,
                    }
                )

    return pd.DataFrame(rows)


def save_bootstrap_dataset(path: Path = BOOTSTRAP_DATA_PATH) -> Path:
    df = generate_bootstrap_sensor_data()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_bootstrap_dataset(path: Path = BOOTSTRAP_DATA_PATH, regenerate_if_missing: bool = True) -> pd.DataFrame:
    if not path.exists():
        if not regenerate_if_missing:
            raise FileNotFoundError(f"Bootstrap dataset not found at: {path}")
        save_bootstrap_dataset(path)

    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Ensure source traceability is explicit.
    if "data_source" not in df.columns:
        df["data_source"] = BOOTSTRAP_SOURCE_LABEL

    return df
