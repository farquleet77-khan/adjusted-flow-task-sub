"""Data loading, integrity checks and cleaning: one module used by training AND the service.

What this file does
-------------------
* load_historical / load_december_inputs / load_prices: read a CSV, check its columns,
  parse timestamps and index by time.
* check_integrity: row count, duplicates, gaps and irregular steps on the 15-minute grid.
* detect_isolated_spikes: one-interval glitches hidden inside rows labelled GOOD.
* label_mismatch_days: workdays whose load curve fits ANOTHER plant state far better than
  their own label (e.g. 1 Aug labelled RAMP_UP but running at full HIGH_DEMAND load).
* clean_historical: adds flag columns and `use_for_training`. Rows are never deleted and
  the load target is never changed or imputed.
* fill_december_temperature: imputes the December temperature forecast where the weather
  service was unavailable (for future ).

Cleaning policy (evidence in scripts/01_data_inspection.py)
------------------------------------------------------------
* quality_code != GOOD          -> excluded: weather inputs missing, wrong or unverifiable.
* temperature outside -40..50 C -> missing (the 88.0 C ).
* isolated load spikes          -> excluded: unpredictable label noise.
* label-mismatch workdays       -> excluded: the plant_state input is wrong on those days,
                                   so they would teach the model a wrong state/load link.
* floors (28 kW, 55 kW)         -> kept: genuine state-specific standby load.
"""

from __future__ import annotations

from os import PathLike

import numpy as np
import pandas as pd

INTERVAL = pd.Timedelta(minutes=15)
INTERVALS_PER_DAY = 96
TEMP_VALID_RANGE_C: tuple[float, float] = (-40.0, 50.0)
GOOD = "GOOD"

HISTORICAL_COLUMNS = ["timestamp", "load_kW", "temperature_actual_C", "relative_humidity_actual_pct",
                      "is_weekend", "is_holiday", "plant_state", "quality_code", "weather_station_online"]
DECEMBER_COLUMNS = ["timestamp", "temperature_forecast_C", "is_weekend", "is_holiday", "plant_state",
                    "forecast_available"]
PRICE_COLUMNS = ["timestamp", "price_EUR_per_MWh"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_csv(path: str | PathLike[str], required: list[str]) -> pd.DataFrame:
    """Read a CSV, fail early if a required column is missing, index by parsed timestamp."""
    df = pd.read_csv(path)
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    # Strict ISO format: a file re-saved by Excel (different date format) fails loudly here.
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    return df.sort_values("timestamp").set_index("timestamp")


def load_historical(path: str | PathLike[str]) -> pd.DataFrame:
    """historical_data.csv: Jan-Nov 2025, the only file with the load_kW target."""
    return _read_csv(path, HISTORICAL_COLUMNS)


def load_december_inputs(path: str | PathLike[str]) -> pd.DataFrame:
    """forecast_input_december.csv: features known in advance for December."""
    return _read_csv(path, DECEMBER_COLUMNS)


def load_prices(path: str | PathLike[str]) -> pd.DataFrame:
    """electricity_prices_december.csv: day-ahead price per 15-minute interval."""
    return _read_csv(path, PRICE_COLUMNS)


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #
def check_integrity(df: pd.DataFrame, name: str, expected_rows: int | None = None) -> dict[str, object]:
    """Structural summary of one time-indexed frame (no cross-file checks here)."""
    idx = df.index
    steps = idx.to_series().diff().dropna()
    grid = pd.date_range(idx.min(), idx.max(), freq=INTERVAL)
    per_day = df.groupby(idx.normalize()).size()
    return {
        "file": name,
        "rows": len(df),
        "expected_rows": expected_rows,
        "rows_ok": expected_rows is None or len(df) == expected_rows,
        "start": idx.min(),
        "end": idx.max(),
        "duplicate_timestamps": int(idx.duplicated().sum()),
        "missing_intervals": int(len(grid.difference(idx))),
        "irregular_steps": int((steps != INTERVAL).sum()),
        "days_without_96_rows": int((per_day != INTERVALS_PER_DAY).sum()),
        "nan_cells": int(df.isna().sum().sum()),
    }


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def day_type(df: pd.DataFrame) -> pd.Series:
    """workday / weekend / holiday per row (holiday wins over weekend)."""
    return pd.Series(np.select([df["is_holiday"] == 1, df["is_weekend"] == 1], ["holiday", "weekend"],
                               default="workday"), index=df.index)


def detect_isolated_spikes(load: pd.Series, min_jump_kw: float = 40.0, ratio: float = 6.0) -> pd.Series:
    """True where one value jumps away from BOTH neighbours while the neighbours agree.

    """
    prev, nxt = load.shift(1), load.shift(-1)
    deviation = (load - (prev + nxt) / 2).abs()
    neighbour_gap = (prev - nxt).abs()
    typical_step = load.diff().abs().rolling(INTERVALS_PER_DAY, center=True, min_periods=10).median()
    spike = (deviation > ratio * typical_step) & (neighbour_gap < deviation / 3) & (deviation > min_jump_kw)
    return spike.fillna(False).astype(bool)


def label_mismatch_days(df: pd.DataFrame, min_rmse_kw: float = 40.0, ratio: float = 0.5) -> pd.DataFrame:
    """Check every single-state WORKDAY: does its load curve fit another plant state far better?


    Returns one row per checked day with the verdict in column `mismatch`.
    """
    g = df[df["quality_code"] == GOOD].assign(day=lambda x: x.index.normalize(),
                                               qh=lambda x: x.index.hour * 4 + x.index.minute // 15)
    g = g[day_type(g) == "workday"]
    info = g.groupby("day").agg(state=("plant_state", "first"), states=("plant_state", "nunique"),
                                rows=("load_kW", "size"))
    info = info[(info["states"] == 1) & (info["rows"] >= 90)]
    curves = g.pivot_table(index="day", columns="qh", values="load_kW").loc[info.index]

    rows = []
    for day, state in info["state"].items():
        rmse = {}
        for other in info["state"].unique():
            peers = info.index[(info["state"] == other) & (info.index != day)]
            if len(peers) >= 2:
                profile = curves.loc[peers].median()
                rmse[other] = float(np.sqrt(np.nanmean((curves.loc[day] - profile) ** 2)))
        if state not in rmse:
            continue
        best = min(rmse, key=rmse.get)
        rows.append({"day": day, "label": state, "rmse_own_kW": rmse[state], "best_fit": best,
                     "rmse_best_kW": rmse[best], "mean_load_kW": float(curves.loc[day].mean())})
    out = pd.DataFrame(rows).set_index("day")
    out["mismatch"] = (out["best_fit"] != out["label"]) & (out["rmse_best_kW"] < ratio * out["rmse_own_kW"]) \
        & (out["rmse_own_kW"] > min_rmse_kw)
    return out


def clean_historical(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaning flags (rows are never dropped) and the final `use_for_training` flag."""
    out = df.copy()
    lo, hi = TEMP_VALID_RANGE_C
    temp = out["temperature_actual_C"]
    out["temp_invalid"] = temp.isna() | (temp < lo) | (temp > hi)
    out["temperature_clean_C"] = temp.where(~out["temp_invalid"])
    out["is_spike"] = detect_isolated_spikes(out["load_kW"])
    bad_days = label_mismatch_days(out)
    bad_days = bad_days.index[bad_days["mismatch"]]
    out["label_mismatch"] = out.index.normalize().isin(bad_days)
    out["use_for_training"] = ((out["quality_code"] == GOOD) & ~out["is_spike"] & ~out["temp_invalid"]
                               & ~out["label_mismatch"])
    return out


def fill_december_temperature(df: pd.DataFrame, short_gap_intervals: int = 8, neighbour_days: int = 3,
                              trust_unavailable: bool = False) -> pd.DataFrame:
    """Impute December temperature where forecast_available == 0 (or the value is NaN).

    * values flagged unavailable are treated as missing, even if a number is present;
    * gaps up to 2 h: time interpolation; longer gaps: mean of the same quarter-hour on the
      3 days before and after (a straight line over 8 h would miss the daily cycle);
    * adds `temperature_filled_C` (no NaN) and `temp_imputed`.
    """
    out = df.copy()
    temp = out["temperature_forecast_C"].astype(float)
    if not trust_unavailable:
        temp = temp.where(out["forecast_available"] == 1)
    missing = temp.isna()

    run_id = missing.ne(missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    interpolated = temp.interpolate(method="time", limit_area="inside")
    short = missing & (run_len <= short_gap_intervals) & interpolated.notna()
    filled = temp.copy()
    filled[short] = interpolated[short]

    window = pd.Timedelta(days=neighbour_days)
    for ts in filled.index[filled.isna()]:
        nearby = temp[(temp.index >= ts - window) & (temp.index <= ts + window)]
        same_slot = nearby[(nearby.index.hour == ts.hour) & (nearby.index.minute == ts.minute)].dropna()
        filled[ts] = same_slot.mean()
    filled = filled.fillna(temp.mean())  # last resort, only for pathological inputs

    out["temperature_filled_C"] = filled
    out["temp_imputed"] = missing
    return out
