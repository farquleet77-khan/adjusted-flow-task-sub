"""Feature engineering, shared by training, uncertainty estimation and the Forecasting Service.
A shortlisted features and then need to decide which features are to be choosed and which ones are redundant.
make_features() builds the candidate features for any time-indexed frame (history or
December). The training script decides which groups are actually used (what to remove).

Feature groups
--------------
CORE      qh (quarter-hour 0-95), dow (0 = Monday), is_weekend, is_holiday, plant state one-hot.
          The load is a production schedule: plant state x day type x quarter-hour explains
          ~95 % of its variance. Tree models learn the 06:00/14:00/22:00 shift steps directly
          from the integer qh, so no sin/cos is needed for them.
CYCLIC    hour, qh_sin/cos, dow_sin/cos. Same information as qh/dow; only tested, kept if the
          holdout MAE improves by more than 1 %.
WEATHER   1-hour smoothed temperature. Tested; the EDA found no effect once the schedule is known.
SEASON    month, ISO week. Tested ONLY to document why they are excluded: they re-encode the
          plant state, December's values (month 12, weeks 49-52) never occur in training, and
          29-31 Dec wrap to ISO week 1.

Never used: humidity and quality flags (not in the December inputs), lagged load (December has
no actuals; a recursive forecast would compound errors over 31 days).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PLANT_STATES = ["SHUTDOWN", "MAINTENANCE", "RAMP_UP", "NORMAL", "HIGH_DEMAND"]  # fixed order, all 5
STATE_COLUMNS = [f"state_{s}" for s in PLANT_STATES]
TEMP_SMOOTHING_INTERVALS = 4  # 1-hour centred mean: the December forecast is ~1.5x noisier than actuals

CORE = ["qh", "dow", "is_weekend", "is_holiday"] + STATE_COLUMNS
CYCLIC = ["hour", "qh_sin", "qh_cos", "dow_sin", "dow_cos"]
WEATHER = ["temp_1h"]
SEASON = ["month", "week_of_year"]

# the selection rule picks the FIRST set within 1 % of the best.
FEATURE_SETS: dict[str, list[str]] = {
    "core": CORE,
    "core + cyclic": CORE + CYCLIC,
    "core + temperature": CORE + WEATHER,
    "core + season": CORE + SEASON,
}
ALL_FEATURES = CORE + CYCLIC + WEATHER + SEASON


def make_features(df: pd.DataFrame, temp_col: str) -> pd.DataFrame:
    """All candidate features for a time-indexed frame.

    Args:
        df: needs is_weekend, is_holiday, plant_state and the temperature column `temp_col`.
        temp_col: 'temperature_clean_C' (history) or 'temperature_filled_C' (December).

    Raises:
        ValueError: unknown plant state (would otherwise give an all-zero one-hot row).
    """
    unknown = sorted(set(df["plant_state"]) - set(PLANT_STATES))
    if unknown:
        raise ValueError(f"Unknown plant_state values: {unknown}")

    idx = df.index
    X = pd.DataFrame(index=idx)
    X["qh"] = idx.hour * 4 + idx.minute // 15
    X["dow"] = idx.dayofweek
    X["is_weekend"] = df["is_weekend"].astype(int)
    X["is_holiday"] = df["is_holiday"].astype(int)
    for state, col in zip(PLANT_STATES, STATE_COLUMNS):
        X[col] = (df["plant_state"] == state).astype(int)

    X["hour"] = idx.hour
    X["qh_sin"] = np.sin(2 * np.pi * X["qh"] / 96)
    X["qh_cos"] = np.cos(2 * np.pi * X["qh"] / 96)
    X["dow_sin"] = np.sin(2 * np.pi * X["dow"] / 7)
    X["dow_cos"] = np.cos(2 * np.pi * X["dow"] / 7)

    # Computed on the FULL continuous series (before any row filtering) so that the rolling
    # window is not broken by excluded rows.
    X["temp_1h"] = df[temp_col].rolling(TEMP_SMOOTHING_INTERVALS, center=True, min_periods=1).mean()

    X["month"] = idx.month
    X["week_of_year"] = idx.isocalendar().week.astype(int).to_numpy()
    return X[ALL_FEATURES]


def plant_state_of(X: pd.DataFrame) -> pd.Series:
    """Plant state label recovered from the one-hot columns."""
    return X[STATE_COLUMNS].idxmax(axis=1).str.replace("state_", "", regex=False)


def day_type_of(X: pd.DataFrame) -> pd.Series:
    """workday / weekend / holiday from the feature frame (holiday wins)."""
    return pd.Series(np.select([X["is_holiday"] == 1, X["is_weekend"] == 1], ["holiday", "weekend"],
                               default="workday"), index=X.index)


def with_day_type(X: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Copy of X relabelled as an ordinary 'workday' or 'weekend' (holiday flag removed).

    Used for holidays in a plant state whose holiday behaviour was never observed
    (SHUTDOWN on 25-26 Dec): the forecast is built from the nearest seen combinations.
    """
    out = X.copy()
    out["is_holiday"] = 0
    if kind == "weekend":
        out["is_weekend"], out["dow"] = 1, 5
    elif kind == "workday":
        out["is_weekend"] = 0
        out["dow"] = out["dow"].where(out["dow"] < 5, 2)
    else:
        raise ValueError(f"kind must be 'workday' or 'weekend', got {kind!r}")
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    return out
