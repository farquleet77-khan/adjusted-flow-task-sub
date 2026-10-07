"""Forecasting Service: loads the trained model once and returns quantile load forecasts.

ForecastingService loads models/load_model.joblib and models/intervals.joblib
 once, then predict() cleans and imputes the inputs, builds the features, applies
the point model (with its unseen-holiday rule), adds the condition-dependent intervals and
the December special cases. The Optimization Service only talks to this interface.

Interface
---------
    predict(inputs, quantiles=(0.1, 0.5, 0.9)) -> pd.DataFrame

    inputs:    DataFrame with a `timestamp` column (or DatetimeIndex) and the columns
               temperature_forecast_C, is_weekend, is_holiday, plant_state, forecast_available
               on a regular 15-minute grid.
    quantiles: any subset of (0.05, 0.1, 0.5, 0.9, 0.95). The median is always returned.

    returns:   one row per input timestamp with
               timestamp, load_kW (median), load_kW_lower, load_kW_upper
               (lowest / highest requested quantile), one column per requested quantile
               (q05, q10, q50, q90, q95), temp_imputed and adjustment notes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from forecasting.data import DECEMBER_COLUMNS, fill_december_temperature
from forecasting.features import make_features, with_day_type
from forecasting.intervals import interval_groups, qcol

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
# In the history, SHUTDOWN load never fell below the 28 kW standby level.
SHUTDOWN_FLOOR_KW = 28.0
from project_paths import MODELS as MODEL_DIR


class ForecastingService:
    """Holds the trained point model and interval model; both are loaded once in __init__."""

    def __init__(self, model_dir: Path | str = MODEL_DIR) -> None:
        model_dir = Path(model_dir)
        for name in ("load_model.joblib", "intervals.joblib"):
            if not (model_dir / name).exists():
                raise FileNotFoundError(f"{model_dir / name} not found: run scripts 03 and 04 first")
        bundle = joblib.load(model_dir / "load_model.joblib")
        iv = joblib.load(model_dir / "intervals.joblib")
        self.model = bundle["model"]
        self.model_name: str = bundle["model_name"]
        self.feature_set: str = bundle["feature_set"]
        self.intervals = iv["intervals"]
        self.imputed_widen: float = float(iv["imputed_widen_factor"])
        self.uses_temperature: bool = bool(iv["uses_temperature"])
        self.available_quantiles: tuple[float, ...] = tuple(self.intervals.quantiles)

    # ------------------------------------------------------------------ #
    def _validate(self, inputs: pd.DataFrame) -> pd.DataFrame:
        df = inputs.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("inputs need a 'timestamp' column or a DatetimeIndex")
        missing = sorted(set(DECEMBER_COLUMNS) - {"timestamp"} - set(df.columns))
        if missing:
            raise ValueError(f"inputs are missing columns: {missing}")
        if df.index.has_duplicates:
            raise ValueError("inputs contain duplicate timestamps")
        if not df.index.is_monotonic_increasing:
            raise ValueError("timestamps must be sorted")
        return df

    def _check_quantiles(self, quantiles: Sequence[float]) -> list[float]:
        qs = sorted(set(float(q) for q in quantiles) | {0.5})
        unsupported = [q for q in qs if not any(np.isclose(q, a) for a in self.available_quantiles)]
        if unsupported:
            raise ValueError(f"unsupported quantiles {unsupported}; available: {self.available_quantiles}")
        return qs

    def _adjust(self, q: pd.DataFrame, X: pd.DataFrame, filled: pd.DataFrame) -> pd.DataFrame:
        """December special cases, identical to scripts/04_uncertainty_task2.py step 7."""
        cols = [qcol(x) for x in self.available_quantiles]
        lower = [c for c in cols if c < "p50"]
        upper = [c for c in cols if c > "p50"]
        notes = pd.Series("", index=q.index)

        imputed = filled["temp_imputed"].to_numpy()
        if self.uses_temperature and self.imputed_widen > 1:
            for c in lower + upper:
                q.loc[imputed, c] = q.loc[imputed, "p50"] + self.imputed_widen * (q.loc[imputed, c] - q.loc[imputed, "p50"])
            notes[imputed] += "imputed temperature widened; "
        else:
            notes[imputed] += "imputed temperature (no effect on forecast); "

        unseen = self.model.unseen_holiday_mask(X)
        if unseen.any():
            variants = []
            for kind in ("weekend", "workday"):
                Xv = with_day_type(X[unseen], kind)
                pv = pd.Series(self.model.raw_predict(Xv), index=Xv.index)
                variants.append(self.intervals.predict(pv, interval_groups(Xv)))
            for c in lower:
                q.loc[unseen, c] = np.minimum.reduce([q.loc[unseen, c]] + [v[c] for v in variants])
            for c in upper:
                q.loc[unseen, c] = np.maximum.reduce([q.loc[unseen, c]] + [v[c] for v in variants])
            notes[unseen] += "unseen SHUTDOWN-holiday envelope; "

        # Days next to a plant-state change are flagged (not widened): errors there are larger.
        day_state = filled["plant_state"].groupby(filled.index.normalize()).agg(lambda s: s.mode().iat[0])
        changed = (day_state != day_state.shift(1)) | (day_state != day_state.shift(-1))
        changed.iloc[0] = day_state.iloc[0] != day_state.iloc[1] if len(day_state) > 1 else False
        changed.iloc[-1] = day_state.iloc[-1] != day_state.iloc[-2] if len(day_state) > 1 else False
        flagged_days = changed[changed].index
        transition = filled.index.normalize().isin(flagged_days) & ~unseen
        notes[transition] += "transition day (flag); "

        
        # Physical floor: no quantile of a SHUTDOWN interval may be below the standby load.
        shutdown = (filled["plant_state"] == "SHUTDOWN").to_numpy()
        for c in cols:
            q.loc[shutdown, c] = np.maximum(q.loc[shutdown, c], SHUTDOWN_FLOOR_KW)
        q[cols] = np.clip(np.sort(q[cols].to_numpy(), axis=1), 0, None)
        q["adjustment"] = notes.str.rstrip("; ")
        return q

    # ------------------------------------------------------------------ #
    def predict(self, inputs: pd.DataFrame, quantiles: Sequence[float] = DEFAULT_QUANTILES) -> pd.DataFrame:
        """Quantile load forecasts for every input timestamp (see module docstring)."""
        df = self._validate(inputs)
        qs = self._check_quantiles(quantiles)
        filled = fill_december_temperature(df)
        X = make_features(filled, "temperature_filled_C")

        point = pd.Series(self.model.predict(X), index=X.index)
        q = self._adjust(self.intervals.predict(point, interval_groups(X)), X, filled)

        out = pd.DataFrame(index=X.index)
        out["load_kW"] = q["p50"]
        lo, hi = qcol(qs[0]), qcol(qs[-1])
        out["load_kW_lower"], out["load_kW_upper"] = q[lo], q[hi]
        for x in qs:
            out[f"q{qcol(x)[1:]}"] = q[qcol(x)]
        out["temp_imputed"] = filled["temp_imputed"]
        out["adjustment"] = q["adjustment"]
        return out.rename_axis("timestamp").reset_index()


_SERVICE: ForecastingService | None = None


def get_service(model_dir: Path | str = MODEL_DIR) -> ForecastingService:
    """Return the shared service instance (the model is loaded only on the first call)."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = ForecastingService(model_dir)
    return _SERVICE


def predict(inputs: pd.DataFrame, quantiles: Sequence[float] = DEFAULT_QUANTILES) -> pd.DataFrame:
    """Module-level entry point used by the Optimization Service."""
    return get_service().predict(inputs, quantiles)
