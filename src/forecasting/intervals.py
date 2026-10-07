"""Prediction intervals from out-of-sample residual quantiles (split-conformal style).


Interval = point forecast + empirical quantile of out-of-sample residuals
(actual - predicted) for rows in similar conditions. Conditions are grouped as
plant state x day type x shift band; small groups fall back to coarser ones.

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.features import day_type_of, plant_state_of

QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
LEVELS = [["state", "day_type", "shift"], ["state", "day_type"], ["state"]]


def qcol(q: float) -> str:
    """Column name for a quantile, e.g. 0.1 -> 'p10'."""
    return f"p{round(q * 100):02d}"


def shift_band(qh: pd.Series) -> pd.Series:
    """Production shift from quarter-hour of day: night 22-06, early 06-14, late 14-22."""
    hour = qh // 4
    return pd.Series(np.select([(hour >= 6) & (hour < 14), (hour >= 14) & (hour < 22)],
                               ["early", "late"], default="night"), index=qh.index)


def interval_groups(X: pd.DataFrame) -> pd.DataFrame:
    """Condition labels used to look up residual quantiles."""
    return pd.DataFrame({
        "state": plant_state_of(X),
        "day_type": day_type_of(X),
        "shift": shift_band(X["qh"]),
    }, index=X.index)


class ResidualQuantileIntervals:
    """Condition-dependent prediction intervals from out-of-sample residuals."""

    def __init__(self, quantiles: tuple[float, ...] = QUANTILES, min_group_size: int = 200) -> None:
        self.quantiles = quantiles
        self.min_group_size = min_group_size

    def fit(self, groups: pd.DataFrame, residuals: pd.Series) -> "ResidualQuantileIntervals":
        df = groups.assign(residual=np.asarray(residuals))
        self.tables_ = []
        for keys in LEVELS:
            g = df.groupby(keys)["residual"]
            table = g.quantile(list(self.quantiles)).unstack()
            table.columns = [qcol(q) for q in self.quantiles]
            table = table[g.size() >= self.min_group_size]
            self.tables_.append((keys, table))
        overall = df["residual"].quantile(list(self.quantiles))
        self.global_ = pd.Series(overall.to_numpy(), index=[qcol(q) for q in self.quantiles])
        return self

    def offsets(self, groups: pd.DataFrame) -> pd.DataFrame:
        """Residual quantiles for each row, from the finest group with enough data."""
        cols = [qcol(q) for q in self.quantiles]
        out = pd.DataFrame(np.nan, index=groups.index, columns=cols)
        out["level"] = ""
        for keys, table in self.tables_:
            todo = out[cols[0]].isna()
            if not todo.any():
                break
            idx = pd.MultiIndex.from_frame(groups.loc[todo, keys]) if len(keys) > 1 else groups.loc[todo, keys[0]]
            found = table.reindex(idx).set_axis(groups.index[todo])
            hit = found[cols[0]].notna()
            out.loc[found.index[hit], cols] = found.loc[hit, cols].to_numpy()
            out.loc[found.index[hit], "level"] = " x ".join(keys)
        todo = out[cols[0]].isna()
        out.loc[todo, cols] = self.global_[cols].to_numpy()
        out.loc[todo, "level"] = "global"
        return out

    def predict(self, point: pd.Series, groups: pd.DataFrame) -> pd.DataFrame:
        """Quantile forecasts: point forecast + residual quantiles, sorted and clipped at zero."""
        off = self.offsets(groups)
        cols = [qcol(q) for q in self.quantiles]
        values = np.asarray(point)[:, None] + off[cols].to_numpy()
        values = np.clip(np.sort(values, axis=1), 0, None)
        out = pd.DataFrame(values, index=groups.index, columns=cols)
        out["level"] = off["level"]
        return out
