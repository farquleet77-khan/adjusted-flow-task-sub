"""Tests for features, the unseen-holiday rule and the interval module."""

import numpy as np
import pandas as pd
import pytest

from forecasting.features import ALL_FEATURES, CORE, make_features, with_day_type
from forecasting.intervals import ResidualQuantileIntervals, interval_groups
from forecasting.models import fit_model


def _frame(days: int = 28, state: str = "NORMAL") -> pd.DataFrame:
    idx = pd.date_range("2025-03-03", periods=96 * days, freq="15min")
    return pd.DataFrame({"is_weekend": (idx.dayofweek >= 5).astype(int), "is_holiday": 0, "plant_state": state,
                         "temp": np.sin(np.arange(len(idx)) / 96 * 2 * np.pi)}, index=idx)


def test_make_features_complete() -> None:
    X = make_features(_frame(), "temp")
    assert list(X.columns) == ALL_FEATURES and X.notna().all().all() and X["qh"].between(0, 95).all()
    assert len(CORE) == 9


def test_unknown_plant_state_raises() -> None:
    with pytest.raises(ValueError):
        make_features(_frame(state="TURBO"), "temp")


def test_unseen_holiday_rule_stays_between_weekend_and_workday() -> None:
    normal, shutdown = _frame(state="NORMAL"), _frame(state="SHUTDOWN")
    shutdown.index = shutdown.index + pd.Timedelta(days=28)
    df = pd.concat([normal, shutdown])
    df.loc[df.index.normalize() == "2025-03-12", "is_holiday"] = 1          # holidays only seen in NORMAL
    load = np.where(df["plant_state"] == "NORMAL", 300.0, 50.0)
    load = np.where(df["is_weekend"] == 1, load / 2, load)
    load = np.where(df["is_holiday"] == 1, 200.0, load)
    X = make_features(df, "temp")
    model = fit_model("profile", X, pd.Series(load, index=X.index), CORE)
    test = df[df["plant_state"] == "SHUTDOWN"].iloc[:96].assign(is_holiday=1)   # SHUTDOWN on a holiday
    Xt = make_features(test, "temp")
    pred = model.predict(Xt)
    assert model.unseen_holiday_mask(Xt).all() and (pred >= 25 - 1e-9).all() and (pred <= 50 + 1e-9).all()


def test_with_day_type_removes_holiday() -> None:
    X = with_day_type(make_features(_frame().assign(is_holiday=1), "temp"), "weekend")
    assert (X["is_holiday"] == 0).all() and (X["dow"] == 5).all()


def test_intervals_are_ordered_and_non_negative() -> None:
    X = make_features(_frame(), "temp")
    groups = interval_groups(X)
    iv = ResidualQuantileIntervals(min_group_size=50).fit(groups, pd.Series(np.random.default_rng(0).normal(0, 10, len(X)), index=X.index))
    q = iv.predict(pd.Series(5.0, index=X.index), groups)[["p05", "p10", "p50", "p90", "p95"]].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all() and (q >= 0).all()
