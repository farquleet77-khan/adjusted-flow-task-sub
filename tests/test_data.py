"""Tests for src/forecasting/data.py: integrity checks, spike rule, label check, December imputation."""

import numpy as np
import pandas as pd

from forecasting.data import check_integrity, detect_isolated_spikes, fill_december_temperature, label_mismatch_days


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-12-01", periods=n, freq="15min")


def test_isolated_spike_is_flagged_but_step_change_is_not() -> None:
    values = np.full(200, 100.0) + np.random.default_rng(0).normal(0, 2, 200)
    values[50] = 400.0          # one-interval glitch
    values[120:] += 200.0       # genuine shift start
    flags = detect_isolated_spikes(pd.Series(values, index=_index(200)))
    assert flags.iloc[50] and not flags.iloc[119:122].any() and flags.sum() == 1


def test_check_integrity_detects_gap_and_duplicate() -> None:
    idx = _index(10).delete(4).append(_index(1)).sort_values()
    report = check_integrity(pd.DataFrame({"x": range(len(idx))}, index=idx), "t", 10)
    assert report["missing_intervals"] == 1 and report["duplicate_timestamps"] == 1


def test_label_mismatch_day_is_found() -> None:
    days = pd.date_range("2025-03-03", periods=30, freq="D")
    days = days[days.dayofweek < 5][:20]                       # 20 workdays
    idx = pd.DatetimeIndex(np.concatenate([pd.date_range(d, periods=96, freq="15min") for d in days]))
    state = np.repeat(["NORMAL"] * 10 + ["HIGH_DEMAND"] * 10, 96)
    load = np.where(state == "NORMAL", 300.0, 60.0) + np.random.default_rng(1).normal(0, 3, len(idx))
    load[96 * 12: 96 * 13] = 300.0                             # day 13 labelled HIGH_DEMAND but behaves NORMAL
    df = pd.DataFrame({"load_kW": load, "plant_state": state, "quality_code": "GOOD", "is_weekend": 0,
                       "is_holiday": 0}, index=idx)
    result = label_mismatch_days(df)
    assert list(result.index[result["mismatch"]]) == [days[12]]


def test_fill_december_temperature_leaves_no_gaps() -> None:
    n = 96 * 7
    temp = pd.Series(np.sin(np.arange(n) / 96 * 2 * np.pi), index=_index(n))
    available = pd.Series(1, index=temp.index)
    temp.iloc[10:14], available.iloc[10:14] = np.nan, 0
    available.iloc[300:340] = 0                                # values present but flagged unavailable
    out = fill_december_temperature(pd.DataFrame({"temperature_forecast_C": temp, "forecast_available": available}))
    assert out["temperature_filled_C"].notna().all() and out["temp_imputed"].sum() == 44
