"""Tests for the battery LP: every constraint holds, no export, no simultaneous flows, Pyomo = SciPy."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from optimization.battery import BatteryParams, ScheduleInputs, solve_rolling_daily, solve_schedule, summarize

PRM = BatteryParams()


def _inputs(days: int = 2, load: float = 200.0) -> ScheduleInputs:
    ts = pd.date_range("2025-12-01", periods=96 * days, freq="15min")
    hour = ts.hour + ts.minute / 60
    return ScheduleInputs(ts, 50 - 30 * np.cos(2 * np.pi * (hour - 12) / 24), np.full(len(ts), load))


def _check(res, inp: ScheduleInputs) -> None:
    s, tol = res.schedule, 1e-4
    assert s["soc_kWh"].between(PRM.minimum_soc_kWh - tol, PRM.maximum_soc_kWh + tol).all()
    assert s["charge_kW"].between(-tol, 100 + tol).all() and s["discharge_kW"].between(-tol, 100 + tol).all()
    assert s["grid_import_kW"].between(-tol, 600 + tol).all()                      # >= 0: no export
    np.testing.assert_allclose(s["grid_import_kW"], inp.load + s["charge_kW"] - s["discharge_kW"], atol=1e-4)
    prev = np.r_[PRM.initial_soc_kWh, s["soc_kWh"].to_numpy()[:-1]]
    np.testing.assert_allclose(s["soc_kWh"], prev + 0.25 * (0.975 * s["charge_kW"] - s["discharge_kW"] / 0.975), atol=1e-4)
    assert (s.loc[inp.day_end(), "soc_kWh"] >= 250 - tol).all()


def test_lp_constraints_and_no_simultaneous_flows() -> None:
    inp = _inputs()
    res = solve_schedule(inp, PRM, backend="scipy")
    _check(res, inp)
    kpis, _ = summarize(res, PRM)
    assert kpis["simultaneous_charge_discharge_intervals"] == 0 and kpis["net_saving_EUR"] > 0


def test_no_export_when_load_is_below_discharge_power() -> None:
    res = solve_schedule(_inputs(load=30.0), PRM, backend="scipy")          # like the shutdown
    assert (res.schedule["discharge_kW"] - res.schedule["charge_kW"] <= 30.0 + 1e-6).all()


def test_rolling_daily_meets_reserve() -> None:
    inp = _inputs(days=3)
    _check(solve_rolling_daily(inp, PRM, backend="scipy"), inp)


@pytest.mark.skipif(importlib.util.find_spec("pyomo") is None or importlib.util.find_spec("highspy") is None,
                    reason="Pyomo/highspy not installed")
def test_pyomo_matches_scipy() -> None:
    inp = _inputs()
    py, sp = solve_schedule(inp, PRM, backend="pyomo"), solve_schedule(inp, PRM, backend="scipy")
    _check(py, inp)
    assert abs(py.objective_EUR - sp.objective_EUR) < 1e-3
