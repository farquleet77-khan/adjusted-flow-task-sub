r"""Battery scheduling as a linear program (LP), solved with HiGHS.


BatteryParams holds the constraints from config/battery.json; ScheduleInputs holds prices and
the load forecasts; solve_schedule() builds and solves the LP (Pyomo + HiGHS, with an identical
SciPy-HiGHS fallback); solve_rolling_daily() solves one day at a time; summarize() computes the
KPIs the task asks for.

Mathematical formulation
------------------------
Sets and data (t = 0..T-1, 15-minute intervals, Δt = 0.25 h):
    p_t        day-ahead price [EUR/MWh]
    L_t        load used in the energy balance and cost (P50 forecast) [kW]
    Llo_t      load used for the no-export safety check (= L_t in strategy A, P10 in B) [kW]
    Lhi_t      load used for the grid-limit safety check (= L_t in strategy A, P90 in B) [kW]
    Smin_t     lower SOC bound (100 kWh; raised in high-uncertainty periods in strategy B)
    D          last interval of every calendar day (the 23:45 interval)

Decision variables:
    c_t ∈ [0, 100]   charge power [kW]          d_t ∈ [0, 100]  discharge power [kW]
    s_t ∈ [Smin_t, 450]  SOC at the END of interval t [kWh]
    g_t ∈ [0, 600]   grid import [kW]

Objective (minimise electricity cost + degradation cost, EUR):
    min  Σ_t  g_t · Δt · p_t / 1000  +  0.015 · Σ_t (c_t + d_t) · Δt

Constraints:
    (1) battery dynamics   s_t = s_{t-1} + Δt·η_c·c_t − Δt·d_t/η_d,     s_{-1} = 250 kWh
    (2) SOC limits         Smin_t ≤ s_t ≤ 450
    (3) charge limit       0 ≤ c_t ≤ 100
    (4) discharge limit    0 ≤ d_t ≤ 100
    (5) grid import limit  0 ≤ g_t ≤ 600
    (6) energy balance     g_t = L_t + c_t − d_t,  g_t ≥ 0  -> GRID EXPORT IS IMPOSSIBLE
        safety versions    d_t − c_t ≤ Llo_t        (no export even if load comes in low)
                           Lhi_t + c_t − d_t ≤ 600  (grid limit even if load comes in high)
    (7) daily reserve      s_t ≥ 250 for t ∈ D (SOC carries over midnight, never reset)
    (8) no simultaneous charge and discharge

Why LP and not MILP for (8)
---------------------------
Constraint (8) would normally need a binary variable per interval (a MILP). It is not needed
here because simultaneous charging and discharging can never be optimal: every price is
positive (15-112 EUR/MWh), a round trip loses 5 % of the energy (0.975 x 0.975), and
degradation is charged on BOTH directions. Doing both at once only buys energy, wastes part
of it and pays degradation twice, so any such solution is beaten by reducing both flows. The LP is also faster, always solved to
proven optimality, and its sensitivities are easy to explain.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BatteryParams:
    capacity_kWh: float = 500.0
    initial_soc_kWh: float = 250.0
    minimum_soc_kWh: float = 100.0
    maximum_soc_kWh: float = 450.0
    maximum_charge_power_kW: float = 100.0
    maximum_discharge_power_kW: float = 100.0
    charge_efficiency: float = 0.975
    discharge_efficiency: float = 0.975
    grid_import_limit_kW: float = 600.0
    interval_minutes: int = 15
    degradation_cost_EUR_per_kWh: float = 0.015
    daily_reserve_kWh: float = 250.0

    @classmethod
    def from_json(cls, path: str | Path) -> "BatteryParams":
        return cls(**json.loads(Path(path).read_text()))

    @property
    def dt_h(self) -> float:
        return self.interval_minutes / 60


@dataclass
class ScheduleInputs:
    """Data for one optimisation run. All arrays have one value per interval."""

    timestamps: pd.DatetimeIndex
    price: np.ndarray
    load: np.ndarray
    load_low: np.ndarray | None = None
    load_high: np.ndarray | None = None
    min_soc: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        self.price = np.asarray(self.price, dtype=float)
        self.load = np.asarray(self.load, dtype=float)
        self.load_low = self.load.copy() if self.load_low is None else np.asarray(self.load_low, dtype=float)
        self.load_high = self.load.copy() if self.load_high is None else np.asarray(self.load_high, dtype=float)
        for name in ("price", "load", "load_low", "load_high"):
            arr = getattr(self, name)
            if len(arr) != n or np.isnan(arr).any():
                raise ValueError(f"{name} must have {n} values and no NaN")

    def day_end(self) -> np.ndarray:
        day = self.timestamps.normalize()
        return np.r_[day[1:] != day[:-1], True]

    def slice(self, mask: np.ndarray) -> "ScheduleInputs":
        return ScheduleInputs(self.timestamps[mask], self.price[mask], self.load[mask], self.load_low[mask],
                              self.load_high[mask], None if self.min_soc is None else self.min_soc[mask])


@dataclass
class ScheduleResult:
    schedule: pd.DataFrame
    objective_EUR: float
    solver: str
    seconds: float


# --------------------------------------------------------------------------- #
# Backend 1: Pyomo + HiGHS (primary)
# --------------------------------------------------------------------------- #
def _solve_pyomo(inp: ScheduleInputs, prm: BatteryParams, s0: float):
    import pyomo.environ as pyo

    T, dt = len(inp.timestamps), prm.dt_h
    smin = np.full(T, prm.minimum_soc_kWh) if inp.min_soc is None else inp.min_soc
    day_end = np.flatnonzero(inp.day_end())

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T - 1)
    m.c = pyo.Var(m.T, bounds=(0, prm.maximum_charge_power_kW))
    m.d = pyo.Var(m.T, bounds=(0, prm.maximum_discharge_power_kW))
    m.s = pyo.Var(m.T, bounds=lambda _, t: (float(smin[t]), prm.maximum_soc_kWh))
    m.g = pyo.Var(m.T, bounds=(0, prm.grid_import_limit_kW))

    m.dynamics = pyo.Constraint(m.T, rule=lambda m, t: m.s[t] == (s0 if t == 0 else m.s[t - 1])
                                + dt * prm.charge_efficiency * m.c[t] - dt * m.d[t] / prm.discharge_efficiency)
    m.balance = pyo.Constraint(m.T, rule=lambda m, t: m.g[t] == float(inp.load[t]) + m.c[t] - m.d[t])
    m.no_export_safe = pyo.Constraint(m.T, rule=lambda m, t: m.d[t] - m.c[t] <= float(inp.load_low[t]))
    m.grid_limit_safe = pyo.Constraint(
        m.T, rule=lambda m, t: float(inp.load_high[t]) + m.c[t] - m.d[t] <= prm.grid_import_limit_kW)
    m.reserve = pyo.Constraint([int(t) for t in day_end], rule=lambda m, t: m.s[t] >= prm.daily_reserve_kWh)

    m.cost = pyo.Objective(expr=sum(dt * float(inp.price[t]) / 1000 * m.g[t]
                                    + prm.degradation_cost_EUR_per_kWh * dt * (m.c[t] + m.d[t]) for t in m.T),
                           sense=pyo.minimize)

    for name in ("appsi_highs", "highs", "cbc", "glpk"):
        try:
            solver = pyo.SolverFactory(name)
            if solver.available(exception_flag=False):
                break
        except Exception:  # noqa: BLE001 - solver plugin missing or misconfigured
            continue
    else:
        raise RuntimeError("No open-source solver found: pip install highspy")
    result = solver.solve(m)
    if not pyo.check_optimal_termination(result):
        raise RuntimeError(f"Solver {name} did not find an optimal solution: {result.solver.termination_condition}")
    get = lambda var: np.array([pyo.value(var[t]) for t in m.T])  # noqa: E731
    return get(m.c), get(m.d), get(m.s), get(m.g), float(pyo.value(m.cost)), f"pyomo/{name}"


# --------------------------------------------------------------------------- #
# Backend 2: SciPy + HiGHS (same model in matrix form; used to verify the Pyomo model)
# --------------------------------------------------------------------------- #
def _solve_scipy(inp: ScheduleInputs, prm: BatteryParams, s0: float):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    T, dt = len(inp.timestamps), prm.dt_h
    nv = 4                                        # variables per interval: c, d, s, g
    C, D, S, G = 0, 1, 2, 3
    idx = lambda k, t: nv * t + k                 # noqa: E731
    smin = np.full(T, prm.minimum_soc_kWh) if inp.min_soc is None else inp.min_soc

    cost = np.zeros(nv * T)
    cost[[idx(C, t) for t in range(T)]] = prm.degradation_cost_EUR_per_kWh * dt
    cost[[idx(D, t) for t in range(T)]] = prm.degradation_cost_EUR_per_kWh * dt
    cost[[idx(G, t) for t in range(T)]] = dt * inp.price / 1000

    rows, cols, vals, b_eq = [], [], [], []
    for t in range(T):                            # (1) dynamics
        r = t
        rows += [r, r, r]; cols += [idx(S, t), idx(C, t), idx(D, t)]
        vals += [1.0, -dt * prm.charge_efficiency, dt / prm.discharge_efficiency]
        if t > 0:
            rows.append(r); cols.append(idx(S, t - 1)); vals.append(-1.0)
        b_eq.append(s0 if t == 0 else 0.0)
    for t in range(T):                            # (6) energy balance
        r = T + t
        rows += [r, r, r]; cols += [idx(G, t), idx(C, t), idx(D, t)]; vals += [1.0, -1.0, 1.0]
        b_eq.append(inp.load[t])
    A_eq = coo_matrix((vals, (rows, cols)), shape=(2 * T, nv * T)).tocsr()

    rows, cols, vals, b_ub = [], [], [], []
    r = 0
    for t in range(T):                            # safety: d - c <= Llo ; c - d <= 600 - Lhi
        rows += [r, r]; cols += [idx(D, t), idx(C, t)]; vals += [1.0, -1.0]; b_ub.append(inp.load_low[t]); r += 1
        rows += [r, r]; cols += [idx(C, t), idx(D, t)]; vals += [1.0, -1.0]
        b_ub.append(prm.grid_import_limit_kW - inp.load_high[t]); r += 1
    for t in np.flatnonzero(inp.day_end()):       # (7) daily reserve: -s_t <= -250
        rows.append(r); cols.append(idx(S, t)); vals.append(-1.0); b_ub.append(-prm.daily_reserve_kWh); r += 1
    A_ub = coo_matrix((vals, (rows, cols)), shape=(r, nv * T)).tocsr()

    lb, ub = np.zeros(nv * T), np.zeros(nv * T)
    for t in range(T):
        ub[idx(C, t)], ub[idx(D, t)] = prm.maximum_charge_power_kW, prm.maximum_discharge_power_kW
        lb[idx(S, t)], ub[idx(S, t)] = smin[t], prm.maximum_soc_kWh
        ub[idx(G, t)] = prm.grid_import_limit_kW

    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lb, ub)), method="highs")
    ok, x, fun, name = res.status == 0, res.x, res.fun, "scipy/highs-lp"
    if not ok:
        raise RuntimeError(f"{name} failed: {res.message}")
    x = x.reshape(T, nv)
    return x[:, C], x[:, D], x[:, S], x[:, G], float(fun), name


# --------------------------------------------------------------------------- #
def available_backend() -> str:
    """'pyomo' if Pyomo and HiGHS (highspy) are installed, otherwise the identical SciPy-HiGHS LP."""
    try:
        import highspy  # noqa: F401
        import pyomo.environ  # noqa: F401
        return "pyomo"
    except ImportError:
        return "scipy"


def solve_schedule(inp: ScheduleInputs, prm: BatteryParams, backend: str = "pyomo",
                   initial_soc: float | None = None) -> ScheduleResult:
    """Optimise the whole horizon in one LP."""
    s0 = prm.initial_soc_kWh if initial_soc is None else initial_soc
    t0 = time.time()
    solve = _solve_pyomo if backend == "pyomo" else _solve_scipy
    c, d, s, g, fun, solver = solve(inp, prm, s0)
    c, d, g = (np.where(np.abs(a) < 1e-7, 0.0, a) for a in (c, d, g))
    schedule = pd.DataFrame({
        "timestamp": inp.timestamps, "price_EUR_per_MWh": inp.price, "load_forecast_kW": inp.load,
        "charge_kW": c, "discharge_kW": d, "soc_kWh": s, "grid_import_kW": g,
    })
    schedule["electricity_cost_EUR"] = schedule["grid_import_kW"] * prm.dt_h * schedule["price_EUR_per_MWh"] / 1000
    return ScheduleResult(schedule, fun, solver, time.time() - t0)


def solve_rolling_daily(inp: ScheduleInputs, prm: BatteryParams, backend: str = "pyomo") -> ScheduleResult:
    """One 24-hour optimisation per day; the end-of-day SOC is the next day's starting SOC."""
    t0, parts, s0, total = time.time(), [], prm.initial_soc_kWh, 0.0
    days = inp.timestamps.normalize()
    for day in days.unique():
        res = solve_schedule(inp.slice(np.asarray(days == day)), prm, backend, initial_soc=s0)
        parts.append(res.schedule)
        s0, total = float(res.schedule["soc_kWh"].iloc[-1]), total + res.objective_EUR
    return ScheduleResult(pd.concat(parts, ignore_index=True), total, res.solver + " (rolling daily)", time.time() - t0)


def summarize(result: ScheduleResult, prm: BatteryParams,
              forecast_unavailable: pd.Series | None = None) -> tuple[dict, pd.DataFrame]:
    """KPIs required by Task 3 plus a per-day activity table."""
    s, dt, tol = result.schedule, prm.dt_h, 1e-3
    charge_kwh, discharge_kwh = s["charge_kW"] * dt, s["discharge_kW"] * dt
    day = s["timestamp"].dt.date
    daily = pd.DataFrame({
        "charge_kWh": charge_kwh.groupby(day).sum(),
        "discharge_kWh": discharge_kwh.groupby(day).sum(),
        "min_soc_kWh": s["soc_kWh"].groupby(day).min(),
        "max_soc_kWh": s["soc_kWh"].groupby(day).max(),
        "end_soc_kWh": s["soc_kWh"].groupby(day).last(),
    })
    daily["throughput_kWh"] = daily["charge_kWh"] + daily["discharge_kWh"]
    daily["hit_min_100"] = daily["min_soc_kWh"] <= prm.minimum_soc_kWh + tol
    daily["hit_max_450"] = daily["max_soc_kWh"] >= prm.maximum_soc_kWh - tol
    if forecast_unavailable is not None:
        daily["forecast_unavailable_rows"] = pd.Series(np.asarray(forecast_unavailable), index=s.index).groupby(day).sum()

    electricity = float(s["electricity_cost_EUR"].sum())
    no_battery = float((s["load_forecast_kW"] * dt * s["price_EUR_per_MWh"] / 1000).sum())
    throughput = float(charge_kwh.sum() + discharge_kwh.sum())
    degradation = prm.degradation_cost_EUR_per_kWh * throughput
    kpis = {
        "solver": result.solver,
        "solve_seconds": round(result.seconds, 2),
        "electricity_cost_EUR": round(electricity, 2),
        "degradation_cost_EUR": round(degradation, 2),
        "total_cost_EUR": round(electricity + degradation, 2),
        "cost_without_battery_EUR": round(no_battery, 2),
        "net_saving_EUR": round(no_battery - electricity - degradation, 2),
        "throughput_kWh": round(throughput, 1),
        "charged_kWh": round(float(charge_kwh.sum()), 1),
        "discharged_kWh": round(float(discharge_kwh.sum()), 1),
        "equivalent_full_cycles": round(float(discharge_kwh.sum()) / (prm.maximum_soc_kWh - prm.minimum_soc_kWh), 1),
        "days_with_throughput": int((daily["throughput_kWh"] > tol).sum()),
        "days_hit_min_soc": int(daily["hit_min_100"].sum()),
        "days_hit_max_soc": int(daily["hit_max_450"].sum()),
        "simultaneous_charge_discharge_intervals": int(((s["charge_kW"] > tol) & (s["discharge_kW"] > tol)).sum()),
        "min_end_of_day_soc_kWh": round(float(daily["end_soc_kWh"].min()), 2),
    }
    return kpis, daily
