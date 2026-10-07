"""05 - Task 3: battery scheduling for December (LP), deterministic vs conservative strategy.

 1. runs the Optimization Service: it CALLS the Forecasting Service for P05-P95 load, then solves the
    battery LP for strategy A (deterministic) and strategy B (conservative)
 2. formulation checks: LP never charges and discharges at once, grid export is zero, full month
    vs rolling daily, Pyomo vs SciPy (when Pyomo is installed)
 3. KPIs required by the task for both strategies, incl. days with forecast_available = 0
 4. why the battery charges and discharges when it does (price-driven daily pattern)
 5. the Christmas shutdown, where the strategies differ
 6. stress test: both fixed plans against 300 simulated Decembers (would they force grid export?)
 7. which constraints actually bind
 8. recommendation

Formulation (full equations in src/optimization/battery.py):
  min  sum_t g_t * 0.25 h * price_t / 1000  +  0.015 * sum_t (c_t + d_t) * 0.25 h
  s.t. SOC dynamics with 97.5 % efficiency each way, 100 <= SOC <= 450, 0 <= c, d <= 100 kW,
       0 <= g <= 600 kW, g = load + c - d (g >= 0 -> NO GRID EXPORT), SOC >= 250 kWh at 23:45 daily.



Strategies (the key insight: load only matters where a constraint can bind):
  A deterministic: P50 load everywhere.
  B conservative:  P50 for the expected cost, but discharge never exceeds the LOW load forecast (P10)
                   so the battery cannot push energy into the grid even if the actual load comes in low;
                   the grid limit is checked against the HIGH forecast (P90); +50 kWh SOC reserve on days
                   the Forecasting Service flags as uncertain (25-26 Dec envelope, 24 and 27 Dec).

Outputs: outputs/battery_schedule_<strategy>.csv, outputs/battery_daily_<strategy>.csv,
outputs/battery_kpis.json, tables and figures 05_*.png.

Run from anywhere (after scripts 03 and 04):  python scripts/05_battery_task3.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forecasting.data import day_type  # noqa: E402
from optimization.battery import BatteryParams, available_backend, solve_rolling_daily, solve_schedule, summarize  # noqa: E402
from optimization.service import OptimizationService  # noqa: E402
from plotting import plt, save  # noqa: E402
from project_paths import BATTERY_JSON, DEC_CSV, OUTPUTS, PRICE_CSV, TABLES, ensure_folders  # noqa: E402

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 20)
SEED = 42


def section(title: str) -> None:
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


ensure_folders()

# ----------------------------------------------------------------------------------------- #
section("1. Run the Optimization Service (it calls the Forecasting Service)")
backend = available_backend()
print("solver backend:", backend, "(pyomo = Pyomo + HiGHS; scipy = the same LP through SciPy's HiGHS, used only "
      "if Pyomo/highspy are not installed)")
params = BatteryParams.from_json(BATTERY_JSON)
print("battery parameters:", params)
inputs, prices = pd.read_csv(DEC_CSV), pd.read_csv(PRICE_CSV)
service = OptimizationService(params)
runs = service.run(inputs, prices, backend=backend)
A, B = runs["A_deterministic"], runs["B_conservative"]
fc = service.forecast.set_index("timestamp")
print(f"forecast received: {len(fc)} intervals with q05-q95 | strategies solved: {list(runs)}")

# ----------------------------------------------------------------------------------------- #
section("2. Formulation checks")
checks = []
for label, res in [("LP full month (strategy A)", A.result), ("LP rolling daily (strategy A)", solve_rolling_daily(A.inputs, params, backend))]:
    k, _ = summarize(res, params)
    checks.append({"run": label, "solver": res.solver, "seconds": round(res.seconds, 2), "total cost EUR": k["total_cost_EUR"],
                   "simultaneous charge+discharge intervals": k["simultaneous_charge_discharge_intervals"]})
if backend == "pyomo":   # verify the Pyomo model against the identical SciPy formulation
    res = solve_schedule(A.inputs, params, "scipy")
    k, _ = summarize(res, params)
    checks.append({"run": "SciPy cross-check (strategy A)", "solver": res.solver, "seconds": round(res.seconds, 2),
                   "total cost EUR": k["total_cost_EUR"], "simultaneous charge+discharge intervals": k["simultaneous_charge_discharge_intervals"]})
print(pd.DataFrame(checks).set_index("run").to_string())

for name, run in runs.items():
    s = run.result.schedule
    print(f"{name}: minimum grid import {s['grid_import_kW'].min():.3f} kW (negative would mean export) | "
          f"max discharge above planned load: {(s['discharge_kW'] - s['charge_kW'] - run.inputs.load_low).max():.3f} kW")
# expected: minimum grid import >= 0 and the second value <= 0 for both strategies: grid export never happens in the plan

# ----------------------------------------------------------------------------------------- #
section("3. KPIs for both strategies")
kpis = pd.DataFrame({n: r.kpis for n, r in runs.items()})
print(kpis.to_string())

kpis.to_csv(TABLES / "05_kpis.csv")
daily_summary = pd.DataFrame({n: r.daily["throughput_kWh"].describe()[["mean", "min", "max"]].round(1) for n, r in runs.items()})
print("\ndaily throughput [kWh]:\n" + daily_summary.to_string())   # expected: mean ~700 / ~690
fa_days = A.daily.index[A.daily["forecast_unavailable_rows"] > 0]
print("\ndays with forecast_available = 0:", [str(d) for d in fa_days])   # expected: 6, 14, 21 Dec
cols = ["throughput_kWh", "min_soc_kWh", "max_soc_kWh", "end_soc_kWh", "forecast_unavailable_rows"]
print(pd.concat({"A": A.daily.loc[fa_days, cols], "B": B.daily.loc[fa_days, cols]}).round(1).to_string())
print("-> The Forecasting Service imputes the missing temperatures. The selected load model does not use\n"
      "   temperature, so forecasts and schedules on these days are unaffected (identical in A and B). If a model\n"
      "   used temperature, the service would widen those intervals and strategy B would react to them.")

# ----------------------------------------------------------------------------------------- #
section("4. Why does the battery charge and discharge when it does?")
s = A.result.schedule.set_index("timestamp")
e_c, e_d = s["charge_kW"] * params.dt_h, s["discharge_kW"] * params.dt_h
print(f"energy-weighted price when charging:    {(e_c * s['price_EUR_per_MWh']).sum() / e_c.sum():.1f} EUR/MWh")   # expected: ~20
print(f"energy-weighted price when discharging: {(e_d * s['price_EUR_per_MWh']).sum() / e_d.sum():.1f} EUR/MWh")   # expected: ~80
day = s.loc["2025-12-10"]
action = (day["charge_kW"] > 0).astype(int) - (day["discharge_kW"] > 0).astype(int)
rid = action.ne(action.shift()).cumsum()
blocks = day.groupby(rid).agg(start=("price_EUR_per_MWh", lambda x: x.index.min().strftime("%H:%M")),
                              end=("price_EUR_per_MWh", lambda x: (x.index.max() + pd.Timedelta(minutes=15)).strftime("%H:%M")),
                              mean_price=("price_EUR_per_MWh", "mean"))
blocks["action"] = action.groupby(rid).first().map({1: "charge", -1: "discharge", 0: "idle"}).to_numpy()
print("\n10 December, strategy A:\n" + blocks.round(1).reset_index(drop=True).to_string())
print("-> Charge in the cheapest night quarter-hours and again 22:00-24:00 (the 250 kWh reserve at 23:45 forces the\n"
      "   evening recharge). Discharge only in the ~14 most expensive midday quarter-hours: 350 kWh usable energy at\n"
      "   25 kWh per interval. Every day cycles fully because the daily spread (60-94 EUR/MWh) beats the ~30 EUR/MWh\n"
      "   that losses and degradation cost.")

hour = s.index.hour + s.index.minute / 60
prof = pd.DataFrame({"price": s["price_EUR_per_MWh"].groupby(hour).mean(), "charge": s["charge_kW"].groupby(hour).mean(),
                     "discharge": s["discharge_kW"].groupby(hour).mean(), "soc": s["soc_kWh"].groupby(hour).mean()})
fig, ax = plt.subplots(1, 2, figsize=(16, 4.5))
ax[0].bar(prof.index, prof["charge"], width=0.25, color="tab:green", label="charge (avg kW)")
ax[0].bar(prof.index, -prof["discharge"], width=0.25, color="tab:red", label="discharge (avg kW)")
tw = ax[0].twinx()
tw.plot(prof.index, prof["price"], color="black", label="price")
tw.set_ylabel("EUR/MWh")
ax[0].set_xlabel("hour of day")
ax[0].set_ylabel("kW (discharge negative)")
ax[0].legend(loc="lower left")
ax[0].set_title("Average daily pattern (A) vs price")
ax[1].plot(prof.index, prof["soc"], color="tab:purple")
ax[1].axhline(250, color="gray", ls="--")
ax[1].set_xlabel("hour of day")
ax[1].set_ylabel("average SOC [kWh]")
ax[1].set_title("Average state of charge")
save(fig, "05_daily_pattern")

w = s.loc["2025-12-08":"2025-12-14"]
fig, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
ax[0].plot(w.index, w["price_EUR_per_MWh"], color="black")
ax[0].set_ylabel("EUR/MWh")
ax[0].set_title("One week (8-14 Dec), strategy A")
ax[1].plot(w.index, w["load_forecast_kW"], label="load forecast (P50)")
ax[1].plot(w.index, w["grid_import_kW"], label="grid import")
ax[1].set_ylabel("kW")
ax[1].legend()
ax[2].fill_between(w.index, w["charge_kW"], step="post", color="tab:green", label="charge")
ax[2].fill_between(w.index, -w["discharge_kW"], step="post", color="tab:red", label="discharge")
tw = ax[2].twinx()
tw.plot(w.index, w["soc_kWh"], color="tab:purple")
tw.set_ylabel("SOC [kWh]")
ax[2].set_ylabel("kW")
ax[2].legend(loc="upper left")
save(fig, "05_week_schedule")

# ----------------------------------------------------------------------------------------- #
section("5. The Christmas shutdown: where the strategies differ")
print(pd.DataFrame({n: r.daily.loc[pd.to_datetime(["2025-12-24", "2025-12-25", "2025-12-26", "2025-12-27"]).date,
                                   ["discharge_kWh", "min_soc_kWh"]].stack() for n, r in runs.items()}).unstack().round(1).to_string())
print("-> On 25-26 Dec the load (~35 kW) is below the 100 kW discharge power, so the no-export limit caps discharge:\n"
      "   A discharges up to the median load, B only up to the low (P10) load and keeps 150 kWh minimum SOC.")
z = slice("2025-12-24 06:00", "2025-12-27 18:00")
fig, ax = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
fz = fc.loc[z]
ax[0].fill_between(fz.index, fz["q10"], fz["q90"], alpha=0.3, label="load P10-P90")
ax[0].plot(fz.index, fz["q50"], lw=1, label="load P50")
for n, r, ls in [("A", A, "-"), ("B", B, "--")]:
    sz = r.result.schedule.set_index("timestamp").loc[z]
    ax[0].plot(sz.index, sz["discharge_kW"], ls=ls, label=f"discharge {n}")
    ax[1].plot(sz.index, sz["soc_kWh"], ls=ls, label=f"SOC {n}")
ax[0].set_ylim(0, 120)
ax[0].set_ylabel("kW")
ax[0].legend(ncol=5)
ax[0].set_title("Shutdown: discharge capped by the load (A: P50, B: P10)")
ax[1].set_ylabel("SOC [kWh]")
ax[1].legend()
save(fig, "05_shutdown_zoom")

# ----------------------------------------------------------------------------------------- #
section("6. Stress test: 300 simulated Decembers")

cvp = pd.read_csv(TABLES / "03_cv_predictions.csv", index_col="timestamp", parse_dates=True)
cvp["resid"], cvp["qh"] = cvp["y"] - cvp["pred"], cvp.index.hour * 4 + cvp.index.minute // 15
pools: dict = {}
for d, g in cvp.groupby(cvp.index.normalize()):
    if len(g) >= 90 and g["plant_state"].nunique() == 1:
        v = np.zeros(96)
        v[g["qh"].to_numpy()] = g["resid"].to_numpy()
        pools.setdefault((g["plant_state"].iat[0], g["day_type"].iat[0]), []).append(v)
dec_in = inputs.assign(timestamp=pd.to_datetime(inputs["timestamp"])).set_index("timestamp")
keys = pd.Series(list(zip(dec_in["plant_state"], day_type(dec_in))), index=dec_in.index).groupby(dec_in.index.normalize()).first()
rng = np.random.default_rng(SEED)
N = 300
scen = np.zeros((N, len(fc)))
for i, d in enumerate(fc.index.normalize().unique()):
    r = slice(96 * i, 96 * (i + 1))
    key = keys[d]
    if key in pools:
        base, pool = np.tile(fc["q50"].to_numpy()[r], (N, 1)), pools[key]
    else:
        u = rng.uniform(size=(N, 1))
        base = fc["q05"].to_numpy()[r] + u * (fc["q95"].to_numpy()[r] - fc["q05"].to_numpy()[r])
        pool = [v for k, vs in pools.items() if k[0] == key[0] for v in vs]
    scen[:, r] = np.clip(base + np.array(pool)[rng.integers(0, len(pool), N)], 0, None)
price = A.inputs.price
dt = params.dt_h


def stress(c: np.ndarray, d: np.ndarray) -> pd.DataFrame:
    grid = scen + c - d
    export = np.clip(-grid, 0, None)
    return pd.DataFrame({"cost": (np.clip(grid, 0, None) * dt * price / 1000).sum(1)
                         + params.degradation_cost_EUR_per_kWh * (c + d).sum() * dt,
                         "export_kWh": export.sum(1) * dt, "export_intervals": (export > 1e-6).sum(1),
                         "over_600": (grid > params.grid_import_limit_kW + 1e-6).sum(1)})


zero = np.zeros(len(fc))
outcomes = {"no battery": stress(zero, zero)}
for n, r in runs.items():
    sch = r.result.schedule
    outcomes[n] = stress(sch["charge_kW"].to_numpy(), sch["discharge_kW"].to_numpy())
stress_tbl = pd.DataFrame({n: {"mean cost EUR": o["cost"].mean(), "P90 cost EUR": o["cost"].quantile(0.9),
                               "mean saving vs no battery EUR": (outcomes["no battery"]["cost"] - o["cost"]).mean(),
                               "scenarios that would export %": 100 * (o["export_intervals"] > 0).mean(),
                               "mean energy that would be exported kWh": o["export_kWh"].mean(),
                               "scenarios above 600 kW %": 100 * (o["over_600"] > 0).mean()}
                           for n, o in outcomes.items()}).round(2)
print(stress_tbl.to_string())

stress_tbl.to_csv(TABLES / "05_stress_test.csv")
grid_a = scen + A.result.schedule["charge_kW"].to_numpy() - A.result.schedule["discharge_kW"].to_numpy()
by_day = pd.Series((np.clip(-grid_a, 0, None) * dt).mean(0), index=fc.index).groupby(fc.index.date).sum()
print("\nstrategy A, mean would-be export per day [kWh]:\n" + by_day[by_day > 0.01].round(2).to_string())   # expected: 25-26 Dec only

# ----------------------------------------------------------------------------------------- #
section("7. Which constraints bind?")
tol = 1e-3
bind = {}
for n, r in runs.items():
    sch, inp = r.result.schedule, r.inputs
    smin = inp.min_soc if inp.min_soc is not None else params.minimum_soc_kWh
    bind[n] = {"SOC at lower bound": int((sch["soc_kWh"] <= smin + tol).sum()),
               "SOC at 450 kWh": int((sch["soc_kWh"] >= params.maximum_soc_kWh - tol).sum()),
               "charging at 100 kW": int((sch["charge_kW"] >= params.maximum_charge_power_kW - tol).sum()),
               "discharging at 100 kW": int((sch["discharge_kW"] >= params.maximum_discharge_power_kW - tol).sum()),
               "no-export limit binding": int((sch["discharge_kW"] - sch["charge_kW"] >= inp.load_low - tol).sum()),
               "grid limit binding": int((inp.load_high + sch["charge_kW"] - sch["discharge_kW"] >= params.grid_import_limit_kW - tol).sum())}
print(pd.DataFrame(bind).to_string())


# ----------------------------------------------------------------------------------------- #
section("8. Save outputs and recommendation")
for n, r in runs.items():
    r.result.schedule.to_csv(OUTPUTS / f"battery_schedule_{n}.csv", index=False)
    r.daily.to_csv(OUTPUTS / f"battery_daily_{n}.csv")
(OUTPUTS / "battery_kpis.json").write_text(json.dumps({n: r.kpis for n, r in runs.items()}, indent=2))
(TABLES / "task3_summary.json").write_text(json.dumps({
    "backend": backend, "kpis": {n: r.kpis for n, r in runs.items()}, "checks": checks,
    "daily_throughput": daily_summary.to_dict(), "stress": stress_tbl.to_dict(),
    "charge_price": float((e_c * s["price_EUR_per_MWh"]).sum() / e_c.sum()),
    "discharge_price": float((e_d * s["price_EUR_per_MWh"]).sum() / e_d.sum()),
    "binding": bind, "forecast_unavailable_days": [str(d) for d in fa_days]}, indent=2, default=str))
print("saved outputs/battery_schedule_*.csv, outputs/battery_daily_*.csv, outputs/battery_kpis.json")
print("\nRECOMMENDATION: deploy strategy B, run as a rolling daily optimisation (prices arrive day-ahead).\n"
      "It costs only a few EUR more per month than A, but A's plan would push energy into the grid in many\n"
      "simulated Decembers (all on the shutdown days), and grid export is not allowed. Prices, not the load,\n"
      "drive the schedule; forecast uncertainty matters only where a constraint binds: the no-export limit at low load.")
