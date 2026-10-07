"""04 - Task 2: prediction intervals, their calibration, and when uncertainty is high or low.


 1. loads the out-of-sample (blocked-CV) predictions of the Task 1 model
 2. builds condition groups (plant state x day type x shift band)
 3. calibration check on history, cross-fitted (quantiles from 4 folds, scored on the 5th)
 4. strictest check: November and January holdouts
 5. when is uncertainty high or low? (state, day type, shift, time since a state change, temperature)
 6. effect of missing weather forecasts, measured by simulation
 7. saves the interval model and produces the December intervals THROUGH the Forecasting Service
 8. conditional table by plant state (average load and uncertainty)

Which percentiles, and why: P10/P90 is the delivered interval (load_kW_lower / load_kW_upper).
 * Strategy B in Task 3 uses P90, so the whole project uses one quantile set.
 * 10 % tails give twice as many tail events as 5 % tails, so per-condition estimates are stable
   (a condition group with 200 rows has 20 tail points per side).
 * An 80 % band is narrow enough to be useful.
P5/P95 are computed with the same method and saved too, for the safety checks in Task 3.

Method: interval = point forecast + empirical quantiles of OUT-OF-SAMPLE errors seen in the same
operating condition. It works with whichever model won Task 1, assumes no error distribution (the
errors are asymmetric), and gives wide intervals exactly where the model is less reliable.

Outputs: outputs/december_forecast.csv (timestamp, load_kW, load_kW_lower, load_kW_upper),
outputs/december_forecast_quantiles.csv (P5-P95 + flags), models/intervals.joblib, tables, figures 04_*.png.

Run from anywhere (after script 03):  python scripts/04_uncertainty_task2.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forecasting.data import clean_historical, fill_december_temperature, load_december_inputs, load_historical  # noqa: E402
from forecasting.features import PLANT_STATES, make_features  # noqa: E402
from forecasting.intervals import ResidualQuantileIntervals, interval_groups  # noqa: E402
from forecasting.service import ForecastingService  # noqa: E402
from plotting import STATE_COLORS, plt, save  # noqa: E402
from project_paths import DEC_CSV, HIST_CSV, MODELS, OUTPUTS, TABLES, ensure_folders  # noqa: E402

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 20)
SEED = 42
Q = ["p05", "p10", "p50", "p90", "p95"]


def section(title: str) -> None:
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


ensure_folders()

# ----------------------------------------------------------------------------------------- #
section("1. Inputs from Task 1")
for f in (MODELS / "load_model.joblib", TABLES / "03_cv_predictions.csv"):
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run scripts/03_forecast_task1.py first")
hist = clean_historical(load_historical(HIST_CSV))
dec_raw = load_december_inputs(DEC_CSV)
X_all = make_features(hist, "temperature_clean_C")
bundle = joblib.load(MODELS / "load_model.joblib")
cv = pd.read_csv(TABLES / "03_cv_predictions.csv", index_col="timestamp", parse_dates=True)
holdout = pd.read_csv(TABLES / "03_holdout_predictions.csv", index_col=["holdout", "timestamp"], parse_dates=["timestamp"])
print(f"point model: {bundle['model_name']} | features: {bundle['feature_set']} | CV rows: {len(cv):,}")

# ----------------------------------------------------------------------------------------- #
section("2. Out-of-sample residuals and condition groups")
# Residual = actual - predicted (positive = the model forecast too low). Groups: plant state x day type
# x shift band (night 22-06, early 06-14, late 14-22). A group with fewer than 200 rows falls back to
# state x day type, then state, then all rows, so no interval rests on a handful of points.
groups = interval_groups(X_all.loc[cv.index])
resid = cv["y"] - cv["pred"]
print("residual quantiles per plant state [kW]:")
print(resid.groupby(cv["plant_state"]).quantile([0.05, 0.1, 0.5, 0.9, 0.95]).unstack().reindex(PLANT_STATES).round(1).to_string())



# ----------------------------------------------------------------------------------------- #
section("3. Calibration on history (cross-fitted)")


def cross_fitted(quantiles=(0.05, 0.10, 0.50, 0.90, 0.95), min_group_size=200) -> pd.DataFrame:
    """Quantiles learned on 4 CV folds, applied to the 5th: an honest out-of-sample test."""
    parts = []
    for k in sorted(cv["fold"].unique()):
        tr, te = cv["fold"] != k, cv["fold"] == k
        iv = ResidualQuantileIntervals(quantiles, min_group_size).fit(groups[tr], resid[tr])
        parts.append(iv.predict(cv.loc[te, "pred"], groups[te]))
    return pd.concat(parts).sort_index().join(cv[["y", "plant_state", "day_type", "month", "hour"]])


def coverage(q: pd.DataFrame, by=None) -> pd.DataFrame:
    def summ(g):
        return pd.Series({
            "rows": len(g),
            "coverage P10-P90 %": 100 * ((g["y"] >= g["p10"]) & (g["y"] <= g["p90"])).mean(),
            "below P10 %": 100 * (g["y"] < g["p10"]).mean(),
            "above P90 %": 100 * (g["y"] > g["p90"]).mean(),
            "width P10-P90 kW": (g["p90"] - g["p10"]).mean(),
            "coverage P5-P95 %": 100 * ((g["y"] >= g["p05"]) & (g["y"] <= g["p95"])).mean(),
        })
    if by is None:
        return summ(q).to_frame("overall").T.round(1)
    return q.groupby(by)[["y"] + Q].apply(summ).round(1)


cal = cross_fitted()
cal_global = cross_fitted(min_group_size=10 ** 9)   # one width for every condition, for contrast
print(coverage(cal).to_string())

by_state = coverage(cal, "plant_state").reindex(PLANT_STATES)
by_state["coverage with ONE global width %"] = coverage(cal_global, "plant_state")["coverage P10-P90 %"]
print("\nby plant state:\n" + by_state.to_string())

print("\nby day type:\n" + coverage(cal, "day_type").to_string())

print("\nby shift:\n" + coverage(cal, groups.loc[cal.index, "shift"]).to_string())
by_state.to_csv(TABLES / "04_coverage_by_state.csv")

rel_q = (0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95)
cal_rel = cross_fitted(rel_q)
pairs = {50: ("p25", "p75"), 60: ("p20", "p80"), 70: ("p15", "p85"), 80: ("p10", "p90"), 90: ("p05", "p95")}
reliability = pd.Series({lvl: 100 * ((cal_rel["y"] >= cal_rel[a]) & (cal_rel["y"] <= cal_rel[b])).mean()
                         for lvl, (a, b) in pairs.items()}).round(1)
print("\nreliability (nominal -> observed %):", reliability.to_dict())   

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
ax[0].plot([45, 95], [45, 95], ls="--", color="gray", label="perfect")
ax[0].plot(reliability.index, reliability.values, marker="o", label="cross-fitted intervals")
ax[0].set_xlabel("nominal coverage [%]")
ax[0].set_ylabel("observed coverage [%]")
ax[0].legend()
ax[0].set_title("Reliability")
x = np.arange(len(PLANT_STATES))
ax[1].bar(x - 0.2, by_state["coverage P10-P90 %"], 0.4, label="per-condition widths")
ax[1].bar(x + 0.2, by_state["coverage with ONE global width %"], 0.4, label="one global width")
ax[1].axhline(80, color="k", ls="--", lw=1)
ax[1].set_xticks(x, PLANT_STATES, rotation=15)
ax[1].set_ylabel("coverage P10-P90 [%]")
ax[1].legend()
ax[1].set_title("Coverage by plant state (target 80 %)")
save(fig, "04_calibration")

# ----------------------------------------------------------------------------------------- #
section("4. Holdout check (models that never saw the test month, quantiles without that month)")
rows = []
for tag, month in [("nov", 11), ("jan", 1)]:
    hp = holdout.loc[tag]
    iv = ResidualQuantileIntervals().fit(groups[cv["month"] != month], resid[cv["month"] != month])
    qh = iv.predict(hp["pred"], interval_groups(X_all.loc[hp.index])).join(hp[["y", "plant_state"]])
    for subset, part in [("all rows", qh), ("NORMAL rows", qh[qh["plant_state"] == "NORMAL"])]:
        rows.append(coverage(part).assign(holdout=tag, subset=subset))
holdout_cov = pd.concat(rows).set_index(["holdout", "subset"])
print(holdout_cov.to_string())

holdout_cov.to_csv(TABLES / "04_holdout_coverage.csv")

# ----------------------------------------------------------------------------------------- #
section("5. When is uncertainty high or low?")
intervals = ResidualQuantileIntervals().fit(groups, resid)   # final interval model: all residuals
q_hist = intervals.predict(cv["pred"], groups)
width = (q_hist["p90"] - q_hist["p10"]).rename("width P10-P90 kW")
abs_err = resid.abs().rename("mean |error| kW")
change = hist["plant_state"].ne(hist["plant_state"].shift())
last_change = pd.Series(hist.index.where(change.to_numpy()), index=hist.index).ffill()
since = ((hist.index - last_change).dt.total_seconds() / 3600).reindex(cv.index)
since_bucket = pd.cut(since, [-1, 24, 72, np.inf], labels=["< 24 h", "24-72 h", "> 72 h"]).rename("since state change")
for by in [cv["plant_state"], cv["day_type"], groups["shift"], since_bucket]:
    tbl = pd.concat([width.groupby(by, observed=True).mean(), abs_err.groupby(by, observed=True).mean()], axis=1).round(1)
    print(tbl.to_string(), "\n")
print("coverage by time since the plant state changed:\n" + coverage(cal, since_bucket.reindex(cal.index)).to_string())

temp = X_all.loc[cv.index, "temp_1h"]
print("\ncorrelation of |error| with temperature per state:",
      {s: round(float(abs_err[cv["plant_state"] == s].corr(temp[cv["plant_state"] == s])), 3) for s in PLANT_STATES})
# expected: ~0 for NORMAL/SHUTDOWN/MAINTENANCE, ~0.1 for HIGH_DEMAND/RAMP_UP -> no practical temperature effect

fig, ax = plt.subplots(figsize=(12, 4.5))
for s in PLANT_STATES:
    sel = cv["plant_state"] == s
    ax.plot(width[sel].groupby(cv.loc[sel, "hour"]).mean(), marker=".", color=STATE_COLORS[s], label=s)
ax.set_xlabel("hour of day")
ax.set_ylabel("mean width P10-P90 [kW]")
ax.legend()
ax.set_title("Interval width by hour and plant state")
save(fig, "04_width_by_hour")

# ----------------------------------------------------------------------------------------- #
section("6. Data quality and forecast availability (simulation)")


uses_temperature = "temp_1h" in bundle["features"]
nov_model = joblib.load(MODELS / "holdout_nov.joblib")
nov = hist[hist.index.month == 11]


def nov_features(temperature: pd.Series, available: pd.Series) -> pd.DataFrame:
    df = nov[["is_weekend", "is_holiday", "plant_state"]].assign(temperature_forecast_C=temperature,
                                                                forecast_available=available)
    return make_features(fill_december_temperature(df), "temperature_filled_C")


avail = pd.Series(1, index=nov.index)
gap = pd.Series((nov.index.day % 4 == 2) & (nov.index.hour >= 8) & (nov.index.hour < 16), index=nov.index)
ok = nov["use_for_training"]
X_base, X_gap = nov_features(nov["temperature_clean_C"], avail), nov_features(nov["temperature_clean_C"], avail.where(~gap, 0))
sigma = np.sqrt(max(dec_raw["temperature_forecast_C"].diff().std() ** 2 - hist["temperature_clean_C"].diff().std() ** 2, 0) / 2)
X_noise = nov_features(nov["temperature_clean_C"] + np.random.default_rng(SEED).normal(0, sigma, len(nov)), avail)


def errs(Xs, rows):
    return np.abs(nov_model.predict(Xs[rows]) - nov.loc[rows, "load_kW"].to_numpy())


sim = pd.DataFrame({"MAE kW": [errs(X_base, gap & ok).mean(), errs(X_gap, gap & ok).mean(),
                               errs(X_base, ok).mean(), errs(X_noise, ok).mean()],
                    "P90 |error| kW": [np.quantile(errs(X_base, gap & ok), 0.9), np.quantile(errs(X_gap, gap & ok), 0.9),
                                       np.quantile(errs(X_base, ok), 0.9), np.quantile(errs(X_noise, ok), 0.9)]},
                   index=["gap rows, real temperature", "gap rows, imputed temperature", "all rows, real temperature",
                          f"all rows, noisy temperature (sigma {sigma:.2f} C)"]).round(2)
IMPUTED_WIDEN = max(1.0, float(sim.iloc[1, 1] / sim.iloc[0, 1]))
print(sim.to_string())
print(f"model uses temperature: {uses_temperature} | widening factor for imputed December rows: {IMPUTED_WIDEN:.3f}")

sim.to_csv(TABLES / "04_forecast_availability_simulation.csv")

# ----------------------------------------------------------------------------------------- #
section("7. December intervals, produced by the Forecasting Service")

joblib.dump({"intervals": intervals, "imputed_widen_factor": IMPUTED_WIDEN, "uses_temperature": uses_temperature,
             "delivered_interval": ("p10", "p90")}, MODELS / "intervals.joblib")
service = ForecastingService(MODELS)
fc = service.predict(pd.read_csv(DEC_CSV), quantiles=(0.05, 0.1, 0.5, 0.9, 0.95)).set_index("timestamp")
assert len(fc) == 2976 and fc[["q05", "q10", "q50", "q90", "q95"]].notna().all().all()
assert ((fc["q10"] <= fc["q50"]) & (fc["q50"] <= fc["q90"])).all()
print("adjustment flags:\n" + fc["adjustment"].replace("", "none").value_counts().to_string())

print("\naround the shutdown, daily means [kW]:")
print(fc.loc["2025-12-24":"2025-12-27", ["q05", "q10", "q50", "q90", "q95"]].resample("D").mean().round(1).to_string())

out = pd.DataFrame({"timestamp": fc.index, "load_kW": fc["q50"].to_numpy(), "load_kW_lower": fc["q10"].to_numpy(),
                    "load_kW_upper": fc["q90"].to_numpy()}).round(2)
out.to_csv(OUTPUTS / "december_forecast.csv", index=False)
fc.join(dec_raw["plant_state"]).round(2).to_csv(OUTPUTS / "december_forecast_quantiles.csv")
print(f"\nsaved outputs/december_forecast.csv ({len(out)} rows, P10-P90) and outputs/december_forecast_quantiles.csv")

# ----------------------------------------------------------------------------------------- #
section("8. Conditional forecasts by plant state (the Task 1 table, with calibrated uncertainty)")
plant = dec_raw["plant_state"]
table = pd.DataFrame({
    "Avg Load [kW]": fc["load_kW"].groupby(plant).mean(),
    "Uncertainty ± [kW] (P10-P90)": ((fc["q90"] - fc["q10"]) / 2).groupby(plant).mean(),
    "Uncertainty ± [kW] (P5-P95)": ((fc["q95"] - fc["q05"]) / 2).groupby(plant).mean(),
    "December intervals": plant.value_counts(),
}).reindex(PLANT_STATES).round(1)
table["Note"] = np.where(table["December intervals"].isna(), "not present in test period",
                         np.where(table.index == "SHUTDOWN", "holidays never seen: envelope interval", ""))
print(table.to_string())
# expected: NORMAL 265.1 ± 15.4 kW (± 19.8 at P5-P95); SHUTDOWN 34.6 ± 16.6 kW; HIGH_DEMAND, RAMP_UP, MAINTENANCE not present
table.to_csv(TABLES / "04_december_by_state.csv")

fig, axes = plt.subplots(2, 1, figsize=(15, 8))
for a, (start, end, title) in zip(axes, [("2025-12-01", "2025-12-31 23:45", "December forecast with intervals"),
                                         ("2025-12-22", "2025-12-29 23:45", "Zoom: around the Christmas shutdown")]):
    p = fc.loc[start:end]
    a.fill_between(p.index, p["q05"], p["q95"], alpha=0.2, color="tab:blue", label="P5-P95")
    a.fill_between(p.index, p["q10"], p["q90"], alpha=0.35, color="tab:blue", label="P10-P90")
    a.plot(p.index, p["q50"], lw=0.8, color="tab:blue", label="P50 (load_kW)")
    a.set_ylabel("load [kW]")
    a.set_title(title)
    a.legend(loc="upper right", ncol=3)
save(fig, "04_december_intervals")

summary = {
    "coverage_overall": coverage(cal).iloc[0].to_dict(),
    "reliability": reliability.to_dict(),
    "coverage_by_state": by_state.reset_index().to_dict(orient="records"),
    "holdout_coverage": holdout_cov.reset_index().to_dict(orient="records"),
    "width_by_state": width.groupby(cv["plant_state"]).mean().round(1).to_dict(),
    "imputed_widen_factor": IMPUTED_WIDEN, "uses_temperature": uses_temperature,
    "december_by_state": table.reset_index().to_dict(orient="records"),
    "shutdown_day": fc.loc["2025-12-25", ["q10", "q50", "q90"]].mean().round(1).to_dict(),
}
(TABLES / "task2_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("\nsaved reports/tables/task2_summary.json")
