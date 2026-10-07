"""03 - Task 1: train, validate and select the load forecast; forecast December.


 1. builds features on the clean training rows (rules from script 01)
 2. defines the validation splits (why each one exists is explained below)
 3. feature-group ablation: core vs + cyclic vs + temperature vs + season
 4. compares six models (profile baseline, Ridge, Random Forest, 2x XGBoost, LightGBM)
 5. reports metrics overall, by plant state, by month and by day type (blocked CV)
 6. forward-chaining by month: when do forecasts fail?
 7. error analysis: hour of day, time since a plant-state change, worst days
 8. permutation importance on unseen data
 9. trains the final model on ALL clean Jan-Nov rows and forecasts December

Outputs: outputs/december_point_forecast.csv, models/load_model.joblib (+ holdout models for
script 04), metric tables and task1_summary.json in reports/tables/, figures 03_*.png.
Runtime: a few minutes (about 80 model fits).

Run from anywhere:  python scripts/03_forecast_task1.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  
import numpy as np  
import pandas as pd  

from forecasting.data import clean_historical, fill_december_temperature, load_december_inputs, load_historical  # noqa: E402
from forecasting.features import FEATURE_SETS, PLANT_STATES, day_type_of, make_features, plant_state_of  # noqa: E402
from forecasting.models import MODEL_NAMES, fit_model  # noqa: E402
from plotting import STATE_COLORS, plt, save  # noqa: E402
from project_paths import DEC_CSV, HIST_CSV, MODELS, OUTPUTS, TABLES, ensure_folders  # noqa: E402

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 20)
SEED = 42


def section(title: str) -> None:
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


ensure_folders()
t_start = time.time()

# ----------------------------------------------------------------------------------------- #
section("1. Data and features")
hist = clean_historical(load_historical(HIST_CSV))
dec = fill_december_temperature(load_december_inputs(DEC_CSV))
X_all = make_features(hist, "temperature_clean_C")  # full continuous series first, then filter rows
mask = hist["use_for_training"]
X, y = X_all[mask], hist.loc[mask, "load_kW"]
X_dec = make_features(dec, "temperature_filled_C")
meta = pd.DataFrame({"plant_state": plant_state_of(X), "day_type": day_type_of(X), "month": X.index.month,
                     "hour": X.index.hour}, index=X.index)
print(f"training rows {len(X):,} | December rows {len(X_dec):,}")   

# ----------------------------------------------------------------------------------------- #
section("2. Validation splits")


months = X.index.month
HOLDOUTS = {"Nov": (months <= 10, months == 11), "Jan": (months >= 2, months == 1)}
week_id = ((X.index.normalize() - X.index.normalize().min()).days // 7).to_numpy()
cv_fold = week_id % 5
for name, (tr, te) in HOLDOUTS.items():
    print(f"holdout {name}: train {tr.sum():,} rows, test {te.sum():,} rows, "
          f"test states {meta.loc[te, 'plant_state'].value_counts().to_dict()}")
print("CV fold sizes:", pd.Series(cv_fold).value_counts().sort_index().to_dict())


# Metrics. MAE is primary: it is in kW and electricity cost is linear in load, so MAE maps straight
# to cost error. RMSE punishes large misses (they matter near battery limits). WAPE makes states with
# very different load levels comparable. Bias shows systematic over/under-forecasting (drift).
# MAPE is NOT used: shutdown load is as low as 28 kW, where small errors become huge percentages.
def metrics(y_true, y_pred) -> pd.Series:
    err = np.asarray(y_pred) - np.asarray(y_true)
    return pd.Series({"n": len(err), "MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()),
                      "WAPE_%": 100 * np.abs(err).sum() / np.abs(np.asarray(y_true)).sum(), "Bias": err.mean()})


def metric_table(res: pd.DataFrame, pred: str = "pred", by=None) -> pd.DataFrame:
    if by is None:
        return metrics(res["y"], res[pred]).to_frame("overall").T.round(2)
    return res.groupby(by)[["y", pred]].apply(lambda g: metrics(g["y"], g[pred])).round(2)


def holdout_run(model: str, features: list[str], split: str):
    tr, te = HOLDOUTS[split]
    m = fit_model(model, X[tr], y[tr], features)
    return m, meta[te].assign(y=y[te], pred=m.predict(X[te]))


def cv_predict(model: str, features: list[str]) -> pd.Series:
    pred = pd.Series(index=X.index, dtype=float)
    for k in range(5):
        m = fit_model(model, X[cv_fold != k], y[cv_fold != k], features)
        pred[cv_fold == k] = m.predict(X[cv_fold == k])
    return pred


# ----------------------------------------------------------------------------------------- #
section("3. Feature-group ablation (model: xgboost)")
# Rule: use the SIMPLEST group whose mean holdout MAE on NORMAL rows is within 1 % of the best.
# A feature that does not clearly help only adds risk (unseen values, confounding). The CV column is
# shown for contrast: in CV every month is in training, so month/week can LOOK harmless there; only
# a genuinely unseen month (the holdouts) reveals the problem.
rows = []
for set_name, cols in FEATURE_SETS.items():
    row = {"feature set": set_name, "n features": len(cols)}
    for split in HOLDOUTS:
        _, res = holdout_run("xgboost", cols, split)
        normal = res[res["plant_state"] == "NORMAL"]
        row[f"MAE NORMAL {split}"] = metrics(normal["y"], normal["pred"])["MAE"]
        row[f"Bias NORMAL {split}"] = metrics(normal["y"], normal["pred"])["Bias"]
    cvp = cv_predict("xgboost", cols)
    row["CV MAE NORMAL"] = (cvp - y)[meta["plant_state"] == "NORMAL"].abs().mean()
    rows.append(row)
ablation = pd.DataFrame(rows).set_index("feature set").round(2)
ablation["holdout MAE NORMAL (mean)"] = ablation[["MAE NORMAL Nov", "MAE NORMAL Jan"]].mean(axis=1).round(2)
best_score = ablation["holdout MAE NORMAL (mean)"].min()
FEATURE_SET = next(n for n in FEATURE_SETS if ablation.loc[n, "holdout MAE NORMAL (mean)"] <= 1.01 * best_score)
FEATURES = FEATURE_SETS[FEATURE_SET]
print(ablation.to_string())
print(f"\nchosen feature set: '{FEATURE_SET}' -> {FEATURES}")

ablation.to_csv(TABLES / "03_feature_ablation.csv")

# ----------------------------------------------------------------------------------------- #
section("4. Model comparison")

share = dec["plant_state"].value_counts(normalize=True)
W_NORMAL, W_SHUTDOWN = float(share.get("NORMAL", 0)), float(share.get("SHUTDOWN", 0))
comparison, holdout_preds, holdout_models, cv_preds = [], {}, {}, {}
for name in MODEL_NAMES:
    t0 = time.time()
    row = {"model": name}
    for split in HOLDOUTS:
        model, res = holdout_run(name, FEATURES, split)
        holdout_preds[(name, split)], holdout_models[(name, split)] = res, model
        normal = res[res["plant_state"] == "NORMAL"]
        row[f"MAE NORMAL {split}"] = metrics(normal["y"], normal["pred"])["MAE"]
    cv_preds[name] = cv_predict(name, FEATURES)
    err = (cv_preds[name] - y).abs()
    row.update({"CV MAE all": err.mean(), "CV MAE NORMAL": err[meta["plant_state"] == "NORMAL"].mean(),
                "CV MAE SHUTDOWN": err[meta["plant_state"] == "SHUTDOWN"].mean(), "seconds": time.time() - t0})
    comparison.append(row)
    print(f"  {name:<16} done in {row['seconds']:.0f} s")
comparison = pd.DataFrame(comparison).set_index("model")
comparison["December score"] = (W_NORMAL * comparison[["MAE NORMAL Nov", "MAE NORMAL Jan"]].mean(axis=1)
                                + W_SHUTDOWN * comparison["CV MAE SHUTDOWN"])
comparison = comparison.sort_values("December score").round(2)
BEST = comparison.index[0]
print(f"\nweights: NORMAL {W_NORMAL:.3f}, SHUTDOWN {W_SHUTDOWN:.3f}")   
print(comparison.to_string())
print(f"\nselected model: {BEST}")

comparison.to_csv(TABLES / "03_model_comparison.csv")

fig, ax = plt.subplots(figsize=(12, 4))
comparison[["MAE NORMAL Nov", "MAE NORMAL Jan", "CV MAE SHUTDOWN", "December score"]].plot(kind="bar", ax=ax, rot=0)
ax.set_ylabel("MAE [kW]")
ax.set_title("Model comparison: December-like holdouts (NORMAL), CV on SHUTDOWN, December score")
save(fig, "03_model_comparison")

for split in HOLDOUTS:   # Task 2 needs the holdout models and predictions of the selected model
    joblib.dump(holdout_models[(BEST, split)], MODELS / f"holdout_{split.lower()}.joblib")
pd.concat({s.lower(): holdout_preds[(BEST, s)] for s in HOLDOUTS}, names=["holdout"]).to_csv(
    TABLES / "03_holdout_predictions.csv")

week = slice("2025-11-10", "2025-11-16")
best_nov, base_nov = holdout_preds[(BEST, "Nov")], holdout_preds[("profile", "Nov")]
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(best_nov.loc[week].index, best_nov.loc[week, "y"], color="black", lw=1, label="actual")
ax.plot(best_nov.loc[week].index, best_nov.loc[week, "pred"], lw=1, label=BEST)
ax.plot(base_nov.loc[week].index, base_nov.loc[week, "pred"], lw=1, ls="--", label="profile baseline")
ax.set_ylabel("load [kW]")
ax.legend()
ax.set_title("One holdout week (10-16 Nov): actual vs forecasts")
save(fig, "03_holdout_week")

# ----------------------------------------------------------------------------------------- #
section(f"5. Metrics of the selected model ({BEST}) vs baseline, blocked CV")
cv = meta.assign(y=y, pred=cv_preds[BEST], baseline=cv_preds["profile"], fold=cv_fold)
cv.to_csv(TABLES / "03_cv_predictions.csv")


def compare(by=None) -> pd.DataFrame:
    m, b = metric_table(cv, "pred", by), metric_table(cv, "baseline", by)
    out = m.assign(MAE_baseline=b["MAE"])
    out["MAE improvement %"] = (100 * (1 - m["MAE"] / b["MAE"])).round(1)
    out["n"] = out["n"].astype(int)
    return out


overall, by_state, by_month, by_daytype = compare(), compare("plant_state").reindex(PLANT_STATES), compare("month"), compare("day_type")
print("overall:\n" + overall.to_string())
print("\nby plant state:\n" + by_state.to_string())

print("\nby month:\n" + by_month.to_string())
print("\nby day type:\n" + by_daytype.to_string())
for tbl, name in [(overall, "overall"), (by_state, "by_state"), (by_month, "by_month"), (by_daytype, "by_daytype")]:
    tbl.to_csv(TABLES / f"03_cv_metrics_{name}.csv")
print("-> Absolute errors are largest in high, volatile states (HIGH_DEMAND, RAMP_UP); relative errors are\n"
      "   largest at low load. Monthly errors follow the dominant plant state, not the season.\n"
      "-> HIGH_DEMAND is not identical in its two blocks: bias ~+11 kW in Apr-Jun and ~-13 kW in Sep-Oct (the\n"
      "   autumn block runs higher). Month features could capture this but break November (step 3); December\n"
      "   has no HIGH_DEMAND, so it does not affect the December forecast.")

# ----------------------------------------------------------------------------------------- #
section("6. Forward-chaining: when do forecasts fail?")
fc_rows = []
for m in range(2, 12):
    tr, te = months < m, months == m
    pred = fit_model(BEST, X[tr], y[tr], FEATURES).predict(X[te])
    new = sorted(set(meta.loc[te, "plant_state"]) - set(meta.loc[tr, "plant_state"]))
    fc_rows.append({"month": m, **metrics(y[te], pred).round(2).to_dict(), "new states": ", ".join(new) or "-"})
forward = pd.DataFrame(fc_rows).set_index("month")
print(forward.to_string())

forward.to_csv(TABLES / "03_forward_chaining.csv")
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(by_month.index, by_month["MAE"], marker="o", label="blocked CV (all states seen)")
ax.plot(forward.index, forward["MAE"], marker="s", label="forward-chaining (only earlier months)")
for mth, r in forward.iterrows():
    if r["new states"] != "-":
        ax.annotate("new: " + r["new states"], (mth, r["MAE"]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
ax.set_xticks(range(1, 12))
ax.set_xlabel("month")
ax.set_ylabel("MAE [kW]")
ax.legend()
ax.set_title("Forecasts fail when a plant state appears for the first time")
save(fig, "03_mae_by_month")

# ----------------------------------------------------------------------------------------- #
section("7. Error analysis")
cv["abs_err"] = (cv["pred"] - cv["y"]).abs()
change = hist["plant_state"].ne(hist["plant_state"].shift())
last_change = pd.Series(hist.index.where(change.to_numpy()), index=hist.index).ffill()
since = ((hist.index - last_change).dt.total_seconds() / 3600).reindex(cv.index)
bucket = pd.cut(since, [-1, 24, 72, np.inf], labels=["< 24 h", "24-72 h", "> 72 h"])
print("MAE by time since the plant state changed:")
print(cv.groupby(bucket, observed=True)["abs_err"].agg(["mean", "size"]).round(2).to_string())

worst = cv.groupby(cv.index.date).agg(MAE=("abs_err", "mean"), state=("plant_state", "first"),
                                      day_type=("day_type", "first")).sort_values("MAE", ascending=False)
print("\nworst days:\n" + worst.head(6).round(1).to_string())

fig, ax = plt.subplots(figsize=(11, 4))
for st in ["NORMAL", "HIGH_DEMAND", "SHUTDOWN"]:
    sel = cv["plant_state"] == st
    ax.plot(cv.loc[sel].groupby("hour")["abs_err"].mean(), marker=".", color=STATE_COLORS[st], label=st)
ax.set_xlabel("hour of day")
ax.set_ylabel("MAE [kW]")
ax.legend()
ax.set_title("Cross-validated error by hour: peaks at the shift changes (06, 14, 22)")
save(fig, "03_mae_by_hour")

# ----------------------------------------------------------------------------------------- #
section("8. Permutation importance (November holdout, unseen data)")

nov_model = holdout_models[(BEST, "Nov")]
X_nov, y_nov = X.loc[HOLDOUTS["Nov"][1], nov_model.columns()], y[HOLDOUTS["Nov"][1]]
rng = np.random.default_rng(SEED)
base_mae = np.abs(nov_model.estimator.predict(X_nov) - y_nov).mean()
importance = {}
for col in X_nov.columns:
    scores = []
    for _ in range(5):
        Xp = X_nov.copy()
        Xp[col] = rng.permutation(Xp[col].to_numpy())
        scores.append(np.abs(nov_model.estimator.predict(Xp) - y_nov).mean() - base_mae)
    importance[col] = float(np.mean(scores))
importance = pd.Series(importance, name="MAE increase when shuffled [kW]").sort_values(ascending=False).round(2)
print(importance.to_string())
print("-> November has only NORMAL and MAINTENANCE, so plant state ranks lower here than its true weight\n"
      "   (see the variance-explained table in script 02).")
importance.to_csv(TABLES / "03_permutation_importance.csv")
fig, ax = plt.subplots(figsize=(9, 5))
importance.plot(kind="barh", ax=ax)
ax.invert_yaxis()
ax.set_xlabel("MAE increase when shuffled [kW]")
ax.set_title(f"Permutation importance, November holdout ({BEST})")
save(fig, "03_permutation_importance")
print("\nplant_state included? YES: it is the planned schedule, known for every December interval, and the\n"
      "biggest single driver; without it a shutdown day cannot be told from a production day.")

# ----------------------------------------------------------------------------------------- #
section("9. Final model on all clean Jan-Nov rows, December forecast")
final = fit_model(BEST, X, y, FEATURES)
dec_point = pd.Series(final.predict(X_dec), index=X_dec.index, name="load_kW")
assert len(dec_point) == 2976 and dec_point.notna().all() and (dec_point >= 0).all()
print(f"model {BEST} | features '{FEATURE_SET}' | trees {final.info.get('n_estimators', '-')}")
print(f"unseen-holiday rule applied to {final.unseen_holiday_mask(X_dec).sum()} December rows, "
      f"holiday weight w = {final.holiday_weight:.3f}")

print("daily mean forecast around the shutdown [kW]:")
print(dec_point.loc["2025-12-23":"2025-12-28"].resample("D").mean().round(1).to_string())


joblib.dump({"model": final, "model_name": BEST, "feature_set": FEATURE_SET, "features": FEATURES,
             "trained_rows": int(len(X))}, MODELS / "load_model.joblib")
dec_point.rename_axis("timestamp").reset_index().to_csv(OUTPUTS / "december_point_forecast.csv", index=False)
print("saved models/load_model.joblib and outputs/december_point_forecast.csv")

state_avg = pd.DataFrame({"Avg Load [kW]": dec_point.groupby(dec["plant_state"]).mean(),
                          "December intervals": dec["plant_state"].value_counts()}).reindex(PLANT_STATES).round(1)
state_avg["Note"] = np.where(state_avg["December intervals"].isna(), "not present in test period", "")
print(state_avg.to_string())
print("(the uncertainty column of this table is produced by script 04)")

fig, ax = plt.subplots(figsize=(15, 4))
ax.plot(dec_point.index, dec_point, lw=0.7)
ax.fill_between(dec_point.index, 0, dec_point.max() * 1.05, where=(dec["plant_state"] == "SHUTDOWN").to_numpy(),
                color="gray", alpha=0.2, label="SHUTDOWN")
ax.set_ylabel("load [kW]")
ax.legend()
ax.set_title(f"December point forecast ({BEST})")
save(fig, "03_december_point_forecast")

summary = {
    "feature_set": FEATURE_SET, "features": FEATURES, "best_model": BEST,
    "holiday_weight": round(final.holiday_weight, 3), "training_rows": int(len(X)),
    "ablation": ablation.reset_index().to_dict(orient="records"),
    "comparison": comparison.reset_index().to_dict(orient="records"),
    "cv_by_state": by_state.reset_index().to_dict(orient="records"),
    "forward": forward.reset_index().to_dict(orient="records"),
    "transition_mae": cv.groupby(bucket, observed=True)["abs_err"].mean().round(2).to_dict(),
    "december_avg_by_state": state_avg["Avg Load [kW]"].dropna().round(1).to_dict(),
}
(TABLES / "task1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"\nsaved reports/tables/task1_summary.json | total runtime {time.time() - t_start:.0f} s")
