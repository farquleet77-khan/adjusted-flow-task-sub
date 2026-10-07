"""02 - Feature analysis: which candidate features carry information, and which are safe for December?

What this script does
---------------------
Screens every candidate feature from src/forecasting/features.py on the clean training rows
with four lenses, because the strongest effects in this data are NOT linear (the load jumps
at 06:00, 14:00, 22:00):
 1. variance explained (eta squared): share of load variance explained by grouping on a feature
 2. mutual information: any kind of dependency
 3. Pearson / Spearman correlation: straight-line / monotonic only (shown to prove they mislead)
 4. redundancy and confounding between features (correlation matrix, Cramer's V)
 5. December check: which feature values in December never occur in training?

The final feature set is NOT chosen here: script 03 tests the groups on holdout months and
picks the smallest group that is within 1 % of the best (the ablation rule). This script
explains why the candidates exist and what to expect.

Run from anywhere:  python scripts/02_feature_analysis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2_contingency  # noqa: E402
from sklearn.feature_selection import mutual_info_regression  # noqa: E402

from forecasting.data import clean_historical, fill_december_temperature, load_december_inputs, load_historical  # noqa: E402
from forecasting.features import ALL_FEATURES, day_type_of, make_features, plant_state_of  # noqa: E402
from plotting import plt, save  # noqa: E402
from project_paths import DEC_CSV, HIST_CSV, TABLES, ensure_folders  # noqa: E402

pd.set_option("display.width", 170)
SEED = 42


def section(title: str) -> None:
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


ensure_folders()
hist = clean_historical(load_historical(HIST_CSV))
dec = fill_december_temperature(load_december_inputs(DEC_CSV))
X_all = make_features(hist, "temperature_clean_C")        # full series first (rolling temperature)
X = X_all[hist["use_for_training"]]
y = hist.loc[hist["use_for_training"], "load_kW"]
X_dec = make_features(dec, "temperature_filled_C")
state, dtype = plant_state_of(X), day_type_of(X)
print(f"training rows: {len(X):,} | December rows: {len(X_dec):,} | candidate features: {len(ALL_FEATURES)}")


# ----------------------------------------------------------------------------------------- #
section("1. Variance explained (eta squared): how much of the load does a grouping explain?")


def eta_squared(groups: pd.Series, values: pd.Series) -> float:
    total = ((values - values.mean()) ** 2).sum()
    g = values.groupby(groups).agg(["mean", "size"])
    return float((g["size"] * (g["mean"] - values.mean()) ** 2).sum() / total)


groupings = {
    "plant_state x day type x quarter-hour": state + "|" + dtype + "|" + X["qh"].astype(str),
    "plant_state x day type": state + "|" + dtype,
    "day type (workday/weekend/holiday)": dtype,
    "day of week": X["dow"],
    "plant_state": state,
    "quarter-hour of day": X["qh"],
    "month": X["month"],
    "temperature (2 C bins)": X["temp_1h"] // 2,
}
eta = pd.Series({k: eta_squared(g, y) for k, g in groupings.items()}, name="eta_squared").round(3)
print(eta.to_string())

print("-> The schedule (state x day type x time of day) explains ~97 % of the load. Each part alone explains far\n"
      "   less: the effects are interactions, which is why tree models suit this problem.")

# ----------------------------------------------------------------------------------------- #
section("2-3. Mutual information vs correlation with the load")
sample = X.sample(8000, random_state=SEED)              
continuous = {"qh_sin", "qh_cos", "dow_sin", "dow_cos", "temp_1h"}
mi = mutual_info_regression(sample[ALL_FEATURES], y[sample.index],
                            discrete_features=[c not in continuous for c in ALL_FEATURES], random_state=SEED)
assoc = pd.DataFrame({"mutual_information": mi,
                      "pearson": X[ALL_FEATURES].corrwith(y).to_numpy(),
                      "spearman": X[ALL_FEATURES].corrwith(y, method="spearman").to_numpy()},
                     index=ALL_FEATURES).round(3).sort_values("mutual_information", ascending=False)
print(assoc.to_string())

assoc.to_csv(TABLES / "02_feature_association.csv")

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
assoc["mutual_information"].plot(kind="barh", ax=ax[0], color="tab:blue")
ax[0].invert_yaxis()
ax[0].set_title("Mutual information with load (any relationship)")
assoc["spearman"].abs().plot(kind="barh", ax=ax[1], color="tab:orange")
ax[1].invert_yaxis()
ax[1].set_title("|Spearman correlation| with load (monotonic only)")
save(fig, "02_feature_target_association")

# ----------------------------------------------------------------------------------------- #
section("4. Redundancy and confounding between features")
corr = X[ALL_FEATURES].corr(method="spearman")
pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
print("strongest feature-to-feature correlations:\n" + pairs[pairs.abs() >= 0.7].round(2).sort_values().to_string())



def cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b)
    return float(np.sqrt(chi2_contingency(table)[0] / (table.values.sum() * (min(table.shape) - 1))))


print(f"Cramer's V month vs plant_state: {cramers_v(X['month'], state):.2f}")   # expected: ~0.7
print("-> month and week re-encode which plant state the plant was in (and the season's temperature);\n"
      "   plant_state already provides that information directly and is known for December.")

fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(ALL_FEATURES)), ALL_FEATURES, rotation=90)
ax.set_yticks(range(len(ALL_FEATURES)), ALL_FEATURES)
fig.colorbar(im, ax=ax, label="Spearman correlation")
ax.set_title("Feature-to-feature correlation")
save(fig, "02_feature_correlation")

# ----------------------------------------------------------------------------------------- #
section("5. December check: feature values never seen in training")
for col in ["month", "week_of_year", "qh", "dow"]:
    unseen = ~X_dec[col].isin(set(X[col].unique()))
    print(f"{col:<13} December values {sorted(int(v) for v in X_dec[col].unique())[:6]}... "
          f"unseen in training: {100 * unseen.mean():.0f} %")

print("ISO week of 29-31 Dec:", sorted(int(v) for v in X_dec.loc["2025-12-29":, "week_of_year"].unique()),
      "-> wraps to week 1 (looks like January)")
print(f"smoothed temperature: December min {X_dec['temp_1h'].min():.1f} C vs training min {X['temp_1h'].min():.1f} C")

combos_train = set(zip(state, dtype))
combos_dec = pd.Series(list(zip(plant_state_of(X_dec), day_type_of(X_dec))))
print("December (state, day type) combinations never seen in training:", sorted(set(combos_dec) - combos_train),
      f"({(~combos_dec.isin(combos_train)).sum()} rows)")   

section("Summary")
print("Keep (core): qh, dow, is_weekend, is_holiday, plant-state one-hot.\n"
      "Test in script 03: cyclic (hour, sin/cos) and temperature, kept only if they beat core by > 1 % on the holdouts.\n"
      "Test only to document exclusion: month and week of year (confounded with plant state, unseen in December).\n"
      "Never: humidity and quality flags (absent from December), lagged load (no December actuals).")
