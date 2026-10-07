"""01 - Data inspection: understand the data and turn every finding into a cleaning decision.

What this script does
---------------------
Reads the three CSV files and checks, step by step:
 1. columns and data types            9. repeated floor values (28 / 55 kW)
 2. timestamps and integrity         10. isolated spikes inside GOOD rows
 3. cross-file alignment             11. plant-state calendar and WRONG LABELS (RAMP_UP vs HIGH_DEMAND)
 4. calendar flags and categories    12. load behaviour: state, day type, time of day, temperature
 5. missing data                     13. December inputs
 6. quality codes, sensor outages    14. prices
 7. are flagged load values correct? 15. cleaning summary
 8. physical plausibility            16. decision log (saved to reports/tables/)


Run from anywhere:  python scripts/01_data_inspection.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forecasting.data import (check_integrity, clean_historical, day_type, detect_isolated_spikes,  # noqa: E402
                              label_mismatch_days, load_december_inputs, load_historical, load_prices)
from plotting import STATE_COLORS, STATE_ORDER, plt, save  # noqa: E402
from project_paths import DEC_CSV, HIST_CSV, PRICE_CSV, TABLES, ensure_folders  # noqa: E402

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 20)
DECISIONS: list[dict] = []


def section(title: str) -> None:
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


def decide(finding: str, rows: int, decision: str) -> None:
    DECISIONS.append({"finding": finding, "rows": int(rows), "decision": decision})


def runs(mask: pd.Series) -> pd.DataFrame:
    """Group consecutive True values into episodes with start, end and length."""
    run_id = mask.ne(mask.shift()).cumsum()
    g = mask[mask].groupby(run_id[mask])
    return pd.DataFrame({"start": g.apply(lambda s: s.index.min()), "end": g.apply(lambda s: s.index.max()),
                         "intervals": g.size()}).reset_index(drop=True)


ensure_folders()

# ----------------------------------------------------------------------------------------- #
section("1. Columns and data types (files read exactly as stored)")
raw = {"historical": pd.read_csv(HIST_CSV), "december_inputs": pd.read_csv(DEC_CSV), "prices": pd.read_csv(PRICE_CSV)}
for name, df in raw.items():
    print(f"\n{name}: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df.dtypes.to_string())
    print("first row:", df.iloc[0].to_dict())


hist = load_historical(HIST_CSV)
dec = load_december_inputs(DEC_CSV)
prices = load_prices(PRICE_CSV)

# ----------------------------------------------------------------------------------------- #
section("2. Timestamps and integrity")
integrity = pd.DataFrame([check_integrity(hist, "historical", 32_064), check_integrity(dec, "december_inputs", 2_976),
                          check_integrity(prices, "prices", 2_976)]).set_index("file")
print(integrity.T.to_string())

for day in ["2025-03-30", "2025-10-26"]:
    print(f"rows on {day} (European daylight-saving day): {len(hist.loc[day])}")  # expected: 96 and 96
print("-> 96 rows on both DST days: timestamps are naive (no daylight-saving shifts).")
decide("Files complete, continuous 15-min grid, no DST shifts", 0, "proceed; timestamps treated as naive")

# ----------------------------------------------------------------------------------------- #
section("3. Cross-file alignment")
print("December inputs and prices have identical timestamps:", dec.index.equals(prices.index))  # expected: True
print("History ends one interval before December:", hist.index.max() + pd.Timedelta(minutes=15) == dec.index.min())


# ----------------------------------------------------------------------------------------- #
section("4. Calendar flags and categories")
for name, df, flag in [("historical", hist, "weather_station_online"), ("december", dec, "forecast_available")]:
    real_weekend = (df.index.dayofweek >= 5).astype(int)
    print(f"{name}: is_weekend disagrees with the calendar on {(df['is_weekend'] != real_weekend).sum()} rows; "
          f"{flag} values {sorted(int(v) for v in df[flag].unique())}")  # expected: 0 rows; [0, 1]
holidays = pd.concat([hist["is_holiday"], dec["is_holiday"]])
print("holiday dates:", sorted({str(d) for d in holidays.index[holidays == 1].date}))
# expected: Jan 1, Apr 18, Apr 21, May 1, May 29, Jun 9, Oct 3, Dec 25, Dec 26 (nationwide German holidays only)
print("plant states (history):", sorted(hist["plant_state"].unique()))
print("plant states (December):", sorted(dec["plant_state"].unique()))  # expected: NORMAL, SHUTDOWN only
print("quality codes:", sorted(hist["quality_code"].unique()))

# ----------------------------------------------------------------------------------------- #
section("5. Missing data")
print("historical:\n" + hist.isna().sum()[lambda s: s > 0].to_string())   # expected: temperature 44, humidity 28
print("december:\n" + dec.isna().sum()[lambda s: s > 0].to_string())      # expected: temperature_forecast_C 56
print("-> load_kW and prices have no gaps; only weather columns are incomplete.")

fig, ax = plt.subplots(figsize=(14, 2.8))
for i, (label, ts) in enumerate([("history temperature", hist.index[hist["temperature_actual_C"].isna()]),
                                 ("history humidity", hist.index[hist["relative_humidity_actual_pct"].isna()]),
                                 ("December temperature", dec.index[dec["temperature_forecast_C"].isna()])]):
    ax.scatter(ts, [i] * len(ts), marker="|", s=200, color="tab:red")
ax.set_yticks([0, 1, 2], ["history temperature", "history humidity", "December temperature"])
ax.set_title("When are values missing? (each tick = one 15-minute row)")
save(fig, "01_missing_values")

# ----------------------------------------------------------------------------------------- #
section("6. Quality codes and weather-station outages")
qc = hist["quality_code"]
bad = qc != "GOOD"
quality = pd.DataFrame({
    "rows": qc.value_counts(),
    "temp_missing": hist["temperature_actual_C"].isna().groupby(qc).sum(),
    "humidity_missing": hist["relative_humidity_actual_pct"].isna().groupby(qc).sum(),
    "station_offline": (hist["weather_station_online"] == 0).groupby(qc).sum(),
    "mean_load_kW": hist.groupby("quality_code")["load_kW"].mean().round(1),
})
print(quality.to_string())

print(pd.crosstab(qc, hist["plant_state"]).to_string())
print("-> 138 of 172 MISSING_SENSOR rows are HIGH_DEMAND: that, not wrong values, explains the high mean.")

episodes = runs(bad)
print(f"\n{bad.sum()} non-GOOD rows in {len(episodes)} episodes; longest: {episodes['intervals'].max()} intervals")


offline = hist["weather_station_online"] == 0
off = runs(offline)
off["temp_missing"] = [int(hist.loc[s:e, "temperature_actual_C"].isna().sum()) for s, e in zip(off["start"], off["end"])]
off["temp_present"] = off["intervals"] - off["temp_missing"]
print("\nweather-station offline periods:\n" + off.to_string())

ms_online = (qc == "MISSING_SENSOR") & ~offline
print(f"MISSING_SENSOR rows with the station online: {ms_online.sum()} "
      f"(humidity missing in {hist.loc[ms_online, 'relative_humidity_actual_pct'].isna().sum()})")  # expected: 28 (8)
decide("Non-GOOD rows: weather data missing, wrong (88 C) or unverifiable (offline but values present)",
       bad.sum(), "exclude from training and validation; never impute the load")

# ----------------------------------------------------------------------------------------- #
section("7. Are the flagged load values correct?")
neighbour_median = hist["load_kW"].where(~bad).rolling(9, center=True, min_periods=3).median()  # GOOD rows, +-1 h
slot = pd.Series(hist.index.hour * 4 + hist.index.minute // 15, index=hist.index)
typical = hist["load_kW"].where(~bad).groupby([hist["plant_state"], day_type(hist), slot]).transform("median")
check = pd.DataFrame({
    "checkable_by_neighbours": (hist["load_kW"] / neighbour_median).notna().groupby(qc).sum(),
    "median_ratio_vs_neighbours": (hist["load_kW"] / neighbour_median).groupby(qc).median().round(3),
    "mean_load": hist.groupby("quality_code")["load_kW"].mean().round(1),
    "typical_GOOD_load_same_state_daytype_time": typical.groupby(qc).mean().round(1),
})
print(check.to_string())


longest = episodes.sort_values("intervals").iloc[-1]
w = hist.loc[longest["start"] - pd.Timedelta(hours=12): longest["end"] + pd.Timedelta(hours=12)]
flag = w["quality_code"] != "GOOD"
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(13, 5.5))
ax[0].plot(w.index, w["load_kW"], lw=1)
ax[0].scatter(w.index[flag], w.loc[flag, "load_kW"], color="tab:red", s=9, label="non-GOOD")
ax[0].set_ylabel("load [kW]")
ax[0].legend()
ax[0].set_title("Longest outage: load keeps its normal shift pattern; the weather station is offline")
ax[1].plot(w.index, w["temperature_actual_C"], lw=1)
ax[1].scatter(w.index[flag], w.loc[flag, "temperature_actual_C"], color="tab:red", s=9, label="station offline")
ax[1].set_ylabel("temperature [°C]")
ax[1].legend()
save(fig, "01_longest_outage")

# ----------------------------------------------------------------------------------------- #
section("8. Physical plausibility")
temp, hum = hist["temperature_actual_C"], hist["relative_humidity_actual_pct"]
print("negative load rows:", int((hist["load_kW"] < 0).sum()))                       
print("temperatures above 50 C:", temp[temp > 50].value_counts().to_dict())          
print(f"valid temperature range: {temp[temp <= 50].min():.1f} to {temp[temp <= 50].max():.1f} C")  
print("loads above the 600 kW grid limit:\n" + hist.loc[hist["load_kW"] > 600, ["load_kW", "plant_state"]].to_string())

jump = hum.diff().abs()
print(f"humidity: median 15-min change {jump.median():.1f} points, jumps > 20 points: {(jump > 20).sum()}, "
      f"exactly 100 %: {(hum == 100).sum()}") 
decide("Temperature sentinel 88.0 C", int((temp > 50).sum()), "treat temperatures outside -40..50 C as missing")
decide("Humidity noise-dominated and absent from December", int((jump > 20).sum()), "never use humidity")

# ----------------------------------------------------------------------------------------- #
section("9. Repeated floor values")
print(hist["load_kW"].value_counts().head(4).to_string())   # expected: 28.0 x 292, 55.0 x 233
floors = hist[hist["load_kW"].isin([28.0, 55.0])]
print(pd.crosstab([floors["load_kW"], floors["plant_state"]], day_type(floors).rename("day_type")).to_string())
# expected: 28 kW = SHUTDOWN weekends; 55 kW = RAMP_UP (+5 isolated values in NORMAL/HIGH_DEMAND = spikes)
decide("Floors 28 kW (SHUTDOWN weekends) and 55 kW (RAMP_UP)", len(floors), "keep: genuine standby load")

summer = hist.loc["2025-07-01":"2025-08-22"]
fig, ax = plt.subplots(figsize=(13, 4.5))
for state, grp in summer.groupby("plant_state"):
    ax.scatter(grp.index, grp["load_kW"], s=2, color=STATE_COLORS[state], label=state)
ax.axhline(28, ls="--", color="k", lw=0.8)
ax.axhline(55, ls=":", color="k", lw=0.8)
ax.set_ylabel("load [kW]")
ax.legend(markerscale=6)
ax.set_title("Summer shutdown and ramp-up: floors at 28 kW and 55 kW")
save(fig, "01_floors_summer")

# ----------------------------------------------------------------------------------------- #
section("10. Isolated spikes inside GOOD rows")
spikes = detect_isolated_spikes(hist["load_kW"])
print(f"isolated spikes: {spikes.sum()} (labelled GOOD: {(spikes & ~bad).sum()})")  
print(hist.loc[spikes, "plant_state"].value_counts().to_string())                
decide("Isolated one-interval spikes labelled GOOD", spikes.sum(), "exclude from training and metrics")
examples = hist.index[spikes.to_numpy()][[0, 3, 10]]
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
for a, ts in zip(axes, examples):
    seg = hist.loc[ts - pd.Timedelta(hours=3): ts + pd.Timedelta(hours=3), "load_kW"]
    a.plot(seg.index, seg.values, marker=".", lw=1)
    a.scatter([ts], [hist.loc[ts, "load_kW"]], color="tab:red", zorder=3)
    a.set_title(str(ts), fontsize=9)
    a.tick_params(axis="x", labelbottom=False)
save(fig, "01_spike_examples")

# ----------------------------------------------------------------------------------------- #
section("11. Plant-state calendar and wrong labels (is RAMP_UP labelled as HIGH_DEMAND?)")
state_run = hist["plant_state"].ne(hist["plant_state"].shift()).cumsum()
blocks = hist.groupby(state_run).agg(state=("plant_state", "first"), start=("plant_state", lambda s: s.index.min()),
                                     end=("plant_state", lambda s: s.index.max()))
long_blocks = blocks[(blocks["end"] - blocks["start"]) >= pd.Timedelta(days=2)]
print("plant-state blocks of 2+ days:\n" + long_blocks.to_string(index=False))

mismatch = label_mismatch_days(hist)
print(f"\nworkdays checked: {len(mismatch)}")                         # expected: 225
print("days whose load fits another state far better than their label:")
print(mismatch[mismatch["mismatch"]].round(1).to_string())

print("-> The ramp-up looks shifted by one day (2-15 Aug instead of 1-14 Aug), or 15 Aug is a regional\n"
      "   holiday (Assumption Day) missing from the nationwide holiday calendar. Either way the plant_state\n"
      "   input is wrong on those days, so they are excluded from training (2 x 96 rows).\n"
      "   December contains only NORMAL and SHUTDOWN, so this does not change the December inputs; it warns\n"
      "   that labels around transitions (24 and 27 Dec) are less reliable.")
decide("Load inconsistent with plant_state on 1 and 15 Aug (RAMP_UP <-> HIGH_DEMAND)",
       int(mismatch["mismatch"].sum()) * 96, "exclude those days from training; flag Dec transition days")

daily = hist.groupby(hist.index.normalize()).agg(load=("load_kW", "mean"), state=("plant_state", lambda s: s.mode().iat[0]))
zoom = daily.loc["2025-07-25":"2025-08-25"]
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(zoom.index, zoom["load"], color="lightgray", lw=1)
for state, grp in zoom.groupby("state"):
    ax.scatter(grp.index, grp["load"], s=45, color=STATE_COLORS[state], label=state, zorder=3)
for d in mismatch.index[mismatch["mismatch"]]:
    ax.annotate("label mismatch", (d, daily.loc[d, "load"]), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8, color="tab:red")
ax.set_ylabel("daily mean load [kW]")
ax.margins(y=0.15)
ax.legend(loc="lower left")
ax.set_title("Label check: 1 Aug (RAMP_UP) runs at full load, 15 Aug (HIGH_DEMAND) at ramp-up load")
save(fig, "01_label_check")

# ----------------------------------------------------------------------------------------- #
section("12. Load behaviour: what drives the load?")
good = hist[~bad].assign(day_type=lambda x: day_type(x))
by_type = good.pivot_table(index="plant_state", columns="day_type", values="load_kW", aggfunc="mean").reindex(STATE_ORDER)
print(by_type.round(1).to_string())

nw = good[(good["plant_state"] == "NORMAL") & (good["day_type"] == "workday")]
print("\nNORMAL workdays by month (is the state stable over time?):")
print(nw.groupby(nw.index.month).agg(mean_load=("load_kW", "mean"), mean_temp=("temperature_actual_C", "mean")).round(1).to_string())


g2 = good[good["temperature_actual_C"] <= 50].copy()
g2["slot"] = g2.index.hour * 4 + g2.index.minute // 15
resid = g2["load_kW"] - g2.groupby(["plant_state", "day_type", "slot"])["load_kW"].transform("mean")
corr = {s: resid[g2["plant_state"] == s].corr(g2.loc[g2["plant_state"] == s, "temperature_actual_C"]) for s in STATE_ORDER}
print("\ncorrelation of temperature with the load left after removing the schedule:", {k: round(float(v), 3) for k, v in corr.items()})


fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), sharey=True)
good["hour_float"] = good.index.hour + good.index.minute / 60
for a, state in zip(axes, STATE_ORDER):
    sub = good[good["plant_state"] == state]
    for dt, grp in sub.groupby("day_type"):
        prof = grp.groupby("hour_float")["load_kW"].mean()
        a.plot(prof.index, prof.values, label=dt)
    a.set_title(state)
    a.set_xticks([0, 6, 12, 18, 24])
    a.legend(fontsize=8)
axes[0].set_ylabel("mean load [kW]")
save(fig, "01_daily_profiles")

fig, ax = plt.subplots(figsize=(15, 4.5))
for state in STATE_ORDER:
    s = hist[hist["plant_state"] == state]
    ax.scatter(s.index, s["load_kW"], s=1, color=STATE_COLORS[state], label=state)
ax.set_ylabel("load [kW]")
ax.legend(markerscale=6, ncol=5)
ax.set_title("Load, every 15-minute row, January-November, by plant state")
save(fig, "01_full_year_load")

# ----------------------------------------------------------------------------------------- #
section("13. December inputs")
print(dec["plant_state"].value_counts().to_string())    
print("SHUTDOWN days:", sorted({str(d) for d in dec.index[dec["plant_state"] == "SHUTDOWN"].date}))  # expected: Dec 25, 26
fa0 = dec["forecast_available"] == 0
print(f"forecast_available = 0: {fa0.sum()} rows; NaN temperature: {dec['temperature_forecast_C'].isna().sum()}; "
      f"flagged but value present: {(fa0 & dec['temperature_forecast_C'].notna()).sum()}")  
print(runs(fa0).to_string())                              
print(f"temperature noise (std of 15-min change): history {hist['temperature_actual_C'].where(temp <= 50).diff().std():.2f} C, "
      f"December forecast {dec['temperature_forecast_C'].diff().std():.2f} C")  
print("-> SHUTDOWN on a holiday never occurs in the history: 25-26 Dec are an extrapolation.")
decide("64 December rows without a weather forecast (8 still carry a value)", int(fa0.sum()),
       "treat all as missing and impute; irrelevant if the model does not use temperature")
decide("SHUTDOWN on holidays (25-26 Dec) never seen in training", int(((dec['plant_state'] == 'SHUTDOWN') & (dec['is_holiday'] == 1)).sum()),
       "unseen-holiday rule in the model + envelope interval")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(dec.index, dec["temperature_forecast_C"], lw=0.7)
for _, r in runs(fa0).iterrows():
    ax.axvspan(r["start"], r["end"], color="tab:red", alpha=0.25)
ax.set_ylabel("temperature [°C]")
ax.set_title("December temperature forecast (red = weather service unavailable)")
save(fig, "01_december_temperature")

# ----------------------------------------------------------------------------------------- #
section("14. Prices")
price = prices["price_EUR_per_MWh"]
print(price.describe().round(1).to_string())             
daily_p = price.resample("D").agg(["min", "max"])
breakeven = (daily_p["min"] + 15) / 0.975 ** 2 + 15       
print(f"daily spread {(daily_p['max'] - daily_p['min']).min():.0f}-{(daily_p['max'] - daily_p['min']).max():.0f} EUR/MWh; "
      f"days where a full cycle pays off: {(daily_p['max'] > breakeven).sum()} of {len(daily_p)}")  # expected: 60-94, 31 of 31
hour_float = price.index.hour + price.index.minute / 60
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(price.groupby(hour_float).mean())
ax.set_xlabel("hour of day")
ax.set_ylabel("mean price [EUR/MWh]")
ax.set_xticks(range(0, 25, 3))
ax.set_title("Average December price profile: cheap at night, expensive around midday")
save(fig, "01_price_profile")

# ----------------------------------------------------------------------------------------- #
section("15. Cleaning summary (rules in src/forecasting/data.py)")
clean = clean_historical(hist)
summary = {
    "total rows": len(clean),
    "non-GOOD quality": int((clean["quality_code"] != "GOOD").sum()),
    "invalid temperature": int(clean["temp_invalid"].sum()),
    "isolated spikes": int(clean["is_spike"].sum()),
    "label-mismatch rows": int(clean["label_mismatch"].sum()),
    "used for training": int(clean["use_for_training"].sum()),
}
for k, v in summary.items():
    print(f"{k:<22} {v:>7,}")
# expected: 32064 / 210 / 50 / 59 / 192 / 31604 (98.57 %); rules overlap, so they do not simply add up
print(f"share kept: {100 * summary['used for training'] / summary['total rows']:.2f} %")
print("missing values left in training load/temperature:",
      int(clean.loc[clean['use_for_training'], ['load_kW', 'temperature_clean_C']].isna().sum().sum()))  # expected: 0

# ----------------------------------------------------------------------------------------- #
section("16. Decision log")
log = pd.DataFrame(DECISIONS)
print(log.to_string(index=False))
log.to_csv(TABLES / "01_decision_log.csv", index=False)
print("\nsaved reports/tables/01_decision_log.csv")
