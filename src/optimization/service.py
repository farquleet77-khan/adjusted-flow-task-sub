"""Optimization Service: gets forecasts from the Forecasting Service and schedules the battery.

Interface
---------
    OptimizationService(params, forecaster=None).run(inputs, prices, backend="pyomo")
        -> dict[str, StrategyRun]   (keys "A_deterministic", "B_conservative")

    inputs:  December forecast inputs (same format as forecasting.service.predict).
    prices:  DataFrame with timestamp and price_EUR_per_MWh for the same timestamps.

Strategies
----------
A, deterministic: P50 load everywhere.
B, conservative: P50 load for the expected cost, but
    * the no-export limit is enforced against the LOW forecast (P10): discharge never
      exceeds load even if the actual load comes in low (critical in the shutdown);
    * the grid limit is enforced against the HIGH forecast (P90);
    * the minimum SOC is raised by RESERVE_BUFFER_kWh in high-uncertainty periods
      (rows the Forecasting Service flags: unseen-holiday envelope, transition days,
      widened intervals for imputed temperature).

The command-line run lives in scripts/05_battery_task3.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting import service as forecasting_service
from optimization.battery import BatteryParams, ScheduleInputs, ScheduleResult, solve_schedule, summarize

RESERVE_BUFFER_kWh = 50.0
HIGH_UNCERTAINTY_NOTES = ("envelope", "transition", "widened")


@dataclass
class StrategyRun:
    name: str
    inputs: ScheduleInputs
    result: ScheduleResult
    kpis: dict
    daily: pd.DataFrame


class OptimizationService:
    def __init__(self, params: BatteryParams, forecaster=None) -> None:
        self.params = params
        self.forecaster = forecaster or forecasting_service.get_service()

    def strategy_inputs(self, forecast: pd.DataFrame, price: np.ndarray) -> dict[str, ScheduleInputs]:
        ts = pd.DatetimeIndex(forecast["timestamp"])
        high_uncertainty = forecast["adjustment"].fillna("").str.contains("|".join(HIGH_UNCERTAINTY_NOTES))
        min_soc_b = np.where(high_uncertainty, self.params.minimum_soc_kWh + RESERVE_BUFFER_kWh,
                             self.params.minimum_soc_kWh)
        return {
            "A_deterministic": ScheduleInputs(ts, price, forecast["q50"].to_numpy()),
            "B_conservative": ScheduleInputs(ts, price, forecast["q50"].to_numpy(),
                                             load_low=forecast["q10"].to_numpy(),
                                             load_high=forecast["q90"].to_numpy(), min_soc=min_soc_b),
        }

    def run(self, inputs: pd.DataFrame, prices: pd.DataFrame, backend: str = "pyomo") -> dict[str, StrategyRun]:
        forecast = self.forecaster.predict(inputs, quantiles=(0.05, 0.1, 0.5, 0.9, 0.95))
        prices = prices.assign(timestamp=pd.to_datetime(prices["timestamp"])).set_index("timestamp")
        price = prices["price_EUR_per_MWh"].reindex(pd.DatetimeIndex(forecast["timestamp"]))
        if price.isna().any():
            raise ValueError(f"{int(price.isna().sum())} forecast intervals have no price")
        unavailable = pd.Series(np.asarray(inputs["forecast_available"]) == 0)

        runs = {}
        for name, inp in self.strategy_inputs(forecast, price.to_numpy()).items():
            result = solve_schedule(inp, self.params, backend=backend)
            kpis, daily = summarize(result, self.params, unavailable)
            runs[name] = StrategyRun(name, inp, result, kpis, daily)
        self.forecast = forecast
        return runs
