# Industrial Load Forecasting & Battery Optimisation

Forecast the December electricity load of an industrial site with calibrated uncertainty (Tasks 1–2), and schedule a 500 kWh battery to minimise electricity plus degradation cost without ever exporting to the grid (Task 3).

## File structure

```
adjusted-flow-task-sub/
├── README.md                     this file
├── requirements.txt              Python packages
├── pyproject.toml                pytest settings 
├── .gitignore
├── config/
│   └── battery.json              battery limits, efficiencies, degradation cost, daily reserve
├── data/
│   └── raw/                      PUT THE 3 CSV FILES HERE 
├── src/                          
│   ├── project_paths.py          all folders, resolved from the file location 
│   ├── plotting.py               figure to reports/figures/
│   ├── forecasting/              FORECASTING SERVICE
│   │   ├── data.py               loading, integrity checks, cleaning flags, label check, December imputation 
│   │   ├── features.py           candidate features (core / cyclic / temperature / season)
│   │   ├── models.py             baseline + ML models, early stopping, unseen-holiday rule
│   │   ├── intervals.py          prediction intervals from out-of-sample residuals
│   │   └── service.py            predict(): the interface used by the optimisation
│   └── optimization/             OPTIMIZATION SERVICE
│       ├── battery.py            the LP (Pyomo + HiGHS), equations in the docstring
│       └── service.py            calls predict(), builds strategies A and B, solves
├── scripts/                      run these,
│   ├── 01_data_inspection.py     data checks, issues, wrong-label check, cleaning decisions
│   ├── 02_feature_analysis.py    which features carry information, which are unsafe for December
│   ├── 03_forecast_task1.py      Task 1: validation, feature ablation, model selection, December forecast
│   ├── 04_uncertainty_task2.py   Task 2: intervals, calibration, uncertainty analysis, final CSV
│   ├── 05_battery_task3.py       Task 3: strategies A/B, checks, schedule analysis, stress test
│   ├── 06_write_reports.py       writes the two reports from the computed results
│   └── run_all.py                runs 01 → 06
├── tests/                        pytest unit tests
│   ├── test_data.py
│   ├── test_features_models.py
│   ├── test_optimization.py
│   └── test_service.py
│
│   created when you run the scripts:
├── models/                       load_model.joblib, intervals.joblib, holdout models
├── outputs/                      december_forecast.csv (deliverable), quantiles, battery schedules, KPIs
└── reports/
    ├── figures/                  all plots (01_*.png … 05_*.png)
    ├── tables/                   metric tables, decision log, task*_summary.json
    ├── analysis_task1_task2.md   one-page analysis (generated)
    └── optimization_report.md    optimisation report (generated)
```

## Setup in VS Code (Windows, Git Bash terminal)

1. Open the project folder in VS Code and click **Trust** (the "Restricted Mode" banner), otherwise Python cannot run.
2. Copy the three CSV files into `data/raw/`.
3. Open a terminal (**Terminal → New Terminal**) in the project folder:

```bash
python -m venv .venv
source .venv/Scripts/activate        # Linux / Mac: source .venv/bin/activate
pip install -r requirements.txt
```

4. Select the interpreter: **Ctrl+Shift+P → Python: Select Interpreter → .venv**.

## How to run

```bash
python scripts/run_all.py            # everything, a few minutes
pytest                               # unit tests (run after the scripts: test_service needs the models)
```

or one script at a time: `python scripts/01_data_inspection.py`, then 02 … 06 (the **Run** ▶ button in VS Code works too). Every script works from any folder, because all paths come from `src/project_paths.py`. Scripts print their results; comments marked `expected:` give the values found during development, so you can confirm your run.

## Service interfaces (Approach 1: local modules)

The Optimization Service imports and calls the Forecasting Service; it never reads forecast files.

### Forecasting Service: `src/forecasting/service.py`

```python
from forecasting.service import predict
predict(inputs: pd.DataFrame, quantiles: Sequence[float] = (0.1, 0.5, 0.9)) -> pd.DataFrame
```

| | |
|---|---|
| inputs | `timestamp` (column or index), `temperature_forecast_C`, `is_weekend`, `is_holiday`, `plant_state`, `forecast_available`; 15-minute grid |
| quantiles | any of 0.05, 0.1, 0.5, 0.9, 0.95 (the median is always added) |
| returns | `timestamp`, `load_kW` (median), `load_kW_lower` / `load_kW_upper` (lowest / highest requested quantile), `q05`…`q95`, `temp_imputed`, `adjustment` |
| errors | `ValueError`: missing columns, duplicate or unsorted timestamps, unknown plant state, unsupported quantile · `FileNotFoundError`: model files missing (run scripts 03 and 04) |

The model and interval tables are loaded **once** (`ForecastingService.__init__`, shared via `get_service()`).

### Optimization Service: `src/optimization/service.py`

```python
OptimizationService(params: BatteryParams, forecaster=None).run(inputs, prices, backend="pyomo")
    -> dict[str, StrategyRun]   # "A_deterministic", "B_conservative": schedule, KPIs, daily table
```

`battery.solve_schedule(inputs, params, backend)` builds the LP; `backend="pyomo"` uses Pyomo + HiGHS (`highspy`), `backend="scipy"` solves the identical LP with SciPy's HiGHS (automatic fallback if Pyomo is missing). `ValueError` if a forecast interval has no price; `RuntimeError` if no solver is available or the LP is not optimal.

## Key decisions

| Decision | Why |
|---|---|
| Exclude non-GOOD rows, isolated spikes and two mislabelled days (1 Aug RAMP_UP at full load, 15 Aug HIGH_DEMAND at ramp-up load) | inputs or labels are wrong there; the load target is never imputed |
| Features chosen by ablation on holdout months (simplest group within 1 % of the best) | expected result: core = quarter-hour, day of week, weekend, holiday, plant state |
| Month / week of year excluded | re-encode plant state, unseen in December, ISO week wraps to 1 on 29–31 Dec |
| Splits: holdout Nov (Jan–Oct), holdout Jan (Feb–Nov), blocked weekly CV, forward-chaining | December-like tests, metrics by state/month, failure analysis; final model retrained on all clean rows |
| Model selected by expected December error (0.935 NORMAL + 0.065 SHUTDOWN) | weights match December's mix of plant states |
| Unseen-holiday rule | trees would forecast ~0 kW for the shutdown on 25–26 Dec |
| P10/P90 delivered, P5/P95 also computed | consistent with Strategy B (P90), stable calibration per condition |
| LP, not MILP | simultaneous charge/discharge is never optimal (positive prices, losses, degradation both ways); verified on every run |
| Strategy B: P10 for the no-export limit, P90 for the grid limit, +50 kWh reserve on flagged days | load matters only where a constraint binds; export happens when load is lower than planned |

## Assumptions

- Timestamps are naive (no daylight-saving shifts: 96 rows on every day).
- `load_kW` is the median forecast; the delivered interval is P10–P90.
- December rows with `forecast_available = 0` are treated as missing (including 8 rows that still carry a value) and imputed.
- SOC is the state at the end of each interval; the 250 kWh reserve applies to the 23:45 value and carries over midnight.
- Throughput = charge energy + discharge energy at the battery terminals; degradation €0.015/kWh on both.
- Grid export is not allowed: grid import ≥ 0 in every interval; strategy B also keeps discharge below the low (P10) load forecast.
- No demand charge; cost = day-ahead price × imported energy.

- SHUTDOWN forecasts and interval bounds are floored at the 28 kW standby load observed in the history.

- Electricity prices will not change in realization and SOC is not degraded for anyother reason



