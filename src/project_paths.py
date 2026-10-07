"""Project paths, resolved from this file's location.

"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # the project folder
DATA_RAW = ROOT / "data" / "raw"                    # put the three CSV files hist, dec and elect price here
CONFIG = ROOT / "config"
MODELS = ROOT / "models"                            # trained artifacts (created by scripts 03 and 04)
OUTPUTS = ROOT / "outputs"                          # forecasts and schedules
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"                         # metric tables and JSON summaries
FIGURES = REPORTS / "figures"                       # all plots

HIST_CSV = DATA_RAW / "historical_data.csv"
DEC_CSV = DATA_RAW / "forecast_input_december.csv"
PRICE_CSV = DATA_RAW / "electricity_prices_december.csv"
BATTERY_JSON = CONFIG / "battery.json"


def ensure_folders() -> None:
    """Create the output folders and check that the input CSVs are in place."""
    for folder in (MODELS, OUTPUTS, TABLES, FIGURES):
        folder.mkdir(parents=True, exist_ok=True)
    missing = [p.name for p in (HIST_CSV, DEC_CSV, PRICE_CSV) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Copy these files into {DATA_RAW}: {missing}")
