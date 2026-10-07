"""Run the whole pipeline in order: 01 inspection -> 02 features -> 03 Task 1 -> 04 Task 2 -> 05 Task 3 -> 06 reports.

What this script does
---------------------
Starts each script with the same Python interpreter (your .venv) and stops at the first error.
Each script prints its own results; together they take a few minutes.

Run from anywhere:  python scripts/run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = ["01_data_inspection.py", "02_feature_analysis.py", "03_forecast_task1.py",
           "04_uncertainty_task2.py", "05_battery_task3.py", "06_write_reports.py"]
here = Path(__file__).resolve().parent

for name in SCRIPTS:
    print("\n" + "#" * 90 + f"\n# RUNNING {name}\n" + "#" * 90, flush=True)
    t0 = time.time()
    result = subprocess.run([sys.executable, str(here / name)])
    if result.returncode != 0:
        sys.exit(f"\n{name} failed (exit code {result.returncode}); fix it and run again.")
    print(f"# {name} finished in {time.time() - t0:.0f} s", flush=True)
print("\nAll steps finished. Results: outputs/, reports/tables/, reports/figures/, reports/*.md")
