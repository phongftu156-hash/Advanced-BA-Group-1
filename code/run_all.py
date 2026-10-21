"""Run the whole pipeline: python code/run_all.py [--rescrape] [--no-exports]

Scraping (01) is skipped when data/raw/mogi_raw.csv already holds the target sample
(6,000 raw listings), unless --rescrape is passed.
Steps after scraping always run: cleaning -> OSM features -> EDA -> Modules 1-3 ->
cross-functional -> value -> numbers for the report -> tests -> app check -> report figures ->
standalone HTML dashboard (with JS parity and browser tests).
The report is maintained in Word and the slide deck in Canva, outside this pipeline.
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
STEPS = ["02_clean.py", "03_osm_features.py", "04_eda.py", "05_model1_avm.py", "06_model2_overpricing.py",
         "07_model3_segments.py", "08_crossfunctional.py", "09_value.py", "10_export_numbers.py"]


def run(cmd, cwd=ROOT):
    t0 = time.time()
    print(f"\n>>> {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"step failed: {cmd} (exit {r.returncode})")
    print(f"<<< done in {time.time() - t0:.0f} s", flush=True)


def scrape_needed():
    raw = ROOT / "data" / "raw" / "mogi_raw.csv"
    if not raw.exists():
        return True
    with open(raw, encoding="utf-8-sig") as f:
        n_rows = sum(1 for _ in csv.reader(f)) - 1
    return n_rows < 6000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescrape", action="store_true")
    ap.add_argument("--no-exports", "--no-report", dest="no_exports", action="store_true",
                    help="skip figure and dashboard exports (--no-report is a legacy alias)")
    a = ap.parse_args()
    code = ROOT / "code"
    if a.rescrape:
        run([PY, code / "01_scrape_mogi.py", "--fresh"])
        run([PY, code / "01b_backfill_img_dates.py"])
    elif scrape_needed():
        run([PY, code / "01_scrape_mogi.py"])
        run([PY, code / "01b_backfill_img_dates.py"])
    else:
        print("raw data already meets the target: scraping skipped (use --rescrape to force)")
    for s in STEPS:
        run([PY, code / s])
    run([PY, "-m", "pytest", "-q", str(code / "tests")])
    run([PY, code / "12_dashboard_check.py"])
    if not a.no_exports:
        run([PY, code / "14_report_figures.py"])        # report versions of the map, waterfall and heatmap
        run([PY, code / "17_export_dashboard.py"])      # standalone HTML dashboard + JS parity test
        run([PY, code / "18_dashboard_html_check.py"])  # headless browser test of the HTML dashboard
    print("\npipeline finished")


if __name__ == "__main__":
    main()
