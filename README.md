# Vietnam Real Estate AVM

Automated valuation and market intelligence for secondary-market homes in Hanoi and Ho Chi Minh City
(Advanced Business Analytics, final project, October 2026). Team: Advanced Business Analytics capstone group.

The platform has three engines, trained on public listings and open map data:

- **Fair Value Engine**: hedonic regression and gradient boosting (XGBoost) valuation with a 90% split-conformal range.
- **Stale Listing Radar**: XGBoost classifier that ranks listings by the risk of staying unsold for more than 180 days.
- **Market Map**: K-Means segmentation of districts by price, speed, access, housing mix, price gap and volume.

## Deliverables

| File | What it is |
|---|---|
| `report/main.pdf` | Final report (15 pages: executive summary on page 1, body to page 12, technical appendix pages 13 to 15) |
| `report/main.tex` | Single self-contained source of the report (style, numbers, diagrams, tables and bibliography inlined) |
| `report/slides/pitch.pdf`, `pitch.tex` | 15-slide pitch deck and its single-file source |
| `report/response_to_feedback.pdf` | One-page response to the instructor feedback |
| `report/vietnam_avm_dashboard.html` | Standalone dashboard: open by double-click; valuations run in the browser |
| `code/app.py` | Streamlit version of the dashboard |
| `data/`, `result/` | Raw and processed data, figures, tables and trained models |

The executive summary is page 1 of `report/main.pdf`.

## Results summary

- **Data:** 6,000 mogi.vn sale listings collected on 15 October 2026 (46 minutes, 7,792 listings an hour, no blocks),
  5,268 after cleaning (2,638 Hanoi, 2,630 HCMC, 51 districts), matched to 34,321 OpenStreetMap points.
  A prototype sample sized for a 1.5-hour collection window.
- **Fair Value Engine:** 53.9% of unseen homes within ±20% of the asking price (PPE20); alley houses 64%,
  apartments 60%, villas 47%, land 43%, street-front houses 42%. Median error 18.1%; RMSE 0.354 on the log scale
  (target 0.20 missed). Regression R² 0.850 (target 0.75 met). Map data alone recovers 60% of the accuracy gain
  from knowing the district.
- **Stale Listing Radar:** AUC 0.81 (95% interval 0.78 to 0.83) against 0.74 for logistic regression and 0.45 for a
  price-gap rule. The top 10% of scores are stale 1.5 times as often as average (99% against 65%). The price gap
  ranks 12 of 26 drivers: stale listings are not mainly overpriced listings.
- **Market Map:** four segments (Prime core, Apartment mid-market, Townhouse mid-market, Peripheral growth) over
  34 districts with at least 30 listings; silhouette 0.21 (target 0.45 missed), so segments are targeting groups.
- **Value:** base case net partner value VND 9.7 billion a year, 4.4 times the run cost; scenarios from a loss of
  VND 2.3 billion to a gain of VND 40.9 billion. Broker adoption moves the result most.
- **Dashboard parity:** the HTML dashboard reproduces the Python models on 500 random listings within 5.7e-6 on log
  price and 3.3e-7 on probability; three browser test cases match the Streamlit app exactly.

## Key design decisions

1. **Sampling quotas** per city: apartments 900, alley houses 800, street-front houses 500, villas and terraced 350,
   land 450; shortfalls (Hanoi apartments ran out at 675) reallocated; a fifth of each quota from the last result
   pages because the site has no oldest-first sort. Type shares reflect quotas, not the market.
2. **Stale threshold 180 days:** at 90 days 82% of listings would be stale; 180 days gives 65%. Checks at 60, 90 and
   120 days are in the appendix.
3. **Listing age from the creation date implied by the listing ID** (calibrated to first photo-upload dates), because
   posted dates are refreshed when sellers re-post (59% re-posted more than 30 days after their photos).
4. **Leakage safeguards:** no listing age, dates, listing ID or poster account age among the radar's inputs; the price
   gap fed to the radar comes from out-of-fold predictions; every accuracy figure is on held-out data.
5. **Administrative names:** Vietnam abolished districts on 1 July 2025, but mogi.vn still lists pre-2025 districts,
   so the analysis and the map (GADM 4.1 boundaries) use the district as listed; Quận 2, Quận 9 and Quận Thủ Đức
   stay separate.
6. **Cleaning:** rentals posted as sales, placeholder or mistyped prices, areas out of range, more than 15 bedrooms,
   the 1st and 99th percentile of price per m² by city and type, and near-duplicate postings are removed (88% kept).
7. **Location features:** distances and counts from OpenStreetMap (operating metro stations only) plus one
   accessibility index; about 71% of map pins are ward-level, which is stated as a limitation.
8. **Market Map:** standardised district features winsorised at ±2.5 SD, listing age as a log median, k = 4 as the
   best silhouette within 3 to 5 segments.
9. **Value case:** ROI counts partner value only (broker time, bank appraisals) against a 3-person team; buyer savings
   count only the excess above the 90% range and stay outside the ROI.
10. **No re-training after the models were fixed:** the report revision and the HTML dashboard reuse the saved models.

## Folder structure

```
Advanced BA/
  README.md, requirements.txt
  code/                pipeline 01 to 18, run_all.py, app.py, helpers, dashboard engine and template, tests/
  data/raw/            mogi_raw.csv (scraped listings)
  data/external/osm/   OpenStreetMap layers (Overpass downloads, including metro lines)
  data/external/boundaries/  GADM 4.1 district boundaries (Vietnam, level 2)
  data/processed/      listings_model.parquet (analysis-ready table)
  result/figures/      PNG + PDF for every exhibit, interactive segments_map_*.html
  result/tables/       CSV tables behind every exhibit and number
  result/models/       trained models and lookups used by app.py and the HTML dashboard
  report/              main.tex, main.pdf, response_to_feedback.pdf, vietnam_avm_dashboard.html
  report/slides/       pitch.tex, pitch.pdf
```

## Recreate the environment (Windows)

```
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m playwright install chromium   # dashboard screenshots and the HTML test
```
LaTeX: MiKTeX with XeLaTeX (fonts TeX Gyre Pagella and Heros, loaded by file name).

## Rerun the pipeline

```
.venv\Scripts\python code\run_all.py              # cleaning -> map features -> models -> exhibits -> dashboard -> PDFs
.venv\Scripts\python code\run_all.py --rescrape   # also re-collect the listings (about 50 minutes)
.venv\Scripts\python code\run_all.py --no-report  # stop after the models, tests and the app check
```
Collection is skipped when `data/raw/mogi_raw.csv` already holds 6,000 listings. Seeds are fixed at 42.
`code/10_export_numbers.py` writes every number cited in the report into the macro block between
`% BEGIN NUMBERS` and `% END NUMBERS` in `report/main.tex` and `report/slides/pitch.tex`
(and into `result/tables/report_numbers.csv`). Compile the report with `xelatex main.tex` twice from `report/`.
Intermediate files (parsed listings, logs, parity test data) are recreated in `data/` and `result/logs/` on a rerun.

## Open the dashboard

- **Standalone:** double-click `report/vietnam_avm_dashboard.html`. Valuation, comparables and charts work offline;
  the map draws district polygons without internet and adds OpenStreetMap tiles when online.
- **Streamlit:** `.venv\Scripts\python -m streamlit run code\app.py`
