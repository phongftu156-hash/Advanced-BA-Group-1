"""Phase 4.5: business value scenarios (illustrative, assumption-driven) and tornado chart.

Every input is either measured in this project (source = result file) or a labelled team
assumption. Partner value (broker time, bank appraisal cost) is compared with the run cost
to give the ROI; buyer savings are reported separately because they do not accrue to the
operator. Outputs: tables/value_assumptions.csv, value_scenarios.csv, value_tornado.csv,
figures/value_tornado.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DGREY, LGREY, MGREY, MODELS, ORANGE, BLUE, PROCESSED, TAB, W2, get_logger, savefig, set_style  # noqa: E402

log = get_logger("09_value")
set_style()
TA = "team assumption"


def measured():
    df = pd.read_parquet(PROCESSED / "listings_model.parquet")
    avm = pd.read_csv(MODELS / "avm_predictions.csv")
    m = df[["listing_id", "price_vnd"]].merge(avm, on="listing_id")
    # statistically clear overpricing: asking price above the upper 90% conformal bound
    hi = np.log(m["avm_hi90_bn"] * 1e9)
    above = m["ln_price"] > hi
    excess = (m.loc[above, "price_vnd"] - m.loc[above, "avm_hi90_bn"] * 1e9) / m.loc[above, "price_vnd"]
    met = pd.read_csv(TAB / "avm_metrics.csv").set_index("model")
    best = "XGBoost" if met.loc["XGBoost", "RMSE_log"] <= met.loc["OLS (4)", "RMSE_log"] else "OLS (4)"
    return {"share_above": float(above.mean()), "excess": float(excess.median()) if above.any() else 0.0,
            "median_price_bn": float(df["price_vnd"].median() / 1e9), "ppe10": float(met.loc[best, "PPE10"])}


def assumptions(meas):
    A = [
        # name, label, conservative, base, optimistic, unit, source
        ("brokers", "Brokers using the tool", 100, 300, 500, "brokers", TA + " (pilot with 1 to 3 agencies)"),
        ("valuations", "Valuations per broker per month", 10, 20, 25, "per month", TA),
        ("hours_saved", "Hours saved per valuation vs manual comparables", 0.5, 1.0, 1.25, "hours", TA),
        ("broker_cost", "Broker cost per hour", 0.10, 0.15, 0.20, "VND m/hour", TA),
        ("applications", "Mortgage applications pre-screened per year (partner bank)", 1000, 3000, 6000,
         "per year", TA),
        ("ppe10", "Share of AVM estimates within ±10% (PPE10)", meas["ppe10"], meas["ppe10"], meas["ppe10"],
         "share", "measured: result/tables/avm_metrics.csv"),
        ("fast_track", "Share of PPE10 cases where the bank waives a full appraisal", 0.3, 0.5, 0.7, "share", TA),
        ("appraisal_cost", "Cost of a manual appraisal avoided", 1.5, 2.5, 4.0, "VND m", TA),
        ("run_cost", "Annual run cost (3 FTE data team, cloud, data refresh)", 3000, 2200, 1600, "VND m/year", TA),
        ("buyers", "Buyer transactions guided by the tool per year", 300, 1000, 2500, "per year", TA),
        ("share_above", "Share of listings priced above the AVM 90% upper bound", meas["share_above"],
         meas["share_above"], meas["share_above"], "share", "measured: result/models/avm_predictions.csv"),
        ("excess", "Median asking-price excess over the upper bound (those listings)", meas["excess"],
         meas["excess"], meas["excess"], "share of price", "measured: result/models/avm_predictions.csv"),
        ("pass_through", "Share of that excess removed in negotiation", 0.10, 0.20, 0.30, "share", TA),
        ("price_bn", "Median asking price", meas["median_price_bn"], meas["median_price_bn"],
         meas["median_price_bn"], "VND bn", "measured: data/processed/listings_model.parquet"),
    ]
    a = pd.DataFrame(A, columns=["name", "label", "conservative", "base", "optimistic", "unit", "source"])
    a.to_csv(TAB / "value_assumptions.csv", index=False, encoding="utf-8-sig")
    return a


def value(p):
    """Annual value in VND million by lever."""
    broker = p["brokers"] * p["valuations"] * 12 * p["hours_saved"] * p["broker_cost"]
    bank = p["applications"] * p["ppe10"] * p["fast_track"] * p["appraisal_cost"]
    buyer = p["buyers"] * p["share_above"] * p["excess"] * p["pass_through"] * p["price_bn"] * 1000
    partner = broker + bank
    return {"Broker time saved": broker, "Bank appraisal cost avoided": bank, "Partner value": partner,
            "Run cost": p["run_cost"], "Net partner value": partner - p["run_cost"],
            "ROI (net / run cost)": (partner - p["run_cost"]) / p["run_cost"],
            "Buyer overpayment avoided (not in ROI)": buyer}


def main():
    meas = measured()
    a = assumptions(meas)
    sc = pd.DataFrame({s: value(dict(zip(a["name"], a[s]))) for s in ("conservative", "base", "optimistic")})
    sc.to_csv(TAB / "value_scenarios.csv", encoding="utf-8-sig")
    log.info("scenarios (VND million/year):\n" + sc.round(2).to_string())

    base = dict(zip(a["name"], a["base"]))
    b = value(base)["Net partner value"]
    rows = []
    for _, r in a.iterrows():
        if r["conservative"] == r["optimistic"] or r["name"] in ("buyers", "pass_through"):
            continue
        lo = value({**base, r["name"]: r["conservative"]})["Net partner value"]
        hi = value({**base, r["name"]: r["optimistic"]})["Net partner value"]
        rows.append({"assumption": r["label"], "low": lo, "high": hi, "swing": abs(hi - lo)})
    tor = pd.DataFrame(rows).sort_values("swing", ascending=False)
    tor.to_csv(TAB / "value_tornado.csv", index=False)
    fig, ax = plt.subplots(figsize=(W2, 5.4 / 2.54))
    for i, r in enumerate(tor.itertuples()):
        ax.barh(i, r.low - b, left=b, color=LGREY, height=0.6)
        ax.barh(i, r.high - b, left=b, color=BLUE, height=0.6)
        ax.text(min(r.low, r.high), i, f"{min(r.low, r.high):,.0f} ", va="center", ha="right", fontsize=6.8, color=MGREY)
        ax.text(max(r.low, r.high), i, f" {max(r.low, r.high):,.0f}", va="center", ha="left", fontsize=6.8, color=DGREY)
    ax.axvline(b, color=ORANGE, lw=0.9)
    ax.text(b, -0.8, f"  Base case {b:,.0f}", color=ORANGE, fontsize=7, ha="left", va="center",
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=5)
    ax.set_yticks(range(len(tor)), tor["assumption"], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel("Annual net partner value (VND million); grey = conservative input, blue = optimistic input")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    span = tor[["low", "high"]].to_numpy()
    w = span.max() - span.min()
    ax.set_xlim(span.min() - 0.22 * w, span.max() + 0.22 * w)
    ax.set_ylim(len(tor) - 0.5, -1.1)
    fig.tight_layout()
    savefig(fig, "value_tornado")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
