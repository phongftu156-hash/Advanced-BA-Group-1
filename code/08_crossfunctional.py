"""Phase 4.4: cross-functional evidence built from the model outputs.

Sales & brokerage : share of high stale-risk listings by district; median AVM gap by type.
Risk & credit     : AVM error by type and city (out-of-sample); share over-valued by > 20%.
Marketing         : segment profiles (Module 3 outputs, reused).
Executive         : executive matrix (Module 3 figure, reused).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (BLUE, CITY_SHORT, DGREY, LGREY, MGREY, MODELS, ORANGE, PROCESSED, TAB,  # noqa: E402
                    TYPE_EN, TYPE_ORDER, W2, get_logger, savefig, set_style)

log = get_logger("08_crossfunctional")
set_style()


def main():
    df = pd.read_parquet(PROCESSED / "listings_model.parquet")
    avm = pd.read_csv(MODELS / "avm_predictions.csv")
    sc = pd.read_csv(MODELS / "overpricing_scores.csv")[["listing_id", "p_stale", "high_risk"]]
    seg = pd.read_csv(TAB / "district_segments.csv")[["city", "district", "segment"]]
    df = df.merge(avm[["listing_id", "avm_pred_ln", "avm_resid", "split"]], on="listing_id") \
           .merge(sc, on="listing_id", how="left").merge(seg, on=["city", "district"], how="left")
    df["city_en"] = df["city"].map(CITY_SHORT)
    # all AVM predictions are out-of-sample: out-of-fold (train rows) or held-out (test rows)
    df["ratio"] = np.exp(df["avm_pred_ln"] - df["ln_price"])   # AVM value / asking price
    df["ape"] = np.abs(df["ratio"] - 1)

    # ---------------- Sales & brokerage
    dist = df.groupby(["city_en", "district"]).agg(n=("listing_id", "size"), share_high_risk=("high_risk", "mean"),
                                                   med_gap=("avm_resid", "median")).reset_index()
    dist = dist[dist["n"] >= 30].sort_values("share_high_risk", ascending=False)
    dist.to_csv(TAB / "xf_sales_districts.csv", index=False, encoding="utf-8-sig")
    gap = df.groupby("type_en").agg(med_gap=("avm_resid", "median"), share_over15=("avm_resid", lambda s: (s > np.log(1.15)).mean()),
                                    share_high_risk=("high_risk", "mean"), n=("listing_id", "size"))
    gap = gap.reindex([TYPE_EN[t] for t in TYPE_ORDER])
    gap.to_csv(TAB / "xf_sales_types.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 7.0 / 2.54), gridspec_kw={"width_ratios": [1.35, 1]})
    top = pd.concat([dist.head(8), dist.tail(4)])
    lab = [f"{d.replace('Quận ', 'Q. ').replace('Huyện ', 'H. ').replace(' (TP. Thủ Đức)', '')} ({c})"
           for c, d in zip(top["city_en"], top["district"])]
    y = np.arange(len(top))
    overall = df["high_risk"].mean()
    a1.barh(y, top["share_high_risk"], color=[ORANGE if v > overall else LGREY for v in top["share_high_risk"]], height=0.65)
    for i, v in enumerate(top["share_high_risk"]):
        a1.text(v, i, f" {v:.0%}", va="center", fontsize=6.8)
    a1.axvline(overall, color=DGREY, lw=0.7, ls="--")
    a1.set_ylim(len(top) - 0.5, -1.5)
    a1.text(overall, -0.95, f" All listings {overall:.0%}", fontsize=6.8, ha="left", va="center")
    a1.set_yticks(y, lab, fontsize=6.8)
    a1.set_xlabel("Share of district listings in the top risk quartile")
    a1.set_title("Top-quartile stale risk: highest and lowest districts")
    a1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a1.set_xlim(0, min(1.0, top["share_high_risk"].max() * 1.2))
    yy = np.arange(len(gap))
    vals = gap["share_over15"]
    a2.barh(yy, vals, color=[ORANGE if v == vals.max() else LGREY for v in vals], height=0.6)
    for i, v in enumerate(vals):
        a2.text(v, i, f" {v:.0%}", va="center", fontsize=6.8)
    a2.set_yticks(yy, gap.index, fontsize=7)
    a2.invert_yaxis()
    a2.set_xlabel("Share of listings priced > 15% above AVM value")
    a2.set_title("Repricing candidates by type")
    a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a2.set_xlim(0, vals.max() * 1.3)
    fig.tight_layout()
    savefig(fig, "xf_sales")

    # ---------------- Risk & credit
    rk = df.groupby(["city_en", "type_en"]).agg(
        n=("listing_id", "size"), PPE10=("ape", lambda s: (s <= 0.10).mean()), PPE20=("ape", lambda s: (s <= 0.20).mean()),
        MdAPE=("ape", "median"), over20=("ratio", lambda s: (s > 1.20).mean()),
        under20=("ratio", lambda s: (s < 0.80).mean())).reset_index()
    rt = df.groupby("type_en").agg(
        n=("listing_id", "size"), PPE10=("ape", lambda s: (s <= 0.10).mean()), PPE20=("ape", lambda s: (s <= 0.20).mean()),
        MdAPE=("ape", "median"), over20=("ratio", lambda s: (s > 1.20).mean())).reindex([TYPE_EN[t] for t in TYPE_ORDER])
    # pre-screening rule: PPE20 >= 60% and over-valuation share <= 10% -> AVM usable for pre-screening
    rt["use"] = np.where((rt["PPE20"] >= 0.6) & (rt["over20"] <= 0.10), "Pre-screen with AVM",
                         np.where(rt["PPE20"] >= 0.45, "AVM + desk review", "Manual appraisal"))
    rk.to_csv(TAB / "xf_risk_city_type.csv", index=False)
    rt.to_csv(TAB / "xf_risk_types.csv")
    log.info("risk by type:\n" + rt.round(3).to_string())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 6.4 / 2.54), gridspec_kw={"width_ratios": [1.5, 1]})
    types = [TYPE_EN[t] for t in TYPE_ORDER]
    pos, data, cols = [], [], []
    for i, t in enumerate(types):
        for j, c in enumerate(["Hanoi", "HCMC"]):
            v = (df.loc[(df["type_en"] == t) & (df["city_en"] == c), "ratio"] - 1).to_numpy()
            pos.append(i + (j - 0.5) * 0.36)
            data.append(v)
            cols.append(BLUE if c == "Hanoi" else MGREY)
    bp = a1.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=False,
                    medianprops={"color": "white", "lw": 1.1}, whiskerprops={"lw": 0.7}, capprops={"lw": 0.7})
    for b, c in zip(bp["boxes"], cols):
        b.set_facecolor(c)
        b.set_edgecolor(c)
    for w, c in zip(bp["whiskers"], np.repeat(cols, 2)):
        w.set_color(c)
    for cp, c in zip(bp["caps"], np.repeat(cols, 2)):
        cp.set_color(c)
    a1.axhspan(-0.2, 0.2, color=ORANGE, alpha=0.08, lw=0)
    a1.axhline(0, color=MGREY, lw=0.5)
    a1.set_xticks(range(len(types)), [t.replace(" / ", "/\n").replace("Street-front house", "Street-front\nhouse") for t in types], fontsize=6.8)
    a1.set_ylabel("AVM value / asking price − 1")
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    a1.set_title("Out-of-sample valuation error (box: P25 to P75; shaded: ±20%)")
    a1.text(0.01, 0.97, "Hanoi", color=BLUE, transform=a1.transAxes, fontsize=7.2, fontweight="bold", va="top")
    a1.text(0.12, 0.97, "HCMC", color=MGREY, transform=a1.transAxes, fontsize=7.2, fontweight="bold", va="top")
    yy = np.arange(len(rt))
    a2.barh(yy, rt["over20"], color=[ORANGE if v > 0.10 else BLUE for v in rt["over20"]], height=0.6)
    for i, v in enumerate(rt["over20"]):
        a2.text(v, i, f" {v:.0%}", va="center", fontsize=6.8)
    a2.axvline(0.10, color=DGREY, lw=0.6, ls="--")
    a2.set_yticks(yy, rt.index, fontsize=7)
    a2.invert_yaxis()
    a2.set_xlabel("Share where AVM exceeds asking price by > 20%")
    a2.set_title("Collateral over-valuation risk")
    a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a2.set_xlim(0, max(rt["over20"].max() * 1.35, 0.15))
    fig.tight_layout()
    savefig(fig, "xf_risk")

    # ---------------- Marketing: segment table with volumes and prices (from Module 3)
    prof = pd.read_csv(TAB / "segment_profiles.csv")
    prof.to_csv(TAB / "xf_marketing_segments.csv", index=False, encoding="utf-8-sig")
    seg_type = pd.crosstab(df["segment"], df["type_en"], normalize="index")
    seg_type.to_csv(TAB / "xf_marketing_segment_typemix.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
