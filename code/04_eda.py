"""Phase 4.0: short exploratory analysis and the data exhibits.

Outputs: tables/descriptives_city_type.csv, tables/eda_correlations.csv,
figures/cleaning_waterfall, sample_size, ppm2_distribution, ppm2_vs_cbd, listing_age_hist.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (BLUE, CITY_SHORT, DGREY, LGREY, MGREY, ORANGE, PROCESSED, TAB, TYPE_EN,  # noqa: E402
                    TYPE_ORDER, W1, W2, get_logger, savefig, set_style)

log = get_logger("04_eda")
set_style()


def main():
    df = pd.read_parquet(PROCESSED / "listings_model.parquet")
    df["city_en"] = df["city"].map(CITY_SHORT)

    d = df.groupby(["city_en", "type_en"]).agg(
        N=("listing_id", "size"), med_price_bn=("price_bn", "median"), med_ppm2_mil=("ppm2_mil", "median"),
        med_area=("area_m2", "median"), med_age_created=("age_id_days", "median"),
        med_age_posted=("listing_age_days", "median")).reset_index()
    tot = df.groupby("city_en").agg(
        N=("listing_id", "size"), med_price_bn=("price_bn", "median"), med_ppm2_mil=("ppm2_mil", "median"),
        med_area=("area_m2", "median"), med_age_created=("age_id_days", "median"),
        med_age_posted=("listing_age_days", "median")).reset_index().assign(type_en="All types")
    d = pd.concat([d, tot], ignore_index=True)
    order = {t: i for i, t in enumerate(list(TYPE_EN.values()) + ["All types"])}
    d = d.sort_values(["city_en", "type_en"], key=lambda s: s.map(order) if s.name == "type_en" else s)
    d.to_csv(TAB / "descriptives_city_type.csv", index=False)
    cor = []
    for city, g in df.groupby("city_en"):
        for v in ("dist_cbd_km", "dist_metro_km", "accessibility_index"):
            rho, p = spearmanr(g["ppm2_mil"], g[v])
            cor.append({"city": city, "variable": v, "spearman_rho": rho, "p_value": p, "n": len(g)})
            for t, gt in g.groupby("type_en"):
                rho, p = spearmanr(gt["ppm2_mil"], gt[v])
                cor.append({"city": city, "type": t, "variable": v, "spearman_rho": rho, "p_value": p, "n": len(gt)})
    pd.DataFrame(cor).to_csv(TAB / "eda_correlations.csv", index=False)
    log.info("correlations (city level):\n" + pd.DataFrame(cor).query("type != type").round(3).to_string(index=False))

    waterfall()
    sample_size(df)
    ppm2_distribution(df)
    ppm2_vs_cbd(df)
    age_hist(df)


def waterfall():
    w = pd.read_csv(TAB / "cleaning_waterfall.csv")
    w = w[(w.index == 0) | (w["n_dropped"] > 0)].reset_index(drop=True)  # steps that removed rows
    fig, ax = plt.subplots(figsize=(W1, 7.0 / 2.54))
    labels = ["Raw scraped"] + [s.replace("Remove ", "").replace(" (city x type)", "")
                                .replace("price per m2 outside 1st to 99th percentile", "price/m² outside P1 to P99")
                                .replace("near-duplicate postings (keep earliest)", "near-duplicates")
                                .replace("listings without valid coordinates", "no valid coordinates")
                                .replace("negotiable or unparseable price", "negotiable price")
                                .replace("area outside type bounds", "area out of bounds")
                                .replace("bedroom count above 15", "> 15 bedrooms")
                                .replace("rental listings posted as sales", "rentals posted as sales")
                                .replace("implausible price (total < VND 300 m or price/m2 below type floor)", "implausible price")
                                .replace("missing or zero area", "missing area")
                                .replace("repeated listing IDs", "repeated IDs")
                                for s in w["step"][1:]] + ["Clean sample"]
    vals = list(w["n_remaining"])
    x = np.arange(len(labels))
    ax.bar(0, vals[0], color=MGREY, width=0.6)
    ax.text(0, vals[0], f"{vals[0]:,}", ha="center", va="bottom", fontsize=7)
    for i in range(1, len(vals)):
        drop = vals[i - 1] - vals[i]
        ax.bar(i, drop, bottom=vals[i], color=ORANGE if drop > 0 else LGREY, width=0.6)
        ax.text(i, vals[i - 1], f"−{drop:,}", ha="center", va="bottom", fontsize=6.3, color=ORANGE if drop else MGREY)
    ax.bar(len(vals), vals[-1], color=BLUE, width=0.6)
    ax.text(len(vals), vals[-1], f"{vals[-1]:,}", ha="center", va="bottom", fontsize=7, color=BLUE, fontweight="bold")
    ax.set_xticks(x, labels, rotation=40, ha="right", fontsize=6.5)
    ax.set_ylabel("Listings")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(min(vals) * 0.85, vals[0] * 1.06)
    ax.text(0.99, 0.98, f"Axis starts at {min(vals) * 0.85:,.0f}", transform=ax.transAxes, ha="right",
            va="top", fontsize=6.5, color=MGREY)
    fig.tight_layout()
    savefig(fig, "cleaning_waterfall")


def sample_size(df):
    ct = pd.crosstab(df["type_en"], df["city_en"]).reindex([TYPE_EN[t] for t in TYPE_ORDER])
    fig, ax = plt.subplots(figsize=(W1, 5.2 / 2.54))
    y = np.arange(len(ct))
    ax.barh(y - 0.19, ct["Hanoi"], height=0.36, color=BLUE)
    ax.barh(y + 0.19, ct["HCMC"], height=0.36, color=LGREY)
    for i, (a, b) in enumerate(zip(ct["Hanoi"], ct["HCMC"])):
        ax.text(a, i - 0.19, f" {a:,}", va="center", fontsize=6.8, color=BLUE)
        ax.text(b, i + 0.19, f" {b:,}", va="center", fontsize=6.8, color=MGREY)
    ax.set_yticks(y, ct.index)
    ax.invert_yaxis()
    ax.set_xlabel("Clean listings")
    ax.text(0.98, 0.02, "Hanoi", color=BLUE, transform=ax.transAxes, ha="right", fontsize=7.5, fontweight="bold")
    ax.text(0.98, 0.12, "HCMC", color=MGREY, transform=ax.transAxes, ha="right", fontsize=7.5, fontweight="bold")
    ax.set_xlim(0, ct.to_numpy().max() * 1.2)
    fig.tight_layout()
    savefig(fig, "sample_size")


def ppm2_distribution(df):
    types = [TYPE_EN[t] for t in TYPE_ORDER]
    fig, ax = plt.subplots(figsize=(W2, 4.8 / 2.54))
    pos, data, cols = [], [], []
    for i, t in enumerate(types):
        for j, c in enumerate(["Hanoi", "HCMC"]):
            v = df.loc[(df["type_en"] == t) & (df["city_en"] == c), "ppm2_mil"]
            pos.append(i + (j - 0.5) * 0.36)
            data.append(v.to_numpy())
            cols.append(BLUE if c == "Hanoi" else MGREY)
    bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=False,
                    medianprops={"color": "white", "lw": 1.2}, whiskerprops={"lw": 0.7}, capprops={"lw": 0.7})
    for b, c in zip(bp["boxes"], cols):
        b.set_facecolor(c)
        b.set_edgecolor(c)
    for w, c in zip(bp["whiskers"], np.repeat(cols, 2)):
        w.set_color(c)
    for cp, c in zip(bp["caps"], np.repeat(cols, 2)):
        cp.set_color(c)
    for p, v, c in zip(pos, data, cols):
        ax.text(p, np.median(v) * 1.0, "", fontsize=6)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xticks(range(len(types)), types)
    ax.set_ylabel("Asking price (million VND/m², log)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.text(0.01, 0.97, "Hanoi", color=BLUE, transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="top")
    ax.text(0.09, 0.97, "HCMC", color=MGREY, transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="top")
    fig.tight_layout()
    savefig(fig, "ppm2_distribution")


def ppm2_vs_cbd(df):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 5.0 / 2.54), sharey=True)
    out = []
    for ax, c in zip(axes, ["Hanoi", "HCMC"]):
        g = df[df["city_en"] == c]
        x, y = g["dist_cbd_km"], g["ppm2_mil"]
        ax.scatter(x, y, s=1.5, color=LGREY, alpha=0.5, lw=0)
        bins = pd.qcut(x, 20, duplicates="drop")
        b = g.groupby(bins, observed=True).agg(x=("dist_cbd_km", "median"), y=("ppm2_mil", "median"))
        ax.plot(b["x"], b["y"], "o", color=BLUE if c == "Hanoi" else ORANGE, ms=3)
        slope, icpt = np.polyfit(np.log(x + 0.1), np.log(y), 1)
        xs = np.linspace(x.min(), x.quantile(0.99), 100)
        ax.plot(xs, np.exp(icpt) * (xs + 0.1) ** slope, color=DGREY, lw=0.9)
        out.append({"city": c, "elasticity_ppm2_dist_cbd": slope})
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlabel("Distance to CBD (km)")
        ax.set_title("Hanoi (Hoàn Kiếm Lake)" if c == "Hanoi" else "HCMC (Bến Thành Market)")
        ax.set_xlim(0, x.quantile(0.99))
        ax.text(0.97, 0.95, f"Log-log slope {slope:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=LGREY, lw=0.5), zorder=5)
    axes[0].set_ylabel("Asking price (million VND/m², log)")
    fig.tight_layout()
    savefig(fig, "ppm2_vs_cbd")
    pd.DataFrame(out).to_csv(TAB / "eda_cbd_gradient.csv", index=False)


def age_hist(df):
    fig, ax = plt.subplots(figsize=(W1, 5.0 / 2.54))
    bins = np.arange(0, 760, 30)
    ax.hist(df["listing_age_days"].clip(upper=749), bins=bins, color=LGREY, label="Since posted date (refreshed)")
    ax.hist(df["age_id_days"].clip(upper=749), bins=bins, histtype="step", color=BLUE, lw=1.3,
            label="Since creation (listing ID)")
    ax.axvline(90, color=ORANGE, lw=0.8, ls="--")
    ax.text(95, ax.get_ylim()[1] * 0.92, "90 days", color=ORANGE, fontsize=7)
    ax.set_xlabel("Listing age (days)")
    ax.set_ylabel("Listings")
    ax.legend(fontsize=6.6, loc="upper right")
    fig.tight_layout()
    savefig(fig, "listing_age_hist")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
