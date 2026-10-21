"""Report-only figure rebuilds from saved results (no model is re-trained).

figures/segments_maps    choropleth of district segments on GADM 4.1 district boundaries, with metro lines
figures/segments_profile segment heatmap with plain, horizontal labels and the silhouette panel
figures/cleaning_waterfall horizontal cleaning waterfall
Inputs: result/tables/district_segments.csv, segment_k_selection.csv, segment_colors.csv, cleaning_waterfall.csv,
data/external/boundaries/gadm41_VNM_2.json, data/external/osm/*_metro*.json.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (BLUE, DGREY, EXTERNAL, LGREY, MGREY, ORANGE, OSM, PROCESSED, TAB, W2,  # noqa: E402
                    get_logger, savefig, set_style)

log = get_logger("14_report_figures")
set_style()
GADM = EXTERNAL / "boundaries" / "gadm41_VNM_2.json"
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_VNM_2.json.zip"
CITY_GADM = {"Hà Nội": "HàNội", "TP. Hồ Chí Minh": "HồChíMinh"}
SLUG = {"Hà Nội": "hanoi", "TP. Hồ Chí Minh": "hcmc"}
FEATS = ["ln_price_index", "ln_med_age", "med_access", "share_house_land", "med_resid", "ln_volume"]
PLAIN = {"ln_price_index": "Price\nlevel", "ln_med_age": "Listing\nage", "med_access": "Access-\nibility",
         "share_house_land": "Houses and\nland share", "med_resid": "Price\ngap", "ln_volume": "Listing\nvolume"}


def key(name):
    """Matching key: no prefix, no parenthesis, no diacritics, no spaces, lower case."""
    from unidecode import unidecode
    s = re.sub(r"\(.*?\)", "", str(name))
    s = re.sub(r"^(Quận|Huyện|Thị xã|Thị Xã|Thịxã|Thành phố|Thànhphố|Q\.|H\.)\s*", "", s.strip())
    return re.sub(r"[^a-z0-9]", "", unidecode(s).lower())


def boundaries():
    import geopandas as gpd
    if not GADM.exists():
        import io
        import zipfile
        import requests
        GADM.parent.mkdir(parents=True, exist_ok=True)
        z = zipfile.ZipFile(io.BytesIO(requests.get(GADM_URL, timeout=300).content))
        z.extractall(GADM.parent)
    g = gpd.read_file(GADM)
    return g[g["NAME_1"].isin(CITY_GADM.values())].to_crs(3857)


def metro(city):
    """Operating metro lines (ways near an operating station) and stations, in EPSG:3857."""
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    slug = SLUG[city]
    st = json.loads((OSM / f"{slug}_metro.json").read_text(encoding="utf-8")).get("elements", [])
    pts = [Point(e.get("lon", e.get("center", {}).get("lon")), e.get("lat", e.get("center", {}).get("lat")))
           for e in st if e.get("lat", e.get("center", {}).get("lat")) is not None]
    names = [e.get("tags", {}).get("name", "") for e in st if e.get("lat", e.get("center", {}).get("lat")) is not None]
    keep = [not re.match(r"^(Ga\s*)?[SC]\d", n) for n in names]  # planned stations are named by code (D18)
    stations = gpd.GeoDataFrame(geometry=[q for q, k in zip(pts, keep) if k], crs=4326).to_crs(3857)
    f = OSM / f"{slug}_metro_lines.json"
    lines = gpd.GeoDataFrame(geometry=[], crs=3857)
    if f.exists():
        ways = []
        for e in json.loads(f.read_text(encoding="utf-8")).get("elements", []):
            name = e.get("tags", {}).get("name", "")
            if "Depot" in name or "chạy thử" in name or len(e.get("geometry", [])) < 2:
                continue
            ways.append(LineString([(p["lon"], p["lat"]) for p in e["geometry"]]))
        lines = gpd.GeoDataFrame(geometry=ways, crs=4326).to_crs(3857)
        near = stations.buffer(1200).union_all()
        lines = lines[lines.intersects(near)]
    return lines, stations


def segments_maps():
    import geopandas as gpd
    from matplotlib.patches import Patch
    t = pd.read_csv(TAB / "district_segments.csv")
    colors = pd.read_csv(TAB / "segment_colors.csv", index_col=0)["color"].to_dict()
    df = pd.read_parquet(PROCESSED / "listings_model.parquet", columns=["city", "district", "lat", "lon"])
    counts = df.groupby(["city", "district"]).size().rename("n_all").reset_index()
    g = boundaries()
    g["city"] = g["NAME_1"].map({v: k for k, v in CITY_GADM.items()})
    g["k"] = g["NAME_2"].map(key)
    counts["k"] = counts["district"].map(key)
    t["k"] = t["district"].map(key)
    unmatched = sorted(set(zip(counts["city"], counts["district"])) -
                       set(zip(counts.merge(g[["city", "k"]], on=["city", "k"])["city"],
                               counts.merge(g[["city", "k"]], on=["city", "k"])["district"])))
    log.info(f"listing districts without a boundary: {unmatched if unmatched else 'none'}")
    seg_unmatched = sorted(set(t["district"]) - set(t.merge(g[["city", "k"]], on=["city", "k"])["district"]))
    log.info(f"segment districts without a boundary: {seg_unmatched if seg_unmatched else 'none'}")
    pd.DataFrame({"unmatched": unmatched + seg_unmatched}).to_csv(TAB / "map_unmatched_districts.csv", index=False)
    g = g.merge(t[["city", "k", "segment", "n", "district"]], on=["city", "k"], how="left")
    g = g.merge(counts[["city", "k", "n_all"]], on=["city", "k"], how="left")

    fig, axes = plt.subplots(1, 2, figsize=(W2, 9.4 / 2.54))
    to_label = []
    for ax, city in zip(axes, ["Hà Nội", "TP. Hồ Chí Minh"]):
        gc = g[g["city"] == city]
        pts = df[df["city"] == city]
        p = gpd.GeoSeries(gpd.points_from_xy(pts["lon"], pts["lat"]), crs=4326).to_crs(3857)
        xs, ys = np.percentile(p.x, [2, 98]), np.percentile(p.y, [2, 98])
        span = max(xs[1] - xs[0], ys[1] - ys[0]) * 1.2
        cx, cy = xs.mean(), ys.mean()
        x0, x1, y0, y1 = cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2
        fill = gc["segment"].map(colors).fillna("#EDEDED")
        gc.plot(ax=ax, color=fill, edgecolor="white", linewidth=0.6)
        gc.dissolve().boundary.plot(ax=ax, color=MGREY, linewidth=0.9)
        lines, stations = metro(city)
        if len(lines):
            lines.plot(ax=ax, color=DGREY, linewidth=1.3, zorder=4)
        stations.plot(ax=ax, color="white", edgecolor=DGREY, markersize=7, linewidth=0.6, zorder=5)
        lab = gc.dropna(subset=["n"])
        lab = pd.concat([lab.nlargest(6, "n"), lab[lab["segment"] == "Prime core"]]).drop_duplicates("district")
        to_label.append((ax, lab, (x0, x1, y0, y1)))
        hidden = gc.dropna(subset=["segment"])
        hidden = hidden[~hidden.representative_point().apply(lambda q: x0 < q.x < x1 and y0 < q.y < y1)]
        if len(hidden):
            log.info(f"{city}: segment districts outside the crop: {list(hidden['district'])}")
        # scale bar (5 km) in the lower left
        km = 5000 / np.cos(np.radians(21.0 if city == "Hà Nội" else 10.8))
        bx, by = x0 + 0.05 * span, y0 + 0.05 * span
        ax.plot([bx, bx + km], [by, by], color=DGREY, lw=1.6, solid_capstyle="butt", zorder=6)
        ax.text(bx + km / 2, by + 0.02 * span, "5 km", ha="center", va="bottom", fontsize=6.3, color=DGREY, zorder=6)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title("Hanoi" if city == "Hà Nội" else "Ho Chi Minh City", loc="left")
    handles = [Patch(facecolor=c, edgecolor="none", label=s) for s, c in colors.items()]
    handles.append(Patch(facecolor="#EDEDED", edgecolor="none", label="Too few listings (< 30)"))
    handles.append(plt.Line2D([], [], color=DGREY, lw=1.3, marker="o", mfc="white", mec=DGREY, ms=3.5,
                              label="Operating metro line and station"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=6.5, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.4, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6)
    # district labels: collision-free placement with leader lines (labels.py)
    from labels import place_labels
    for ax, lab, (x0, x1, y0, y1) in to_label:
        items = []
        for _, r in lab.sort_values("n", ascending=False).iterrows():
            c = r.geometry.centroid
            pt = c if r.geometry.contains(c) else r.geometry.representative_point()
            if not (x0 < pt.x < x1 and y0 < pt.y < y1):
                continue
            name = r["district"].replace("Quận ", "Q. ").replace("Huyện ", "").replace(" (TP. Thủ Đức)", "")
            items.append((pt.x, pt.y, name, 2.0))
        ax.scatter([i[0] for i in items], [i[1] for i in items], s=4, color=DGREY, zorder=6)
        place_labels(fig, ax, items, [(i[0], i[1], 2.0) for i in items], fontsize=6.3)
    savefig(fig, "segments_maps")


def segments_profile():
    t = pd.read_csv(TAB / "district_segments.csv")
    sel = pd.read_csv(TAB / "segment_k_selection.csv")
    colors = pd.read_csv(TAB / "segment_colors.csv", index_col=0)["color"].to_dict()
    z = (t[FEATS] - t[FEATS].mean()) / t[FEATS].std(ddof=0)
    z = z.clip(-2.5, 2.5)
    z["segment"] = t["segment"]
    pz = z.groupby("segment").mean().loc[list(colors)]
    k_best = t["cluster"].nunique()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 5.0 / 2.54), gridspec_kw={"width_ratios": [1, 2.6]})
    a1.plot(sel["k"], sel["silhouette"], "o-", color=MGREY, ms=3, lw=1)
    r = sel.set_index("k").loc[k_best]
    a1.plot(k_best, r["silhouette"], "o", color=ORANGE, ms=5)
    a1.annotate(f"k = {k_best}", (k_best, r["silhouette"]), xytext=(4, 6), textcoords="offset points",
                fontsize=7, color=ORANGE)
    a1.set_xlabel("Number of segments")
    a1.set_ylabel("Separation score")
    a1.set_title("How many segments")
    a2.imshow(pz.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    a2.set_xticks(range(pz.shape[1]), [PLAIN[c] for c in pz.columns], fontsize=6.8)
    a2.xaxis.tick_top()
    a2.tick_params(axis="x", length=0, pad=2)
    a2.tick_params(axis="y", length=0)
    a2.set_yticks(range(pz.shape[0]), pz.index, fontsize=7)
    for i in range(pz.shape[0]):
        for j in range(pz.shape[1]):
            v = pz.iloc[i, j]
            a2.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6.8,
                    color="white" if abs(v) > 1.1 else DGREY)
    a2.set_xlabel("Standardised segment means (0 = average district; red = above, blue = below)",
                  fontsize=6.8, color=MGREY)
    for s in a2.spines.values():
        s.set_visible(False)
    fig.tight_layout(w_pad=1.2)
    savefig(fig, "segments_profile")


def waterfall():
    w = pd.read_csv(TAB / "cleaning_waterfall.csv")
    w = w[(w.index == 0) | (w["n_dropped"] > 0)].reset_index(drop=True)
    plain = {"area outside type bounds": "Floor or land area out of range",
             "bedroom count above 15": "More than 15 bedrooms",
             "rental listings posted as sales": "Rentals posted as sales",
             "implausible price": "Placeholder or mistyped price",
             "price per m2 outside": "Extreme price per m² (top, bottom 1%)",
             "near-duplicate": "Same home posted by several brokers"}

    def name(s):
        for k, v in plain.items():
            if k in s:
                return v
        return s.replace("Remove ", "")
    labels = ["Listings collected"] + [name(s) for s in w["step"][1:]] + ["Clean sample"]
    vals = list(w["n_remaining"])
    n = len(labels)
    fig, ax = plt.subplots(figsize=(8.6 / 2.54, 5.6 / 2.54))
    y = np.arange(n)[::-1]
    lo = 5000  # axis break: totals start at the break, drops are drawn to scale
    x0 = lo - 120
    ax.barh(y[0], vals[0] - x0, left=x0, color=MGREY, height=0.62)
    ax.text((vals[0] + lo) / 2, y[0], f"{vals[0]:,}", ha="center", va="center", fontsize=7, color="white",
            fontweight="bold")
    for i in range(1, len(vals)):
        drop = vals[i - 1] - vals[i]
        ax.barh(y[i], drop, left=vals[i], color=ORANGE, height=0.62)
        ax.plot([vals[i], vals[i]], [y[i] - 0.31, y[i] + 0.69], color=LGREY, lw=0.5, zorder=0)
        ax.text(vals[i] - 25, y[i], f"−{drop:,}", ha="right", va="center", fontsize=6.6, color=ORANGE)
    ax.barh(y[-1], vals[-1] - x0, left=x0, color=BLUE, height=0.62)
    ax.text(vals[-1] + 25, y[-1], f"{vals[-1]:,} clean ({vals[-1] / vals[0]:.0%})", ha="left", va="center",
            fontsize=7, color=BLUE, fontweight="bold")
    # break marks across the two total bars and the axis
    for yy in (y[0], y[-1], y[-1] - 0.75):
        ax.plot([lo - 70, lo - 40], [yy - 0.38, yy + 0.38], color="white" if yy != y[-1] - 0.75 else MGREY,
                lw=1.6, clip_on=False, zorder=5)
        ax.plot([lo - 40, lo - 10], [yy - 0.38, yy + 0.38], color="white" if yy != y[-1] - 0.75 else MGREY,
                lw=1.6, clip_on=False, zorder=5)
    ax.set_yticks(y, labels, fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(x0, vals[0] * 1.01)
    ax.set_ylim(y[-1] - 0.75, y[0] + 0.5)
    ax.set_xticks([lo, 5500, 6000])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Listings (axis broken below 5,000)", color=DGREY)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    savefig(fig, "cleaning_waterfall")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    waterfall()
    segments_profile()
    segments_maps()
