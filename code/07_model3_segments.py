"""Module 3: neighbourhood segmentation (district as listed x city).

Districts with at least 30 listings. Standardised features: type-adjusted price level, median
listing age (liquidity, creation-based), median accessibility index, share of houses and land,
median out-of-fold AVM residual, log listing volume. K-Means k = 2..8, elbow + silhouette;
robustness with Ward hierarchical clustering (adjusted Rand index).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (BLUE, CLUSTER_COLORS, DGREY, FIG, LGREY, MGREY, MODELS, ORANGE, OSM,  # noqa: E402
                    PROCESSED, SEED, TAB, W1, W2, get_logger, savefig, set_style)
from features import add_log_dist  # noqa: E402

log = get_logger("07_model3_segments")
set_style()
MIN_N = 30
FEATS = {
    "ln_price_index": "Price level (type-adjusted, log)",
    "ln_med_age": "Median listing age (log)",
    "med_access": "Median accessibility index",
    "share_house_land": "Share houses + land",
    "med_resid": "Median AVM gap",
    "ln_volume": "Listing volume (log)",
}


def district_table(df):
    # type-adjusted price: listing ppm2 relative to its city x type median
    base = df.groupby(["city", "property_type"])["price_per_m2"].transform("median")
    df = df.assign(rel=df["price_per_m2"] / base,
                   house_land=df["property_type"].isin(["Nhà hẻm/ngõ", "Nhà mặt tiền/phố", "Đất"]).astype(int))
    g = df.groupby(["city", "district", "district_key"])
    t = g.agg(n=("listing_id", "size"), med_ppm2_mil=("ppm2_mil", "median"),
              price_index=("rel", "median"), med_age=("age_id_days", "median"),
              med_age_posted=("listing_age_days", "median"), med_access=("accessibility_index", "median"),
              share_house_land=("house_land", "mean"), med_resid=("avm_resid", "median"),
              share_high_risk=("high_risk", "mean"), lat=("lat", "median"), lon=("lon", "median"),
              med_dist_cbd=("dist_cbd_km", "median")).reset_index()
    t["ln_price_index"] = np.log(t["price_index"])
    t["ln_volume"] = np.log(t["n"])
    t["ln_med_age"] = np.log(t["med_age"])
    return t


def name_clusters(prof):
    """Name clusters from their standardised profile, relative to the other clusters."""
    names = {}
    rest = list(prof.index)
    top = prof["ln_price_index"].idxmax()
    if prof.loc[top, "ln_price_index"] > 0.8:
        names[top] = "Prime core"
        rest.remove(top)
    low = prof.loc[rest, "med_access"].idxmin()
    if prof.loc[low, "med_access"] < -0.8:
        names[low] = "Peripheral growth"
        rest.remove(low)
    for c in rest:
        r = prof.loc[c]
        base = ("Townhouse mid-market" if r["share_house_land"] > 0.3 else
                "Apartment mid-market" if r["share_house_land"] < -0.3 else "Mixed mid-market")
        names[c] = base
    seen = {}
    for c in sorted(names, key=lambda k: prof.loc[k, "ln_med_age"]):
        seen[names[c]] = seen.get(names[c], 0) + 1
        if seen[names[c]] > 1:
            names[c] += " (slower)"
    return names


def main():
    df = add_log_dist(pd.read_parquet(PROCESSED / "listings_model.parquet"))
    avm = pd.read_csv(MODELS / "avm_predictions.csv")[["listing_id", "avm_resid"]]
    sc = pd.read_csv(MODELS / "overpricing_scores.csv")[["listing_id", "p_stale", "high_risk"]]
    df = df.merge(avm, on="listing_id", how="left").merge(sc, on="listing_id", how="left")
    t_all = district_table(df)
    t = t_all[t_all["n"] >= MIN_N].reset_index(drop=True)
    log.info(f"districts: {len(t_all)} in total, {len(t)} with >= {MIN_N} listings kept "
             f"({t['n'].sum()} listings, {t['n'].sum() / len(df):.1%})")
    # standardise, then winsorise at +/-2.5 SD so one extreme district cannot form its own cluster
    Z = np.clip(StandardScaler().fit_transform(t[list(FEATS)]), -2.5, 2.5)

    rows = []
    for k in range(2, 9):
        km = KMeans(k, n_init=50, random_state=SEED).fit(Z)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Z, km.labels_)})
    sel = pd.DataFrame(rows)
    sel.to_csv(TAB / "segment_k_selection.csv", index=False)
    # choose k by silhouette, preferring 3 to 5 interpretable clusters
    pref = sel[sel["k"].between(3, 5)]
    k_best = int(pref.loc[pref["silhouette"].idxmax(), "k"])
    k_overall = int(sel.loc[sel["silhouette"].idxmax(), "k"])
    log.info("k selection:\n" + sel.round(3).to_string(index=False))
    log.info(f"chosen k={k_best} (best silhouette within 3-5; overall best k={k_overall})")
    km = KMeans(k_best, n_init=50, random_state=SEED).fit(Z)
    t["cluster"] = km.labels_
    ward = AgglomerativeClustering(n_clusters=k_best, linkage="ward").fit(Z)
    ari = adjusted_rand_score(km.labels_, ward.labels_)
    log.info(f"Ward vs K-Means adjusted Rand index: {ari:.3f}")

    zdf = pd.DataFrame(Z, columns=list(FEATS))
    zdf["cluster"] = km.labels_
    prof_z = zdf.groupby("cluster").mean()
    names = name_clusters(prof_z)
    t["segment"] = t["cluster"].map(names)
    prof = t.groupby("segment").agg(n_districts=("district", "size"), n_listings=("n", "sum"),
                                    med_ppm2_mil=("med_ppm2_mil", "median"), price_index=("price_index", "median"),
                                    med_age=("med_age", "median"), med_access=("med_access", "median"),
                                    share_house_land=("share_house_land", "median"),
                                    med_resid=("med_resid", "median"), share_high_risk=("share_high_risk", "mean"),
                                    med_dist_cbd=("med_dist_cbd", "median"),
                                    hanoi_districts=("city", lambda s: (s == "Hà Nội").sum()))
    prof = prof.sort_values("price_index", ascending=False)
    prof.to_csv(TAB / "segment_profiles.csv", encoding="utf-8-sig")
    t.to_csv(TAB / "district_segments.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"k": k_best, "k_best_overall": k_overall,
                   "silhouette": float(sel.set_index("k").loc[k_best, "silhouette"]),
                   "ari_ward": ari, "n_districts_total": len(t_all), "n_districts_kept": len(t),
                   "listings_covered": int(t["n"].sum())}]).to_csv(TAB / "segment_summary.csv", index=False)
    log.info("profiles:\n" + prof.round(2).to_string())

    order = list(prof.index)
    colors = {s: CLUSTER_COLORS[i] for i, s in enumerate(order)}
    pd.Series(colors).to_csv(TAB / "segment_colors.csv", header=["color"])
    figures(sel, k_best, prof_z.rename(index=names).loc[order], t, colors, df)
    folium_maps(t, colors)


def figures(sel, k_best, pz, t, colors, df):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 5.6 / 2.54), gridspec_kw={"width_ratios": [1, 2.1]})
    a1.plot(sel["k"], sel["silhouette"], "o-", color=MGREY, ms=3, lw=1)
    r = sel.set_index("k").loc[k_best]
    a1.plot(k_best, r["silhouette"], "o", color=ORANGE, ms=5)
    a1.annotate(f"k = {k_best}", (k_best, r["silhouette"]), xytext=(4, 6), textcoords="offset points",
                fontsize=7, color=ORANGE)
    a1.set_xlabel("Number of clusters k")
    a1.set_ylabel("Silhouette score")
    a1.set_title("Cluster count")
    im = a2.imshow(pz.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    a2.set_xticks(range(pz.shape[1]), [FEATS[c] for c in pz.columns], rotation=30, ha="right", fontsize=6.8)
    a2.set_yticks(range(pz.shape[0]), pz.index, fontsize=7)
    for i in range(pz.shape[0]):
        for j in range(pz.shape[1]):
            v = pz.iloc[i, j]
            a2.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6.8,
                    color="white" if abs(v) > 1.1 else DGREY)
    a2.set_title("Segment profiles (standardised, 0 = average district)")
    for s in a2.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    savefig(fig, "segments_profile")

    # maps
    # Basemap tiles are not used: OSM's tile servers block contextily requests (HTTP 403 under the
    # tile usage policy) and CARTO tiles now need an API key (logged in DECISIONS.md). The maps use a
    # clean scatter with metro stations; the interactive folium maps load tiles in the browser.
    has_cx = False
    fig, axes = plt.subplots(1, 2, figsize=(W2, 8.2 / 2.54))
    for ax, city in zip(axes, ["Hà Nội", "TP. Hồ Chí Minh"]):
        tc = t[t["city"] == city]
        lst = df[df["district_key"].isin(tc["district_key"])].merge(tc[["district_key", "segment"]], on="district_key")
        allc = df[df["city"] == city]
        gx, gy = to_merc(allc["lon"].to_numpy(), allc["lat"].to_numpy())
        ax.scatter(gx, gy, s=0.8, color=LGREY, alpha=0.5, lw=0)
        x, y = to_merc(lst["lon"].to_numpy(), lst["lat"].to_numpy())
        ax.scatter(x, y, s=1.2, c=lst["segment"].map(colors), alpha=0.45, lw=0)
        cx_, cy_ = to_merc(tc["lon"].to_numpy(), tc["lat"].to_numpy())
        ax.scatter(cx_, cy_, s=18 + tc["n"] / 6, c=tc["segment"].map(colors), edgecolor="white", lw=0.6, zorder=3)
        metro = load_metro(city)
        if len(metro):
            mx, my = to_merc(metro["lon"].to_numpy(), metro["lat"].to_numpy())
            ax.scatter(mx, my, marker="s", s=5, color=DGREY, zorder=4)
        # frame on the listings' 2nd-98th percentile extent
        xs, ys = np.percentile(x, [2, 98]), np.percentile(y, [2, 98])
        pad = 0.08 * max(xs[1] - xs[0], ys[1] - ys[0])
        ax.set_xlim(xs[0] - pad, xs[1] + pad)
        ax.set_ylim(ys[0] - pad, ys[1] + pad)
        ax.set_aspect("equal")
        ax.set_anchor("N")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("Hanoi" if city == "Hà Nội" else "Ho Chi Minh City")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=5, label=s) for s, c in colors.items()]
    handles.append(plt.Line2D([], [], marker="s", ls="", color=DGREY, ms=3.5, label="Metro / urban rail station"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    savefig(fig, "segments_maps")
    pd.DataFrame([{"basemap": has_cx}]).to_csv(TAB / "segments_map_meta.csv", index=False)

    # executive matrix: liquidity vs price level
    fig, ax = plt.subplots(figsize=(W2, 8.4 / 2.54))
    bubbles = []
    for s, g in t.groupby("segment"):
        bubbles.append(ax.scatter(g["med_age"], g["med_ppm2_mil"], s=g["n"] / 2.2, color=colors[s], alpha=0.75,
                                  edgecolor="white", lw=0.5, label=s))
    lab = pd.concat([t.nlargest(5, "med_ppm2_mil"), t.nlargest(3, "med_age"), t.nlargest(3, "n"),
                     t.nsmallest(2, "med_ppm2_mil")]).drop_duplicates("district_key").sort_values("med_ppm2_mil",
                                                                                              ascending=False)
    ax.axvline(t["med_age"].median(), color=LGREY, lw=0.6, ls="--")
    ax.axhline(t["med_ppm2_mil"].median(), color=LGREY, lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}" if str(int(v))[0] in "25" else ""))
    ax.set_xlabel("Median listing age since creation, days, log scale (higher = slower market)")
    ax.set_ylabel("Median asking price (million VND/m²)")
    hd = [plt.Line2D([], [], marker="o", ls="", color=colors[s_], ms=6, label=s_) for s_ in colors]
    ax.legend(handles=hd, fontsize=7, loc="upper right")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}" if str(int(v))[0] in "25" else ""))
    fig.tight_layout()
    # labels placed in display space so they never cover a bubble, another label or the legend
    from labels import place_labels
    leg = ax.get_legend()
    rad = lambda n: np.sqrt(n / 2.2) / 2 * fig.dpi / 72  # scatter size is an area in points^2
    circles = [(r["med_age"], r["med_ppm2_mil"], rad(r["n"])) for _, r in t.iterrows()]
    items = []
    for _, r in lab.iterrows():
        short = r["district"].replace("Quận ", "Q. ").replace("Huyện ", "H. ").replace(" (TP. Thủ Đức)", "")
        items.append((r["med_age"], r["med_ppm2_mil"], f"{short} ({'HN' if r['city'] == 'Hà Nội' else 'HCM'})",
                      rad(r["n"])))
    place_labels(fig, ax, items, circles, avoid=[leg] if leg else [])
    fig.tight_layout()
    savefig(fig, "executive_matrix")


def to_merc(lon, lat):
    x = np.radians(lon) * 6378137.0
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * 6378137.0
    return x, y


def load_metro(city):
    import json
    f = OSM / f"{'hanoi' if city == 'Hà Nội' else 'hcmc'}_metro.json"
    els = json.loads(f.read_text(encoding="utf-8")).get("elements", [])
    pts = [(e.get("lat", e.get("center", {}).get("lat")), e.get("lon", e.get("center", {}).get("lon"))) for e in els]
    return pd.DataFrame([p for p in pts if p[0] is not None], columns=["lat", "lon"])


def folium_maps(t, colors):
    import folium
    for city, slug in (("Hà Nội", "hanoi"), ("TP. Hồ Chí Minh", "hcmc")):
        tc = t[t["city"] == city]
        m = folium.Map(location=[tc["lat"].median(), tc["lon"].median()], zoom_start=11, tiles="OpenStreetMap")
        for _, r in tc.iterrows():
            folium.CircleMarker(
                [r["lat"], r["lon"]], radius=5 + np.sqrt(r["n"]) / 2, color=colors[r["segment"]], fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(f"<b>{r['district']}</b><br>Segment: {r['segment']}<br>Listings: {r['n']}<br>"
                                   f"Median price: {r['med_ppm2_mil']:.0f} million VND/m²<br>"
                                   f"Median listing age: {r['med_age']:.0f} days", max_width=260)).add_to(m)
        m.save(str(FIG / f"segments_map_{slug}.html"))
    _ = (BLUE, W1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
