"""Standalone HTML dashboard: export the trained models and lookups, test JS parity, build one HTML file.

No model is re-trained. Steps:
1. Dump both XGBoost boosters (get_dump json) into compact flat arrays. First attempt: thresholds and leaves
   rounded to 6 significant digits. If that fails parity, both are written as the shortest decimal that
   reproduces their float32 value (logged in result/logs/js_parity.txt).
2. Embed the lookups the Streamlit app uses: ward and district medians of the location features, district
   lookup for the stale model, median prices by district x type and city x type, conformal widths, segment
   table, a slim listing table for comparables, simplified district polygons and metro lines.
3. Parity: 500 random listings scored by the Python app logic and by the JS engine under node;
   max |diff| < 1e-4 on log price and on probability. Report: result/logs/js_parity.txt.
4. Write dashboard/vietnam_avm_dashboard.html (data and engine inlined; Leaflet from cdnjs).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CBD, MODELS, OSM, PROCESSED, ROOT, SEED, TAB, TYPE_EN, get_logger  # noqa: E402
from features import add_log_dist, tree_matrix  # noqa: E402

log = get_logger("17_export_dashboard")
OUTDIR = ROOT / "dashboard"
OUTDIR.mkdir(parents=True, exist_ok=True)
LOGS = ROOT / "result" / "logs"
LOC_COLS = ["lat", "lon", "accessibility_index", "n_bus_stop_500m", "n_school_1000m", "n_market_1000m"] + \
           [f"dist_{l}_km" for l in ["metro", "bus_stop", "school", "university", "hospital", "park", "market",
                                     "water", "cbd"]]
LEGAL = ["Sổ đỏ/Sổ hồng", "Hợp đồng mua bán", "Đang chờ sổ", "Khác/không rõ"]
LEGAL_EN = {"Sổ đỏ/Sổ hồng": "Red or pink book", "Hợp đồng mua bán": "Sale contract",
            "Đang chờ sổ": "Certificate pending", "Khác/không rõ": "Other or unknown"}


def sig(x, n=6):
    return float(f"{x:.{n}g}")


def f32_short(x):
    """Shortest decimal that parses back to the same float32."""
    v = np.float32(x)
    for d in range(6, 10):
        s = float(f"{float(v):.{d}g}")
        if np.float32(s) == v:
            return s
    return float(v)


def dump_trees(booster, cols, thr_mode):
    idx = {c: i for i, c in enumerate(cols)}
    trees, changed = [], 0
    for js in booster.get_dump(dump_format="json"):
        root = json.loads(js)
        nodes = {}

        def walk(n):
            nodes[n["nodeid"]] = n
            for c in n.get("children", []):
                walk(c)
        walk(root)
        flat = []
        for k in range(max(nodes) + 1):
            n = nodes.get(k)
            if n is None:
                raise ValueError("non-contiguous node ids")
            if "leaf" in n:
                flat += [-1, sig(n["leaf"]) if thr_mode in ("sig6", "mixed") else f32_short(n["leaf"])]
            else:
                assert n["no"] == n["yes"] + 1 and n["missing"] in (n["yes"], n["no"])
                t = n["split_condition"]
                t_out = sig(t) if thr_mode == "sig6" else f32_short(t)
                changed += np.float32(t_out) != np.float32(t)
                flat += [idx[n["split"]], t_out, n["yes"], 1 if n["missing"] == n["yes"] else 0]
        trees.append(flat)
    return trees, int(changed)


def base_score(booster):
    cfg = json.loads(booster.save_config())
    raw = cfg["learner"]["learner_model_param"]["base_score"]
    return float(re.sub(r"[\[\]]", "", str(raw))), cfg["learner"]["objective"]["name"]


def model_spec(d, thr_mode):
    b = d["model"].get_booster()
    base, obj = base_score(b)
    trees, changed = dump_trees(b, d["columns"], thr_mode)
    if obj == "binary:logistic":
        base = float(np.log(base / (1 - base)))
    return {"cols": list(d["columns"]), "base": base, "objective": obj, "trees": trees}, changed


def load():
    df = add_log_dist(pd.read_parquet(PROCESSED / "listings_model.parquet"))
    avm = joblib.load(MODELS / "xgb_avm.joblib")
    conf = joblib.load(MODELS / "conformal.joblib")
    stale = joblib.load(MODELS / "xgb_stale.joblib")
    seg = pd.read_csv(TAB / "district_segments.csv")
    lookup = pd.read_csv(MODELS / "district_lookup.csv").set_index("district_key")
    scores = pd.read_csv(MODELS / "overpricing_scores.csv")
    return df, avm, conf, stale, seg, lookup, scores


def app_value(inp, df, avm, conf, stale, lookup, scores):
    """Python reference: the Streamlit app's valuation logic, unchanged (code/app.py)."""
    city, district, ward, ptype = inp["city"], inp["district"], inp["ward"], inp["type"]
    d = df[(df["city"] == city) & (df["district"] == district)]
    w = d[d["ward"] == ward] if ward else d
    src = w if len(w) >= 3 else d
    loc = src[LOC_COLS].median()
    area, beds, baths = inp["area"], inp["bedrooms"], inp["bathrooms"]
    row = pd.DataFrame([{**loc.to_dict(), "city": city, "district": district, "ward": ward,
                         "property_type": ptype, "legal_group": inp["legal"], "ln_area": np.log(area),
                         "area_m2": area, "bedrooms": beds, "bathrooms": baths, "floors": inp["floors"],
                         "frontage_m": inp["frontage"], "road_width_m": inp["road"], "floors_missing": 0,
                         "frontage_m_missing": 0, "road_width_m_missing": 0, "bedrooms_missing": 0,
                         "bathrooms_missing": 0, "kw_car_access": 0, "kw_corner": 0, "kw_business": 0,
                         "kw_lake_view": 0, "kw_near_school": 0, "kw_furnished": 0}])
    row = add_log_dist(row)
    pred = float(avm["model"].predict(tree_matrix(row, avm["columns"]))[0])
    q = conf["q_type"].get(ptype, conf["q_global"])
    out = {"pred": pred, "fv": np.exp(pred) / 1e9, "lo": np.exp(pred - q) / 1e9, "hi": np.exp(pred + q) / 1e9}
    ask = inp["ask"]
    if ask > 0:
        gap = np.log(ask * 1e9) - pred
        dk = row["district_key"].iloc[0]
        lk = lookup.loc[dk] if dk in lookup.index else lookup.median(numeric_only=True)
        dt = df[(df["district_key"] == dk) & (df["property_type"] == ptype)]["price_vnd"].median()
        if np.isnan(dt):
            dt = df[(df["city"] == city) & (df["property_type"] == ptype)]["price_vnd"].median()
        m2 = pd.DataFrame([{
            "avm_resid": gap, "rel_ppm2_district": ask * 1e9 / area / lk["med_ppm2"],
            "rel_price_district_type": ask * 1e9 / dt, "district_liquidity": lk["district_liquidity"],
            "ln_district_volume": lk["ln_district_volume"], "ln_area": np.log(area), "bedrooms": beds,
            "n_images": df["n_images"].median(), "ln_desc_len": df["ln_desc_len"].median(),
            "accessibility_index": loc["accessibility_index"], "broker_profile": 1,
            "kw_car_access": 0, "kw_corner": 0, "kw_business": 0, "kw_lake_view": 0, "kw_near_school": 0,
            "kw_owner": 0, "kw_urgent": 0, "kw_furnished": 0, "property_type": ptype, "city": city,
            "legal_group": inp["legal"]}])
        Xs = pd.get_dummies(m2, columns=["property_type", "city", "legal_group"], dtype=float) \
            .reindex(columns=stale["columns"], fill_value=0.0)
        p = float(stale["model"].predict_proba(Xs)[0, 1])
        t, green = stale["high_risk_cutoff"], scores["p_stale"].quantile(0.5)
        out.update(gap=float(np.exp(gap) - 1), p=p, light="High" if p >= t else ("Medium" if p >= green else "Low"))
    pool = df[(df["city"] == city) & (df["property_type"] == ptype)].copy()
    z = pool[["ln_area", "bedrooms", "bathrooms", "lat", "lon"]]
    sd = z.std().replace(0, 1)
    target = pd.Series({"ln_area": np.log(area), "bedrooms": beds, "bathrooms": baths,
                        "lat": loc["lat"], "lon": loc["lon"]})
    wts = pd.Series({"ln_area": 1.0, "bedrooms": 0.5, "bathrooms": 0.3, "lat": 1.5, "lon": 1.5})
    pool["dist"] = np.sqrt((((z - target) / sd) ** 2 * wts).sum(axis=1))
    out["comps"] = [int(i) for i in pool.nsmallest(5, "dist").index]
    return out


def geodata(seg):
    """Simplified district polygons (GADM 4.1) with segment, metro lines and stations, as GeoJSON."""
    import importlib
    rf = importlib.import_module("14_report_figures")
    g = rf.boundaries().to_crs(4326)
    g["city"] = g["NAME_1"].map({v: k for k, v in rf.CITY_GADM.items()})
    g["k"] = g["NAME_2"].map(rf.key)
    s = seg.copy()
    s["k"] = s["district"].map(rf.key)
    df = pd.read_parquet(PROCESSED / "listings_model.parquet", columns=["city", "district"])
    names = df.drop_duplicates().assign(k=lambda x: x["district"].map(rf.key))
    g = g.merge(s[["city", "k", "segment", "n", "med_ppm2_mil", "med_age"]], on=["city", "k"], how="left") \
        .merge(names, on=["city", "k"], how="left")
    feats = []
    for _, r in g.iterrows():
        geom = r.geometry.simplify(0.0008, preserve_topology=True)
        gj = json.loads(json.dumps(geom.__geo_interface__))

        def rnd(c):
            return [rnd(x) for x in c] if isinstance(c[0], (list, tuple)) else [round(c[0], 4), round(c[1], 4)]
        gj["coordinates"] = rnd(gj["coordinates"])
        label = r["district"] if isinstance(r["district"], str) else r["NAME_2"]
        feats.append({"type": "Feature", "geometry": gj,
                      "properties": {"city": r["city"], "name": label,
                                     "segment": r["segment"] if isinstance(r["segment"], str) else None,
                                     "n": None if pd.isna(r["n"]) else int(r["n"]),
                                     "ppm2": None if pd.isna(r["med_ppm2_mil"]) else round(float(r["med_ppm2_mil"])),
                                     "age": None if pd.isna(r["med_age"]) else round(float(r["med_age"]))}})
    metro = {}
    for city in ("Hà Nội", "TP. Hồ Chí Minh"):
        lines, stations = rf.metro(city)
        lines, stations = lines.to_crs(4326), stations.to_crs(4326)
        metro[city] = {
            "lines": [[[round(y, 5), round(x, 5)] for x, y in ln.simplify(0.0002).coords] for ln in lines.geometry],
            "stations": [[round(p.y, 5), round(p.x, 5)] for p in stations.geometry]}
    return {"type": "FeatureCollection", "features": feats}, metro


def payload(df, avm, conf, stale, seg, lookup, scores, thr_mode):
    A, ca = model_spec(avm, thr_mode)
    S, cs = model_spec(stale, thr_mode)
    loc = {}
    for (city, district), d in df.groupby(["city", "district"]):
        wards = {}
        for ward, w in d.groupby("ward"):
            if ward and len(w) >= 3:
                wards[ward] = {c: float(v) for c, v in w[LOC_COLS].median().items()}
        loc[f"{city}|{district}"] = {"district": {c: float(v) for c, v in d[LOC_COLS].median().items()},
                                     "wards": wards,
                                     "wardList": sorted(x for x in d["ward"].unique() if x),
                                     "n": int(len(d))}
    lk = {k: {c: float(r[c]) for c in ("med_ppm2", "district_liquidity", "ln_district_volume")}
          for k, r in lookup.iterrows()}
    lkm = {c: float(v) for c, v in lookup.median(numeric_only=True).items()}
    dt = df.groupby(["district_key", "property_type"])["price_vnd"].median()
    ct = df.groupby(["city", "property_type"])["price_vnd"].median()
    # slim listing table (no URLs, no seller fields), dictionary-encoded strings
    L = df.reset_index(drop=True)
    cities, districts = sorted(L["city"].unique()), sorted(L["district"].unique())
    wardsl, types = sorted(L["ward"].fillna("").unique()), list(TYPE_EN)
    month = pd.to_datetime(L["posted_date"], format="%d/%m/%Y", errors="coerce").dt.strftime("%m/%Y").fillna("")
    listings = {
        "cities": cities, "districts": districts, "wardsDict": wardsl, "types": types,
        "city": L["city"].map(cities.index).tolist(), "district": L["district"].map(districts.index).tolist(),
        "ward": L["ward"].fillna("").map(wardsl.index).tolist(), "type": L["property_type"].map(types.index).tolist(),
        "area": L["area_m2"].round(1).tolist(), "bedrooms": L["bedrooms"].astype(float).tolist(),
        "bathrooms": L["bathrooms"].astype(float).tolist(), "price_bn": L["price_bn"].round(3).tolist(),
        "ppm2": L["ppm2_mil"].round(1).tolist(), "month": month.tolist(),
        "ln_area": L["ln_area"].tolist(), "lat": L["lat"].tolist(), "lon": L["lon"].tolist()}
    med = df.groupby("property_type")[["area_m2", "bedrooms", "bathrooms", "floors", "frontage_m",
                                        "road_width_m"]].median()
    typeMed = {t: {"area": float(r["area_m2"]), "bedrooms": int(r["bedrooms"]), "bathrooms": int(r["bathrooms"]),
                   "floors": int(r["floors"]), "frontage": float(r["frontage_m"]), "road": float(r["road_width_m"])}
               for t, r in med.iterrows()}
    colors = pd.read_csv(TAB / "segment_colors.csv", index_col=0)["color"].to_dict()
    prof = pd.read_csv(TAB / "segment_profiles.csv", encoding="utf-8-sig")
    nums = pd.read_csv(TAB / "report_numbers.csv", index_col=0, encoding="utf-8-sig")["value"].to_dict()
    segs = seg[["city", "district", "district_key", "segment", "n", "med_ppm2_mil", "med_age", "share_high_risk"]]
    geo, metro = geodata(seg)
    D = {
        "avm": A, "stale": S, "conf": {"q_type": conf["q_type"], "q_global": conf["q_global"]},
        "loc": loc, "lookup": lk, "lookupMedian": lkm,
        "dtMed": {f"{k}|{t}": float(v) for (k, t), v in dt.items()},
        "ctMed": {f"{c}|{t}": float(v) for (c, t), v in ct.items()},
        "nImagesMed": float(df["n_images"].median()), "lnDescMed": float(df["ln_desc_len"].median()),
        "staleCut": float(stale["high_risk_cutoff"]), "staleGreen": float(scores["p_stale"].quantile(0.5)),
        "staleDays": int(stale["label_days"]),
        "typeEN": TYPE_EN, "legal": LEGAL, "legalEN": LEGAL_EN, "typeMed": typeMed, "listings": listings,
        "segments": json.loads(segs.to_json(orient="records", force_ascii=False)), "segColors": colors,
        "profiles": json.loads(prof.to_json(orient="records", force_ascii=False)),
        "geo": geo, "metro": metro, "cbd": {c: list(v) for c, v in CBD.items()},
        "about": {"N": str(nums["Nclean"]), "Nraw": str(nums["Nraw"]), "date": str(nums["scrapeDate"]),
                  "poi": str(nums["NpoiTotal"]), "ppe20": str(nums["xgbPPEtwenty"]), "auc": str(nums["xgbAUC"]),
                  "lift": str(nums["topDecileLift"]),
                  "ppeType": {t: str(nums["ppeTwenty" + m]) for t, m in (
                      ("Căn hộ chung cư", "Apartment"), ("Nhà hẻm/ngõ", "AlleyHouse"),
                      ("Nhà mặt tiền/phố", "StreetFrontHouse"), ("Biệt thự/liền kề", "VillaTerraced"),
                      ("Đất", "Land"))}},
    }
    return D, ca + cs


def parity(D, df, avm, conf, stale, lookup, scores, tag):
    """Score 500 random listings with the Python app logic and the JS engine (node); compare."""
    rng = np.random.default_rng(SEED)
    rows = df.sample(500, random_state=SEED)
    cases = []
    for i, r in rows.iterrows():
        inp = {"city": r["city"], "district": r["district"], "ward": r["ward"] or "", "type": r["property_type"],
               "area": float(r["area_m2"]), "bedrooms": int(r["bedrooms"]), "bathrooms": int(r["bathrooms"]),
               "floors": float(r["floors"]), "frontage": float(r["frontage_m"]), "road": float(r["road_width_m"]),
               "legal": r["legal_group"], "ask": float(r["price_bn"])}
        cases.append(inp)
    py = [app_value(c, df, avm, conf, stale, lookup, scores) for c in cases]
    # direct evaluator check on the training design matrices as well
    X = tree_matrix(rows, avm["columns"]).to_numpy(dtype=float)
    direct_py = avm["model"].predict(tree_matrix(rows, avm["columns"])).astype(float)
    work = LOGS / "js_parity_work"
    work.mkdir(exist_ok=True)
    (work / "data.json").write_text(json.dumps(D, ensure_ascii=False), encoding="utf-8")
    (work / "cases.json").write_text(json.dumps({"cases": cases, "X": X.tolist()}, ensure_ascii=False),
                                     encoding="utf-8")
    js = (r"""
const E = require(process.argv[2]); const fs = require('fs');
const D = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const C = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));
const eng = new E.Engine(D);
const out = C.cases.map(c => { const r = eng.value(c); return {pred: r.pred, p: r.p, light: r.light,
  comps: eng.comparables(c, r.loc, 5)}; });
const direct = C.X.map(x => eng.avm.predict(x));
fs.writeFileSync(process.argv[5], JSON.stringify({out, direct}));
""")
    (work / "run.js").write_text(js, encoding="utf-8")
    subprocess.run(["node", str(work / "run.js"), str(ROOT / "code" / "dashboard_engine.js"), str(work / "data.json"),
                    str(work / "cases.json"), str(work / "js_out.json")], check=True)
    J = json.loads((work / "js_out.json").read_text(encoding="utf-8"))
    d_pred = np.abs(np.array([o["pred"] for o in J["out"]]) - np.array([p["pred"] for p in py]))
    d_p = np.abs(np.array([o["p"] for o in J["out"]]) - np.array([p["p"] for p in py]))
    d_direct = np.abs(np.array(J["direct"]) - direct_py)
    light_ok = sum(o["light"] == p["light"] for o, p in zip(J["out"], py))
    # comparables: JS returns row positions in df order; Python returns df index labels
    pos = {lab: k for k, lab in enumerate(df.index)}
    comp_ok = sum([pos[i] for i in p["comps"]] == o["comps"] for o, p in zip(J["out"], py))
    ok = d_pred.max() < 1e-4 and d_p.max() < 1e-4 and d_direct.max() < 1e-4
    lines = [f"JS parity test ({tag})", f"listings scored: {len(cases)} (random, seed {SEED})",
             f"max |log price diff| (full app chain): {d_pred.max():.2e}  mean {d_pred.mean():.2e}",
             f"max |log price diff| (design matrix):  {d_direct.max():.2e}",
             f"max |probability diff|:               {d_p.max():.2e}  mean {d_p.mean():.2e}",
             f"traffic-light label agreement:        {light_ok}/{len(cases)}",
             f"comparable listings identical:        {comp_ok}/{len(cases)}",
             f"result: {'PASS' if ok else 'FAIL'} (threshold 1e-4)"]
    return ok, lines


def build_html(D):
    tpl = (ROOT / "code" / "dashboard_template.html").read_text(encoding="utf-8")
    eng = (ROOT / "code" / "dashboard_engine.js").read_text(encoding="utf-8")
    data = json.dumps(D, ensure_ascii=False, separators=(",", ":"))
    html = tpl.replace("/*__ENGINE__*/", eng).replace("/*__DATA__*/", "window.AVM_DATA = " + data + ";")
    out = OUTDIR / "vietnam_avm_dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


def main():
    df, avm, conf, stale, seg, lookup, scores = load()
    report = []
    modes = [("sig6", "thresholds and leaves rounded to 6 significant digits"),
             ("mixed", "leaves rounded to 6 significant digits; thresholds kept at their float32 value"),
             ("f32short", "thresholds and leaves as the shortest decimal of their float32 value (6 to 9 digits)")]
    for mode, desc in modes:
        D, changed = payload(df, avm, conf, stale, seg, lookup, scores, mode)
        ok, lines = parity(D, df, avm, conf, stale, lookup, scores, desc)
        report += lines + [f"thresholds whose float32 value changed after rounding: {changed}", ""]
        log.info("\n".join(lines))
        if ok:
            report.append(f"export used: {desc}")
            break
        report.append("-> failed; trying the next, less rounded export")
        report.append("")
    out = build_html(D)
    mb = out.stat().st_size / 1e6
    report.append(f"HTML: {out.relative_to(ROOT)} ({mb:.1f} MB)")
    (LOGS / "js_parity.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    if not ok:
        sys.exit("parity test failed")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
