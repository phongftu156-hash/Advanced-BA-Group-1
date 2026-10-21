"""Phase 4.6: export every number cited in the report as a LaTeX macro.

The macros are written to result/tables/report_numbers.csv and into the blocks between
"% BEGIN NUMBERS" and "% END NUMBERS" in report/main.tex and report/slides/pitch.tex.

The report and slides use only these macros for results, so text and result files stay in sync.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CITY_SHORT, MODELS, OSM, PROCESSED, RAW, REPORT, TAB, TYPE_EN, get_logger  # noqa: E402

log = get_logger("10_export_numbers")
M = {}


def n(x):  # integer with thousands separator
    return f"{int(round(x)):,}".replace(",", "{,}")


def pct(x, d=0):
    v = f"{100 * abs(x):.{d}f}\\%"
    return ("$-$" + v) if round(100 * x, d) < 0 else v


def f(x, d=2):
    s = f"{x:,.{d}f}".replace(",", "{,}")
    return s.replace("-", "$-$") if s.startswith("-") else s


def put(name, val):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    M[name] = val


def main():
    raw = pd.read_csv(RAW / "mogi_raw.csv", encoding="utf-8-sig", usecols=["listing_id", "scraped_at"])
    # collection plan (supply per category): from the scraper state when present, else the saved table
    state_f = RAW / "scrape_state.json"
    if state_f.exists():
        st = json.loads(state_f.read_text(encoding="utf-8"))
        plan = pd.DataFrame(st["plan"]).assign(n_failed=len(st.get("failed", [])))
        plan.to_csv(TAB / "scrape_plan.csv", index=False, encoding="utf-8-sig")
    plan = pd.read_csv(TAB / "scrape_plan.csv", encoding="utf-8-sig")
    st = {"plan": plan.to_dict("records"), "failed": [None] * int(plan["n_failed"].iloc[0])}
    df = pd.read_parquet(PROCESSED / "listings_model.parquet")
    ts = pd.to_datetime(raw["scraped_at"])
    # throughput from the detail-page timestamps (first to last listing saved), which is more
    # conservative than the scraper's own active-time counter
    hours = (ts.max() - ts.min()).total_seconds() / 3600
    put("Nraw", n(len(raw)))
    put("Nclean", n(len(df)))
    put("NcleanHN", n((df["city"] == "Hà Nội").sum()))
    put("NcleanHCM", n((df["city"] == "TP. Hồ Chí Minh").sum()))
    put("NdroppedTotal", n(len(raw) - len(df)))
    put("cleanShare", pct(len(df) / len(raw)))
    put("scrapeMinutes", n(hours * 60))
    put("scrapeHours", f(hours, 1))
    put("throughput", n(len(raw) / hours))
    put("scrapeDate", ts.min().strftime("%-d %B %Y") if sys.platform != "win32" else ts.min().strftime("%#d %B %Y"))
    put("Nfailed", n(len(st.get("failed", []))))
    supply = sum(t["supply"] for t in st["plan"])  # pages x 15; categories at the 400-page probe cap are lower bounds
    put("supplyTotal", n(round(supply, -3)))
    apt_hn = [t for t in st["plan"] if t["city"] == "Hà Nội" and t["slug"] == "mua-can-ho-chung-cu"][0]
    put("supplyAptHN", n(apt_hn["supply"]))
    import subprocess
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                          str(Path(__file__).parent / "tests")], capture_output=True, text=True).stdout
    put("Ntests", n(int(re.search(r"(\d+) tests? collected", out).group(1))))
    put("fullRunHours", n(np.ceil(supply / (len(raw) / hours))))
    put("Ndistricts", n(df["district_key"].nunique() if "district_key" in df else df.groupby(["city", "district"]).ngroups))
    put("Nwards", n(df.groupby(["city", "district", "ward"]).ngroups))
    for t, en in TYPE_EN.items():
        key = re.sub(r"[^A-Za-z]", "", en.title())
        put(f"Ntype{key}", n((df["property_type"] == t).sum()))
    cs = pd.read_csv(TAB / "cleaning_stats.csv").iloc[0]
    put("dupShare", pct(cs["near_duplicate_share"], 1))
    put("priceAgree", pct(cs["price_agree_ld"], 1))
    put("sharePoint", pct(cs["share_point"]))
    put("shareWardCentroid", pct(cs["share_ward_centroid"]))
    put("shareGeocoded", pct(cs["share_geocoded_fallback"], 1))
    wf = pd.read_csv(TAB / "cleaning_waterfall.csv")
    for key, pat in (("dropArea", "area outside"), ("dropBeds", "bedroom"), ("dropRental", "rental"),
                     ("dropImplausible", "implausible"), ("dropPctile", "percentile"), ("dropDup", "near-duplicate")):
        put(key, n(wf.loc[wf["step"].str.contains(pat, case=False), "n_dropped"].sum()))

    ag = pd.read_csv(TAB / "listing_age_checks.csv").set_index("check")["value"]
    put("rhoIDposted", f(ag.iloc[0], 2))
    put("rhoIDphoto", f(ag.iloc[1], 3))
    put("shareRefreshed", pct(ag.iloc[2]))
    put("medAgePosted", n(ag.iloc[3]))
    put("medAgeCreated", n(ag.iloc[4]))
    put("medAgeID", n(df["age_id_days"].median()))

    poi = pd.read_csv(TAB / "osm_poi_counts.csv")
    put("NpoiTotal", n(poi["n_poi"].sum()))
    put("NpoiNoWater", n(poi.loc[poi["layer"] != "water", "n_poi"].sum()))
    pv = poi.pivot(index="layer", columns="city", values="n_poi")
    put("NmetroHN", n(pv.loc["metro", "Hà Nội"]))
    put("NmetroHCM", n(pv.loc["metro", "TP. Hồ Chí Minh"]))
    put("NbusStops", n(pv.loc["bus_stop"].sum()))
    put("NschoolsPOI", n(pv.loc["school"].sum()))
    acc = pd.read_csv(TAB / "accessibility_loadings.csv")
    put("accVar", pct(acc["explained_variance_ratio"].iloc[0]))
    put("accMinLoad", f(acc["loading"].min(), 2))
    put("accMaxLoad", f(acc["loading"].max(), 2))

    grad = pd.read_csv(TAB / "eda_cbd_gradient.csv").set_index("city")["elasticity_ppm2_dist_cbd"]
    put("cbdSlopeHN", f(grad["Hanoi"], 2))
    put("cbdSlopeHCM", f(grad["HCMC"], 2))
    cor = pd.read_csv(TAB / "eda_correlations.csv")
    cc = cor[cor["type"].isna()].set_index(["city", "variable"])["spearman_rho"]
    put("rhoCbdHN", f(cc[("Hanoi", "dist_cbd_km")], 2))
    put("rhoCbdHCM", f(cc[("HCMC", "dist_cbd_km")], 2))
    put("rhoMetroHN", f(cc[("Hanoi", "dist_metro_km")], 2))
    put("rhoMetroHCM", f(cc[("HCMC", "dist_metro_km")], 2))
    ds = pd.read_csv(TAB / "descriptives_city_type.csv")
    for city, ck in (("Hanoi", "HN"), ("HCMC", "HCM")):
        r = ds[(ds["city_en"] == city) & (ds["type_en"] == "All types")].iloc[0]
        put(f"medPrice{ck}", f(r["med_price_bn"], 1))
        put(f"medPpm{ck}", n(r["med_ppm2_mil"]))
        for t in TYPE_EN.values():
            key = re.sub(r"[^A-Za-z]", "", t.title())
            rr = ds[(ds["city_en"] == city) & (ds["type_en"] == t)]
            if len(rr):
                put(f"ppm{key}{ck}", n(rr.iloc[0]["med_ppm2_mil"]))

    # ---------------- Module 1
    lad = pd.read_csv(TAB / "hedonic_ladder_stats.csv")
    for i, L in enumerate("ABCD"):
        put(f"Rtwo{L}", f(lad.loc[i, "R2"], 3))
        put(f"cv{L}", f(lad.loc[i, "CV_RMSE_log"], 3))
    cv = lad["CV_RMSE_log"].to_numpy()
    put("cutDistrict", pct(1 - cv[1] / cv[0]))
    put("cutOSM", pct(1 - cv[2] / cv[1], 1))
    put("cutOSMfromPhys", pct(1 - cv[2] / cv[0]))
    put("cutTotal", pct(1 - cv[3] / cv[0]))
    put("shareOSMofFE", pct((cv[0] - cv[4]) / (cv[0] - cv[1])))
    for city, ck in (("Hanoi", "HN"), ("HCMC", "HCM")):
        put(f"twiceFar{ck}", pct(1 - 2 ** grad[city]))
        put(f"ratioSFapt{ck}", f(ds[(ds["city_en"] == city) & (ds["type_en"] == "Street-front house")]["med_ppm2_mil"].iloc[0]
                                 / ds[(ds["city_en"] == city) & (ds["type_en"] == "Apartment")]["med_ppm2_mil"].iloc[0], 1))
    put("RtwoE", f(lad.loc[4, "R2"], 3))
    put("cvE", f(lad.loc[4, "CV_RMSE_log"], 3))
    put("cutOSMnoFE", pct(1 - cv[4] / cv[0]))
    coef = pd.read_csv(TAB / "hedonic_ladder.csv", encoding="utf-8-sig").set_index("variable")
    put("bAreaA", f(coef.loc["Log area", "(1) Physical"], 2))
    put("bAreaD", f(coef.loc["Log area", "(4) + Interactions"], 2))
    put("areaTen", pct(1.1 ** coef.loc["Log area", "(4) + Interactions"] - 1, 1))
    put("bCbdC", f(coef.loc["Log distance to CBD", "(3) + OSM spatial"], 3))
    put("pCbdC", f(coef.loc["Log distance to CBD", "(3) + OSM spatial_p"], 3))
    put("bMetroC", f(coef.loc["Log distance to metro", "(3) + OSM spatial"], 3))
    put("pMetroC", f(coef.loc["Log distance to metro", "(3) + OSM spatial_p"], 3))
    put("bFrontA", f(coef.loc["Frontage (m)", "(1) Physical"], 3))
    put("bFrontD", f(coef.loc["Frontage (m)", "(4) + Interactions"], 3))
    put("bRoadA", f(coef.loc["Access road width (m)", "(1) Physical"], 3))
    put("bRoadD", f(coef.loc["Access road width (m)", "(4) + Interactions"], 3))
    put("bContractD", f(coef.loc["Legal: sale contract", "(4) + Interactions"], 3))
    put("pContractD", f(coef.loc["Legal: sale contract", "(4) + Interactions_p"], 3))
    put("bOtherLegalD", f(coef.loc["Legal: other/unknown", "(4) + Interactions"], 3))
    put("pctOtherLegal", pct(np.exp(coef.loc["Legal: other/unknown", "(4) + Interactions"]) - 1))
    put("pctOtherLegalAbs", pct(abs(np.exp(coef.loc["Legal: other/unknown", "(4) + Interactions"]) - 1)))
    put("bStoreysD", f(coef.loc["Storeys", "(4) + Interactions"], 3))
    put("pctStoreys", pct(np.exp(coef.loc["Storeys", "(4) + Interactions"]) - 1, 1))
    put("pctRoad", pct(np.exp(coef.loc["Access road width (m)", "(4) + Interactions"]) - 1, 1))
    put("bParkC", f(coef.loc["Log distance to park", "(3) + OSM spatial"], 3))
    put("bMarketC", f(coef.loc["Log distance to market/mall", "(3) + OSM spatial"], 3))
    put("pctContract", pct(np.exp(coef.loc["Legal: sale contract", "(4) + Interactions"]) - 1))
    met = pd.read_csv(TAB / "avm_metrics.csv").set_index("model")
    for mod, k in (("XGBoost", "xgb"), ("OLS (4)", "ols")):
        r = met.loc[mod]
        put(f"{k}RMSE", f(r["RMSE_log"], 3))
        put(f"{k}MAE", f(r["MAE_log"], 3))
        put(f"{k}MAPE", pct(r["MAPE"], 1))
        put(f"{k}MdAPE", pct(r["MdAPE"], 1))
        put(f"{k}PPEten", pct(r["PPE10"], 1))
        put(f"{k}PPEtwenty", pct(r["PPE20"], 1))
        put(f"{k}COD", f(r["COD"], 1))
        put(f"{k}PRD", f(r["PRD"], 2))
    put("Ntest", n(met.loc["XGBoost", "N"]))
    put("rmseGainXGB", pct(1 - met.loc["XGBoost", "RMSE_log"] / met.loc["OLS (4)", "RMSE_log"]))
    put("ppeGainXGB", f(100 * (met.loc["XGBoost", "PPE20"] - met.loc["OLS (4)", "PPE20"]), 1))
    seg = pd.read_csv(TAB / "avm_metrics_by_segment.csv", encoding="utf-8-sig")
    sx = seg[(seg["model"] == "XGBoost") & (seg["city"] == "Both")].set_index("property_type")
    best, worst = sx["PPE20"].idxmax(), sx["PPE20"].idxmin()
    put("bestTypeName", TYPE_EN[best].lower().replace("villa / terraced", "villas and terraced houses"))
    put("bestTypePPE", pct(sx.loc[best, "PPE20"]))
    put("worstTypeName", TYPE_EN[worst].lower().replace("villa / terraced", "villas and terraced houses"))
    put("worstTypePPE", pct(sx.loc[worst, "PPE20"]))
    for t in TYPE_EN:
        key = re.sub(r"[^A-Za-z]", "", TYPE_EN[t].title())
        put(f"ppeTwenty{key}", pct(sx.loc[t, "PPE20"]))
        put(f"mdape{key}", pct(sx.loc[t, "MdAPE"]))
    cov = pd.read_csv(TAB / "conformal_coverage.csv", encoding="utf-8-sig")
    put("confCovGlobal", pct(cov.loc[cov["interval"] == "Global", "coverage"].iloc[0], 1))
    put("confCovType", pct(cov.loc[(cov["interval"] != "Global") & (cov["property_type"] == "All"), "coverage"].iloc[0], 1))
    ct = cov[(cov["interval"] != "Global") & (cov["property_type"] != "All")].set_index("property_type")
    put("confWidthGlobal", pct(cov.loc[cov["interval"] == "Global", "width_pct"].iloc[0]))
    for t in TYPE_EN:
        key = re.sub(r"[^A-Za-z]", "", TYPE_EN[t].title())
        put(f"confWidth{key}", pct(ct.loc[t, "width_pct"]))
        put(f"confCov{key}", pct(ct.loc[t, "coverage"]))
    put("confLowHalf", pct(np.exp(-ct["q_log"].min()) - 1))
    shp = pd.read_csv(TAB / "avm_shap_importance.csv")
    shp.columns = ["feature", "v"]
    from features import nice
    put("avmTopOne", nice(shp.iloc[0]["feature"]).lower() if shp.iloc[0]["feature"] != "District (as listed)" else "district")
    put("avmTopTwo", nice(shp.iloc[1]["feature"]).lower())
    put("avmTopThree", nice(shp.iloc[2]["feature"]).lower())
    xp = pd.read_csv(TAB / "xgb_avm_params.csv").iloc[0]
    put("xgbDepth", n(xp["max_depth"]))
    put("xgbTrees", n(xp["n_estimators"]))
    put("xgbEta", f(xp["learning_rate"], 2))

    # ---------------- Module 2
    sm = pd.read_csv(TAB / "stale_model_comparison.csv", encoding="utf-8-sig").set_index("model")
    sp = pd.read_csv(TAB / "stale_model_params.csv").iloc[0]
    x, lg, rl = sm.loc["XGBoost"], sm.loc["Logistic regression"], sm.loc["Rule: AVM gap > +15%"]
    put("staleDays", n(sp["threshold_days"]))
    put("stalePosShare", pct(sp["positive_share"]))
    age = df["age_id_days"]
    put("staleShareNinety", pct((age > 90).mean()))
    put("xgbAUC", f(x["AUC"], 2))
    put("xgbAUClo", f(x["AUC_lo"], 2))
    put("xgbAUChi", f(x["AUC_hi"], 2))
    put("logitAUC", f(lg["AUC"], 2))
    put("logitAUClo", f(lg["AUC_lo"], 2))
    put("logitAUChi", f(lg["AUC_hi"], 2))
    put("ruleAUC", f(rl["AUC"], 2))
    put("ruleAUClo", f(rl["AUC_lo"], 2))
    put("ruleAUChi", f(rl["AUC_hi"], 2))
    put("xgbPRAUC", f(x["PR_AUC"], 2))
    put("logitPRAUC", f(lg["PR_AUC"], 2))
    put("rulePRAUC", f(rl["PR_AUC"], 2))
    put("baseRate", pct(x["pos_share"]))
    put("topDecileCapture", pct(x["top_decile_capture"]))
    put("topDecilePrecision", pct(x["top_decile_precision"]))
    put("topDecileLift", f(x["top_decile_precision"] / x["pos_share"], 1))
    put("ruleTopDecilePrecision", pct(rl["top_decile_precision"]))
    put("staleThreshold", f(x["threshold"], 2))
    put("stalePrecision", pct(x["precision"]))
    put("staleRecall", pct(x["recall"]))
    put("rulePrecision", pct(rl["precision"]))
    put("ruleRecall", pct(rl["recall"]))
    put("xgbBrier", f(x["Brier"], 3))
    put("logitBrier", f(lg["Brier"], 3))
    put("NstaleTest", n(x["N_test"]))
    rob = pd.read_csv(TAB / "stale_robustness.csv", encoding="utf-8-sig")
    rx = rob[rob["model"] == "XGBoost"]
    for lab, key in (("Creation age (listing ID)", "Created"), ("Posted date (refreshed)", "Posted")):
        for d in (60, 90, 120, 180):
            r = rx[(rx["label"] == lab) & (rx["threshold_days"] == d)]
            if len(r):
                word = {60: "Sixty", 90: "Ninety", 120: "OneTwenty", 180: "OneEighty"}[d]
                put(f"auc{key}{word}", f(r["AUC"].iloc[0], 2))
                put(f"share{key}{word}", pct(r["pos_share"].iloc[0]))
    rr = rob[(rob["model"].str.startswith("Rule")) & (rob["label"] == "Posted date (refreshed)") & (rob["threshold_days"] == 90)]
    if len(rr):
        put("rulePostedNinety", f(rr["AUC"].iloc[0], 2))
    opt = pd.read_csv(TAB / "stale_cost_optimal.csv").set_index("fn_cost_ratio")
    for r, w in ((1, "One"), (3, "Three"), (5, "Five")):
        put(f"costThr{w}", f(opt.loc[r, "threshold"], 2))
        put(f"costFlag{w}", pct(opt.loc[r, "flag_rate_test"]))
    ss = pd.read_csv(TAB / "stale_shap_importance.csv")
    ss.columns = ["feature", "v"]
    put("staleTopOne", nice(ss.iloc[0]["feature"]).lower())
    put("staleTopTwo", nice(ss.iloc[1]["feature"]).lower())
    put("staleTopThree", nice(ss.iloc[2]["feature"]).lower())
    rank = int(np.where(ss["feature"] == "avm_resid")[0][0]) + 1
    put("residRank", n(rank))
    put("NstaleFeatures", n(len(ss)))
    sc = pd.read_csv(MODELS / "overpricing_scores.csv")
    put("shareHighRisk", pct(sc["high_risk"].mean()))
    av = pd.read_csv(MODELS / "avm_predictions.csv")
    from scipy.stats import spearmanr
    mm = df[["listing_id", "age_id_days"]].merge(av[["listing_id", "avm_resid"]], on="listing_id")
    put("rhoResidAge", f(spearmanr(mm["avm_resid"], mm["age_id_days"])[0], 2))

    # ---------------- Module 3
    ss3 = pd.read_csv(TAB / "segment_summary.csv").iloc[0]
    put("kBest", n(ss3["k"]))
    put("kOverall", n(ss3["k_best_overall"]))
    put("silBest", f(ss3["silhouette"], 2))
    put("ariWard", f(ss3["ari_ward"], 2))
    put("NdistAll", n(ss3["n_districts_total"]))
    put("NdistKept", n(ss3["n_districts_kept"]))
    put("segCover", pct(ss3["listings_covered"] / len(df)))
    prof = pd.read_csv(TAB / "segment_profiles.csv", encoding="utf-8-sig")
    for i, r in prof.iterrows():
        L = "ABCDEFGH"[i]
        put(f"seg{L}name", r["segment"])
        put(f"seg{L}n", n(r["n_districts"]))
        put(f"seg{L}listings", n(r["n_listings"]))
        put(f"seg{L}ppm", n(r["med_ppm2_mil"]))
        put(f"seg{L}age", n(r["med_age"]))
        put(f"seg{L}cbd", f(r["med_dist_cbd"], 1))
        put(f"seg{L}risk", pct(r["share_high_risk"]))
        put(f"seg{L}hn", n(r["hanoi_districts"]))
    put("Nsegments", n(len(prof)))
    mid = prof[prof["segment"].str.contains("mid-market")]["med_ppm2_mil"].median()
    put("ratioPrime", f(prof.iloc[0]["med_ppm2_mil"] / mid, 1))
    dsg = pd.read_csv(TAB / "district_segments.csv", encoding="utf-8-sig")
    for i, r in prof.iterrows():
        L = "ABCDEFGH"[i]
        names = dsg.loc[dsg["segment"] == r["segment"]].sort_values("n", ascending=False)["district"]
        put(f"seg{L}districts", ", ".join(names.head(4).str.replace("Huyện ", "").str.replace(" (TP. Thủ Đức)", "")))

    # ---------------- cross-functional
    xs = pd.read_csv(TAB / "xf_sales_districts.csv", encoding="utf-8-sig")
    top = xs.iloc[0]
    put("topRiskDistrict", f"{top['district']} ({top['city_en']})")
    put("topRiskShare", pct(top["share_high_risk"]))
    low = xs.iloc[-1]
    put("lowRiskDistrict", f"{low['district']} ({low['city_en']})")
    put("lowRiskShare", pct(low["share_high_risk"]))
    xt = pd.read_csv(TAB / "xf_sales_types.csv").set_index("type_en")
    for t in TYPE_EN.values():
        key = re.sub(r"[^A-Za-z]", "", t.title())
        put(f"gap{key}", pct(np.exp(xt.loc[t, "med_gap"]) - 1, 1))
        put(f"overFifteen{key}", pct(xt.loc[t, "share_over15"]))
    xr = pd.read_csv(TAB / "xf_risk_types.csv").set_index("type_en")
    for t in TYPE_EN.values():
        key = re.sub(r"[^A-Za-z]", "", t.title())
        put(f"over{key}", pct(xr.loc[t, "over20"]))
        put(f"oosPPE{key}", pct(xr.loc[t, "PPE20"]))
        put(f"use{key}", xr.loc[t, "use"])
    put("overAll", pct((np.exp(av["avm_pred_ln"] - av["ln_price"]) > 1.2).mean()))

    # ---------------- value
    v = pd.read_csv(TAB / "value_scenarios.csv", index_col=0)
    for s, k in (("conservative", "Cons"), ("base", "Base"), ("optimistic", "Opt")):
        put(f"netPartner{k}", f(v.loc["Net partner value", s] / 1000, 1))
        put(f"partner{k}", f(v.loc["Partner value", s] / 1000, 1))
        put(f"roi{k}", f(v.loc["ROI (net / run cost)", s], 1))
        put(f"buyer{k}", f(v.loc["Buyer overpayment avoided (not in ROI)", s] / 1000, 1))
        put(f"runCost{k}", f(v.loc["Run cost", s] / 1000, 1))
        put(f"broker{k}", f(v.loc["Broker time saved", s] / 1000, 1))
        put(f"bank{k}", f(v.loc["Bank appraisal cost avoided", s] / 1000, 1))
    put("netPartnerConsAbs", f(abs(v.loc["Net partner value", "conservative"]) / 1000, 1))
    va = pd.read_csv(TAB / "value_assumptions.csv", encoding="utf-8-sig").set_index("name")
    put("shareAboveBand", pct(va.loc["share_above", "base"]))
    put("excessAboveBand", pct(va.loc["excess", "base"]))
    tor = pd.read_csv(TAB / "value_tornado.csv")
    put("tornadoTop", tor.iloc[0]["assumption"].lower())

    lines = ["% Auto-generated by code/10_export_numbers.py. Do not edit by hand."]
    for k, val in M.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{val}\\xspace}}" if not val.endswith("\\%") and not
                     re.fullmatch(r"[\d{},.$\-]+", val) else f"\\newcommand{{\\{k}}}{{{val}\\xspace}}")
    block = "\n".join(lines) + "\n"
    for tex in (REPORT / "main.tex", REPORT / "slides" / "pitch.tex"):
        t = tex.read_text(encoding="utf-8")
        a, b = t.index("% BEGIN NUMBERS\n") + len("% BEGIN NUMBERS\n"), t.index("% END NUMBERS")
        tex.write_text(t[:a] + block + t[b:], encoding="utf-8")
    pd.Series(M).to_csv(TAB / "report_numbers.csv", header=["value"], encoding="utf-8-sig")
    log.info(f"wrote {len(M)} macros into report/main.tex and report/slides/pitch.tex")
    _ = (CITY_SHORT, OSM)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
