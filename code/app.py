"""Streamlit prototype: python -m streamlit run code/app.py

Page 1 (Valuation): AVM fair value with a 90% conformal interval, overpricing gap and stale-listing
risk for an optional asking price, district segment, 5 comparable listings, segment map.
Page 2 (Market overview): executive matrix, segment profiles, district price table.
Models are loaded from result/models/ (no retraining in the app).
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIG, MODELS, PROCESSED, TAB, TYPE_EN  # noqa: E402
from features import add_log_dist, tree_matrix  # noqa: E402

st.set_page_config(page_title="Vietnam Real Estate AVM", page_icon="🏠", layout="wide")
LEGAL = ["Sổ đỏ/Sổ hồng", "Hợp đồng mua bán", "Đang chờ sổ", "Khác/không rõ"]
LOC_COLS = ["lat", "lon", "accessibility_index", "n_bus_stop_500m", "n_school_1000m", "n_market_1000m"] + \
           [f"dist_{l}_km" for l in ["metro", "bus_stop", "school", "university", "hospital", "park", "market",
                                     "water", "cbd"]]


@st.cache_resource
def load():
    df = add_log_dist(pd.read_parquet(PROCESSED / "listings_model.parquet"))
    avm = joblib.load(MODELS / "xgb_avm.joblib")
    conf = joblib.load(MODELS / "conformal.joblib")
    stale = joblib.load(MODELS / "xgb_stale.joblib")
    seg = pd.read_csv(TAB / "district_segments.csv")
    lookup = pd.read_csv(MODELS / "district_lookup.csv").set_index("district_key")
    scores = pd.read_csv(MODELS / "overpricing_scores.csv")
    return df, avm, conf, stale, seg, lookup, scores


df, avm, conf, stale, seg, lookup, scores = load()


def location_row(city, district, ward):
    """Location features: median over listed properties in the ward (fallback: district)."""
    d = df[(df["city"] == city) & (df["district"] == district)]
    w = d[d["ward"] == ward] if ward else d
    src = w if len(w) >= 3 else d
    return src[LOC_COLS].median(), ("ward" if len(w) >= 3 else "district")


def valuation_page():
    st.title("Vietnam Real Estate AVM")
    st.caption("Prototype. Estimates are trained on asking prices from mogi.vn (Oct 2026) and OpenStreetMap "
               "features; they are a reference, not a formal appraisal.")
    sb = st.sidebar
    sb.header("Property")
    city = sb.selectbox("City", sorted(df["city"].unique()))
    dists = df[df["city"] == city]["district"].value_counts()
    district = sb.selectbox("District (as listed)", list(dists.index))
    wards = df[(df["city"] == city) & (df["district"] == district)]["ward"].replace("", np.nan).dropna()
    ward = sb.selectbox("Ward", [""] + sorted(wards.unique()))
    ptype = sb.selectbox("Property type", list(TYPE_EN), format_func=lambda t: f"{TYPE_EN[t]} ({t})")
    sub = df[df["property_type"] == ptype]
    area = sb.number_input("Area (m²)", 10.0, 5000.0, float(sub["area_m2"].median()), step=5.0)
    beds = sb.number_input("Bedrooms", 0, 15, int(sub["bedrooms"].median()))
    baths = sb.number_input("Bathrooms", 0, 15, int(sub["bathrooms"].median()))
    floors = sb.number_input("Storeys", 0, 15, int(sub["floors"].median()))
    front = sb.number_input("Frontage (m)", 0.0, 100.0, float(sub["frontage_m"].median()), step=0.5)
    road = sb.number_input("Access road width (m)", 0.0, 60.0, float(sub["road_width_m"].median()), step=0.5)
    legal = sb.selectbox("Legal status", LEGAL)
    ask = sb.number_input("Asking price (VND bn, optional; 0 = none)", 0.0, 2000.0, 0.0, step=0.1)

    loc, loc_level = location_row(city, district, ward)
    row = pd.DataFrame([{**loc.to_dict(), "city": city, "district": district, "ward": ward,
                         "property_type": ptype, "legal_group": legal, "ln_area": np.log(area),
                         "area_m2": area, "bedrooms": beds, "bathrooms": baths, "floors": floors,
                         "frontage_m": front, "road_width_m": road, "floors_missing": 0, "frontage_m_missing": 0,
                         "road_width_m_missing": 0, "bedrooms_missing": 0, "bathrooms_missing": 0,
                         "kw_car_access": 0, "kw_corner": 0, "kw_business": 0, "kw_lake_view": 0,
                         "kw_near_school": 0, "kw_furnished": 0}])
    row = add_log_dist(row)
    X = tree_matrix(row, avm["columns"])
    pred = float(avm["model"].predict(X)[0])
    q = conf["q_type"].get(ptype, conf["q_global"])
    fv, lo, hi = np.exp(pred) / 1e9, np.exp(pred - q) / 1e9, np.exp(pred + q) / 1e9

    c1, c2, c3 = st.columns(3)
    c1.metric("AVM fair value", f"{fv:,.2f} bn VND")
    c2.metric("90% interval", f"{lo:,.2f} to {hi:,.2f} bn")
    c3.metric("Implied price per m²", f"{fv * 1000 / area:,.0f} m VND/m²")
    st.caption(f"Location features taken from the {loc_level} median of listed properties.")

    s = seg[(seg["city"] == city) & (seg["district"] == district)]
    st.subheader("District segment")
    if len(s):
        r = s.iloc[0]
        st.write(f"**{r['segment']}**: median {r['med_ppm2_mil']:,.0f} million VND/m², median listing age "
                 f"{r['med_age']:.0f} days, {int(r['n'])} listings.")
    else:
        st.write("Fewer than 30 listings in this district: no segment assigned.")

    if ask > 0:
        st.subheader("Asking price check")
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
            "kw_car_access": 0, "kw_corner": 0,
            "kw_business": 0, "kw_lake_view": 0, "kw_near_school": 0, "kw_owner": 0, "kw_urgent": 0,
            "kw_furnished": 0, "property_type": ptype, "city": city, "legal_group": legal}])
        Xs = pd.get_dummies(m2, columns=["property_type", "city", "legal_group"], dtype=float) \
            .reindex(columns=stale["columns"], fill_value=0.0)
        p = float(stale["model"].predict_proba(Xs)[0, 1])
        t = stale["high_risk_cutoff"]
        green = scores["p_stale"].quantile(0.5)
        light = "🔴 High" if p >= t else ("🟠 Medium" if p >= green else "🟢 Low")
        c1, c2 = st.columns(2)
        c1.metric("Asking price vs AVM value", f"{np.exp(gap) - 1:+.1%}")
        c2.metric(f"Stale-listing risk (> {stale['label_days']} days)", f"{light} ({p:.0%})")
        st.caption("Risk is a ranking signal trained on listing age; it is not a probability of sale.")

    st.subheader("Comparable listings")
    pool = df[(df["city"] == city) & (df["property_type"] == ptype)].copy()
    z = pool[["ln_area", "bedrooms", "bathrooms", "lat", "lon"]]
    mu, sd = z.mean(), z.std().replace(0, 1)
    target = pd.Series({"ln_area": np.log(area), "bedrooms": beds, "bathrooms": baths,
                        "lat": loc["lat"], "lon": loc["lon"]})
    w = pd.Series({"ln_area": 1.0, "bedrooms": 0.5, "bathrooms": 0.3, "lat": 1.5, "lon": 1.5})
    pool["dist"] = np.sqrt((((z - target) / sd) ** 2 * w).sum(axis=1))
    comp = pool.nsmallest(5, "dist")[["district", "ward", "area_m2", "bedrooms", "price_bn", "ppm2_mil",
                                      "legal_status", "posted_date", "url"]]
    st.dataframe(comp.rename(columns={"area_m2": "Area m²", "price_bn": "Price (VND bn)",
                                      "ppm2_mil": "m VND/m²"}), hide_index=True, use_container_width=True)

    st.subheader("Segment map")
    f = FIG / f"segments_map_{'hanoi' if city == 'Hà Nội' else 'hcmc'}.html"
    if f.exists():
        st.components.v1.html(f.read_text(encoding="utf-8"), height=420)


def overview_page():
    st.title("Market overview")
    st.image(str(FIG / "executive_matrix.png"), caption="Districts by liquidity and price level (Module 3)")
    st.subheader("Segment profiles")
    st.dataframe(pd.read_csv(TAB / "segment_profiles.csv"), hide_index=True, use_container_width=True)
    st.subheader("Price index by district (median million VND/m²)")
    piv = df.pivot_table(index=["city", "district"], columns="type_en", values="ppm2_mil", aggfunc="median")
    st.dataframe(piv.round(0), use_container_width=True)


page = st.sidebar.radio("Page", ["Valuation", "Market overview"])
valuation_page() if page == "Valuation" else overview_page()
