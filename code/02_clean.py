"""Phase 2: parse, clean and engineer features for the mogi.vn listings.

Input : data/raw/mogi_raw.csv
Output: data/processed/listings_clean.parquet (analysis-ready)
        result/tables/cleaning_waterfall.csv, listing_age_checks.csv,
        sample_extract.csv, data_dictionary.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BBOX, EXTERNAL, PROCESSED, RAW, TAB, TYPE_EN, get_logger  # noqa: E402
from dictionary import write_dictionary  # noqa: E402
from mogi_parsers import parse_area, parse_price, parse_vn_date  # noqa: E402

log = get_logger("02_clean")

AREA_BOUNDS = {"Căn hộ chung cư": (20, 400), "Nhà hẻm/ngõ": (15, 600), "Nhà mặt tiền/phố": (15, 600),
               "Biệt thự/liền kề": (60, 2000), "Đất": (20, 5000)}
# minimum plausible asking price per m2 (VND million) in Hanoi / HCMC, by type
PPM2_FLOOR = {"Căn hộ chung cư": 15, "Nhà hẻm/ngõ": 15, "Nhà mặt tiền/phố": 15, "Biệt thự/liền kề": 10, "Đất": 1}
NUM_IMPUTE = ["bedrooms", "bathrooms", "floors", "frontage_m", "road_width_m", "depth_m",
              "agent_tenure_months"]
KW = ["kw_car_access", "kw_corner", "kw_business", "kw_lake_view", "kw_near_school", "kw_owner",
      "kw_urgent", "kw_furnished"]


def legal_group(s):
    if s in ("Sổ đỏ", "Sổ hồng"):
        return "Sổ đỏ/Sổ hồng"
    if s == "Hợp đồng mua bán":
        return "Hợp đồng mua bán"
    if s == "Đang chờ sổ":
        return "Đang chờ sổ"
    return "Khác/không rõ"


def parse(raw):
    df = raw.copy()
    df["scraped_date"] = pd.to_datetime(df["scraped_at"]).dt.date
    # area_m2 is stored numeric by the scraper; re-derive from the raw strings when empty
    area = pd.to_numeric(df["area_m2"], errors="coerce")
    alt = [parse_area(a) or parse_area(b) if t != "Căn hộ chung cư" else parse_area(b) or parse_area(a)
           for t, a, b in zip(df["property_type"], df["land_area_raw"], df["area_used_raw"])]
    df["area_m2"] = area.fillna(pd.Series(alt, index=df.index, dtype=float))
    pv = [parse_price(p, a) for p, a in zip(df["price_raw"], df["area_m2"])]
    df["price_vnd"] = pd.to_numeric(pd.Series(pv, index=df.index), errors="coerce")
    df["price_vnd_ld"] = pd.to_numeric(df["price_vnd_ld"], errors="coerce")
    df["negotiable"] = df["price_raw"].str.lower().str.contains("thỏa thuận|thoả thuận|liên hệ", na=False)
    # fall back on the numeric JSON-LD price only when the string could not be parsed
    fb = df["price_vnd"].isna() & ~df["negotiable"] & (df["price_vnd_ld"] > 0)
    df.loc[fb, "price_vnd"] = df.loc[fb, "price_vnd_ld"]
    df["price_from_ld"] = fb.astype(int)
    for c in ["bedrooms", "bathrooms", "floors", "frontage_m", "road_width_m", "depth_m", "lat", "lon",
              "n_images", "description_len", "agent_tenure_months", "listing_id"] + KW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["posted"] = [parse_vn_date(p, d) for p, d in zip(df["posted_date"], df["scraped_date"])]
    df["img_first"] = pd.to_datetime(df["img_first_date"], errors="coerce").dt.date
    for c in ["district", "ward", "street", "legal_status", "project_name", "seller_type"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df


def main():
    t0 = time.time()
    raw = pd.read_csv(RAW / "mogi_raw.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False,
                      na_values=[""])
    df = parse(raw)
    agree = (np.abs(df["price_vnd"] / df["price_vnd_ld"] - 1) < 0.01).mean()
    log.info(f"price string vs JSON-LD numeric price agree within 1% for {agree:.1%} of rows")

    wf = [("Raw listings scraped (with detail page)", len(df))]

    def step(label, mask):
        nonlocal df
        df = df[mask].copy()
        wf.append((label, len(df)))
        log.info(f"{label}: {wf[-2][1] - len(df)} dropped, {len(df)} left")

    step("Remove repeated listing IDs", ~df["listing_id"].duplicated())
    step("Remove negotiable or unparseable price", df["price_vnd"].notna() & (df["price_vnd"] > 0))
    step("Remove missing or zero area", df["area_m2"].notna() & (df["area_m2"] > 0))
    step("Remove unknown district", df["district"] != "")
    lo = df["property_type"].map(lambda t: AREA_BOUNDS[t][0])
    hi = df["property_type"].map(lambda t: AREA_BOUNDS[t][1])
    step("Remove area outside type bounds", df["area_m2"].between(lo, hi))
    step("Remove bedroom count above 15", ~(df["bedrooms"] > 15))
    df["price_per_m2"] = df["price_vnd"] / df["area_m2"]
    # rentals posted in sale categories ('cho thuê' in the title without 'bán')
    t = df["title"].fillna("").str.lower()
    step("Remove rental listings posted as sales", ~(t.str.contains("cho thuê") & ~t.str.contains("bán")))
    # placeholder or per-m2 prices typed as totals (e.g. '999.999 đ', '82 triệu' for a 97 m2 plot)
    floor = df["property_type"].map(PPM2_FLOOR)
    step("Remove implausible price (total < VND 300 m or price/m2 below type floor)",
         (df["price_vnd"] >= 300e6) & (df["price_per_m2"] >= floor * 1e6))
    q = df.groupby(["city", "property_type"])["price_per_m2"].transform
    step("Remove price per m2 outside 1st to 99th percentile (city x type)",
         df["price_per_m2"].between(q(lambda s: s.quantile(0.01)), q(lambda s: s.quantile(0.99))))

    # near-duplicates: same property posted by several brokers -> keep earliest posted
    key = ["city", "district", "ward", "property_type", "area_r", "price_r", "bedrooms_k"]
    df["area_r"] = df["area_m2"].round(0)
    df["price_r"] = (df["price_vnd"] / 50e6).round(0)
    df["bedrooms_k"] = df["bedrooms"].fillna(-1)
    df = df.sort_values(["posted", "listing_id"])
    dup = df.duplicated(key, keep="first")
    n_before = len(df)
    step("Remove near-duplicate postings (keep earliest)", ~dup)
    dup_share = dup.sum() / n_before
    log.info(f"near-duplicate share removed: {dup_share:.1%}")
    df = df.drop(columns=["area_r", "price_r", "bedrooms_k"])

    # coordinates: site-supplied; validate against city boxes, fall back to Nominatim
    inside = pd.Series(False, index=df.index)
    for city, (s, w, n, e) in BBOX.items():
        inside |= (df["city"] == city) & df["lat"].between(s, n) & df["lon"].between(w, e)
    df.loc[~inside, ["lat", "lon"]] = np.nan
    df.loc[~inside, "geo_precision"] = ""
    log.info(f"listings without valid site coordinates: {(~inside).sum()}")
    if (~inside).any():
        geocode_missing(df)
    step("Remove listings without valid coordinates", df["lat"].notna())
    # mogi.vn pins many listings on a ward centroid when the poster sets no map point:
    # a coordinate shared by listings on 3+ different streets is treated as ward-level precision
    pt = df.groupby(["lat", "lon"]).agg(geo_shared_n=("listing_id", "size"), geo_n_streets=("street", "nunique"))
    df = df.merge(pt, left_on=["lat", "lon"], right_index=True, how="left")
    site = df["geo_precision"] == "exact"
    df.loc[site, "geo_precision"] = np.where(df.loc[site, "geo_n_streets"] >= 3, "ward_centroid", "point")
    log.info("coordinate precision:\n" + df["geo_precision"].value_counts(normalize=True).round(3).to_string())

    pd.DataFrame(wf, columns=["step", "n_remaining"]).assign(
        n_dropped=lambda d: (-d["n_remaining"].diff()).fillna(0).astype(int)
    ).to_csv(TAB / "cleaning_waterfall.csv", index=False, encoding="utf-8-sig")

    df = listing_age(df)
    df = features(df)
    df = df.sort_values("listing_id").reset_index(drop=True)
    keep = [c for c in df.columns if c not in ("description_full",)]
    df[keep].to_parquet(PROCESSED / "listings_clean.parquet", index=False)
    pd.DataFrame([{"near_duplicate_share": dup_share, "price_agree_ld": agree, "n_clean": len(df),
                   "share_point": (df["geo_precision"] == "point").mean(),
                   "share_ward_centroid": (df["geo_precision"] == "ward_centroid").mean(),
                   "share_geocoded_fallback": df["geo_precision"].isin(["ward", "district"]).mean()}]).to_csv(TAB / "cleaning_stats.csv", index=False)
    sample_extract(df)
    write_dictionary(df, TAB / "data_dictionary.csv")
    log.info(f"clean rows: {len(df)}  ({time.time() - t0:.0f} s)")
    log.info("by city x type:\n" + pd.crosstab(df["property_type"], df["city"], margins=True).to_string())


def geocode_missing(df):
    """Nominatim fallback (ward, then district level) with a persistent cache."""
    from geopy.geocoders import Nominatim
    cache_f = EXTERNAL / "geocode_cache.csv"
    cache = pd.read_csv(cache_f, encoding="utf-8-sig") if cache_f.exists() else \
        pd.DataFrame(columns=["query", "lat", "lon"])
    known = {r.query: (r.lat, r.lon) for r in cache.itertuples()}
    geo = Nominatim(user_agent="AdvancedBA-ResearchBot/1.0 (university course project)", timeout=20)
    new = []

    def lookup(q):
        if q in known:
            return known[q]
        time.sleep(1.1)
        try:
            res = geo.geocode(q)
        except Exception as e:  # network trouble -> treat as not found, keep going
            log.info(f"  geocode error {q}: {e}")
            res = None
        val = (res.latitude, res.longitude) if res else (np.nan, np.nan)
        known[q] = val
        new.append({"query": q, "lat": val[0], "lon": val[1]})
        return val

    for i in df.index[df["lat"].isna()]:
        r = df.loc[i]
        s, w, n, e = BBOX[r["city"]]
        for level, q in (("ward", f"{r['ward']}, {r['district']}, {r['city']}, Vietnam"),
                         ("district", f"{r['district']}, {r['city']}, Vietnam")):
            if level == "ward" and not r["ward"]:
                continue
            lat, lon = lookup(q)
            if pd.notna(lat) and s <= lat <= n and w <= lon <= e:
                df.loc[i, ["lat", "lon", "geo_precision"]] = [lat, lon, level]
                break
    if new:
        pd.concat([cache, pd.DataFrame(new)]).to_csv(cache_f, index=False, encoding="utf-8-sig")
    log.info(f"geocoded by fallback: {df['geo_precision'].isin(['ward', 'district']).sum()}")


def listing_age(df):
    sd = pd.to_datetime(df["scraped_date"])
    df["listing_age_days"] = (sd - pd.to_datetime(df["posted"])).dt.days
    df["age_created_days"] = (sd - pd.to_datetime(df["img_first"])).dt.days
    ok = df["posted"].notna()
    post_ord = np.array([d.toordinal() for d in df.loc[ok, "posted"]])
    rho_post, p_post = spearmanr(df.loc[ok, "listing_id"], post_ord)
    okc = df["img_first"].notna()
    img_ord = np.array([d.toordinal() for d in df.loc[okc, "img_first"]])
    rho_img, p_img = spearmanr(df.loc[okc, "listing_id"], img_ord)
    both = ok & okc
    refreshed = ((pd.to_datetime(df.loc[both, "posted"]) - pd.to_datetime(df.loc[both, "img_first"]))
                 .dt.days > 30).mean()
    # ID-based creation date: monotone fit of photo upload date on listing ID (isotonic)
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(df.loc[okc, "listing_id"], img_ord)
    created_id = iso.predict(df["listing_id"])
    df["age_id_days"] = (np.array([d.toordinal() for d in df["scraped_date"]]) - created_id).round()
    checks = pd.DataFrame([
        {"check": "Spearman rho, listing ID vs posted date", "value": rho_post, "p_value": p_post,
         "n": int(ok.sum())},
        {"check": "Spearman rho, listing ID vs first photo upload date", "value": rho_img,
         "p_value": p_img, "n": int(okc.sum())},
        {"check": "Share of listings whose posted date is >30 days after first photo upload",
         "value": refreshed, "p_value": np.nan, "n": int(both.sum())},
        {"check": "Median listing age, posted date (days)", "value": df["listing_age_days"].median(),
         "p_value": np.nan, "n": int(ok.sum())},
        {"check": "Median listing age, first photo upload (days)",
         "value": df["age_created_days"].median(), "p_value": np.nan, "n": int(okc.sum())},
    ])
    checks.to_csv(TAB / "listing_age_checks.csv", index=False, encoding="utf-8-sig")
    log.info("listing age checks:\n" + checks.to_string(index=False))
    log.info("listing_age_days quantiles:\n" + df["listing_age_days"].describe(
        percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
    return df


def features(df):
    df["ln_price"] = np.log(df["price_vnd"])
    df["ln_area"] = np.log(df["area_m2"])
    df["ppm2_mil"] = df["price_per_m2"] / 1e6
    df["ln_ppm2"] = np.log(df["price_per_m2"])
    df["price_bn"] = df["price_vnd"] / 1e9
    df["legal_group"] = df["legal_status"].map(legal_group)
    df["has_project"] = (df["project_name"] != "").astype(int)
    df["broker_profile"] = (df["seller_type"] == "broker_profile").astype(int)
    df["furnished"] = df["kw_furnished"].fillna(0).astype(int)
    df["type_en"] = df["property_type"].map(TYPE_EN)
    df["is_land"] = (df["property_type"] == "Đất").astype(int)
    # implausible text-derived values -> missing before imputation
    df.loc[~df["floors"].between(1, 15), "floors"] = np.nan
    df.loc[~df["road_width_m"].between(1, 60), "road_width_m"] = np.nan
    df.loc[~df["frontage_m"].between(1, 100), "frontage_m"] = np.nan
    for c in NUM_IMPUTE:
        df[f"{c}_missing"] = df[c].isna().astype(int)
        med = df.groupby("property_type")[c].transform("median")
        df[c] = df[c].fillna(med).fillna(0)
    for c in KW:
        df[c] = df[c].fillna(0).astype(int)
    df["n_images"] = df["n_images"].fillna(0)
    df["description_len"] = df["description_len"].fillna(0)
    df["ln_desc_len"] = np.log1p(df["description_len"])
    return df


def sample_extract(df):
    rng = np.random.default_rng(42)
    picks = []
    for city in df["city"].unique():
        for t in ["Căn hộ chung cư", "Nhà hẻm/ngõ", "Nhà mặt tiền/phố", "Đất"]:
            sub = df[(df["city"] == city) & (df["property_type"] == t)]
            if len(sub):
                picks.append(sub.iloc[rng.integers(len(sub))])
    ex = pd.DataFrame(picks).head(8)
    cols = {"listing_id": "ID", "city": "City", "district": "District (as listed)",
            "property_type": "Type (site label)", "price_bn": "Price (VND bn)", "area_m2": "Area (m²)",
            "ppm2_mil": "VND m/m²", "bedrooms": "Beds", "legal_status": "Legal", "posted_date": "Posted"}
    ex = ex[list(cols)].rename(columns=cols)
    ex.to_csv(TAB / "sample_extract.csv", index=False, encoding="utf-8-sig")



if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
