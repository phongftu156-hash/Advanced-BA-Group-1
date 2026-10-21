"""Data dictionary (field, type, description, source, % missing before imputation)."""
import pandas as pd

M, D, O, T = "mogi.vn detail page", "derived (team)", "OpenStreetMap (Overpass)", "mogi.vn text (regex)"
DESC = {
    "listing_id": ("Site listing ID (from URL)", M),
    "url": ("Listing URL", M),
    "city": ("City (Hà Nội, TP. Hồ Chí Minh)", M),
    "district": ("District as listed on the site (pre-2025 names)", M),
    "ward": ("Ward as listed on the site", M),
    "street": ("Street as listed", M),
    "property_type": ("Property group used for quotas and models", M),
    "subcategory": ("mogi.vn search category slug", M),
    "land_subtype": ("Đất thổ cư or Đất nền dự án (land only)", M),
    "price_vnd": ("Asking price, VND", M),
    "price_bn": ("Asking price, VND billion", D),
    "area_m2": ("Area, m2: usable area (apartments), land area (other types)", M),
    "price_per_m2": ("Asking price per m2, VND", D),
    "ppm2_mil": ("Asking price per m2, VND million", D),
    "ln_price": ("Log asking price", D),
    "ln_area": ("Log area", D),
    "ln_ppm2": ("Log price per m2", D),
    "bedrooms": ("Bedrooms (imputed: type median)", M),
    "bathrooms": ("Bathrooms (imputed: type median)", M),
    "floors": ("Storeys (houses; from text)", T),
    "frontage_m": ("Frontage / plot width, m (plot dimensions, else text)", M),
    "depth_m": ("Plot depth, m (plot dimensions)", M),
    "road_width_m": ("Width of access road or alley, m (text)", T),
    "legal_status": ("Legal document (Sổ đỏ, Sổ hồng, HĐMB, chờ sổ, khác)", M),
    "legal_group": ("Legal status grouped for modelling", D),
    "project_name": ("Project name mentioned in title (regex)", T),
    "has_project": ("1 if a project name is mentioned", D),
    "posted_date": ("Posting date shown on the site (refreshed on re-post)", M),
    "img_first_date": ("Earliest photo upload date in image URLs (creation proxy)", M),
    "listing_age_days": ("Scrape date minus posted date, days", D),
    "age_created_days": ("Scrape date minus first photo upload date, days", D),
    "age_id_days": ("Listing age implied by listing ID (isotonic fit on photo dates)", D),
    "seller_type": ("broker_profile = registered broker page; no_profile = plain account", M),
    "broker_profile": ("1 if seller has a registered broker profile", D),
    "agent_tenure_months": ("Months since the poster account joined", M),
    "n_images": ("Number of photos", M),
    "description_len": ("Description length, characters", D),
    "kw_car_access": ("Text mentions car access (ô tô, xe hơi)", T),
    "kw_corner": ("Text mentions corner lot (góc, 2 mặt tiền)", T),
    "kw_business": ("Text mentions business use (kinh doanh, cho thuê)", T),
    "kw_lake_view": ("Text mentions lake or river view", T),
    "kw_near_school": ("Text mentions nearby school", T),
    "kw_owner": ("Text says owner-listed (chính chủ)", T),
    "kw_urgent": ("Text signals urgency or price cut (bán gấp, cắt lỗ)", T),
    "kw_furnished": ("Text mentions full furniture", T),
    "lat": ("Latitude (site map embed; Nominatim fallback)", M),
    "lon": ("Longitude (site map embed; Nominatim fallback)", M),
    "geo_precision": ("point (site map pin), ward_centroid (site pin shared by 3+ streets), ward/district (Nominatim)", D),
    "dist_metro_km": ("Distance to nearest metro / urban rail station, km", O),
    "dist_bus_stop_km": ("Distance to nearest bus stop, km", O),
    "dist_school_km": ("Distance to nearest school or kindergarten, km", O),
    "dist_university_km": ("Distance to nearest university, km", O),
    "dist_hospital_km": ("Distance to nearest hospital or clinic, km", O),
    "dist_park_km": ("Distance to nearest park, km", O),
    "dist_market_km": ("Distance to nearest market, mall or supermarket, km", O),
    "dist_water_km": ("Distance to nearest water body centroid, km", O),
    "dist_cbd_km": ("Distance to CBD (Hoàn Kiếm Lake / Bến Thành Market), km", O),
    "n_bus_stop_500m": ("Bus stops within 500 m", O),
    "n_bus_stop_1000m": ("Bus stops within 1 km", O),
    "n_school_500m": ("Schools within 500 m", O),
    "n_school_1000m": ("Schools within 1 km", O),
    "n_market_500m": ("Markets/malls/supermarkets within 500 m", O),
    "n_market_1000m": ("Markets/malls/supermarkets within 1 km", O),
    "accessibility_index": ("First principal component of inverted distances (std.)", D),
}


def write_dictionary(df, path):
    rows = []
    for c, (desc, src) in DESC.items():
        if c not in df.columns:
            continue
        miss_col = f"{c}_missing"
        miss = df[miss_col].mean() if miss_col in df.columns else df[c].isna().mean()
        if df[c].dtype == object:
            miss = max(miss, (df[c].astype(str).str.strip() == "").mean())
        rows.append({"field": c, "type": str(df[c].dtype).replace("object", "text"),
                     "description": desc, "source": src, "pct_missing": round(100 * miss, 1)})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
