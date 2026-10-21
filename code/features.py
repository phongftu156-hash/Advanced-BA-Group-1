"""Feature definitions shared by Module 1 (AVM), Module 2 (stale-listing risk) and the app."""
import numpy as np
import pandas as pd

DIST_LAYERS = ["metro", "bus_stop", "school", "university", "hospital", "park", "market", "water", "cbd"]
LOG_DIST = [f"ln_dist_{l}" for l in DIST_LAYERS]
DENSITY = ["n_bus_stop_500m", "n_school_1000m", "n_market_1000m"]
PHYS_NUM = ["ln_area", "bedrooms", "bathrooms", "floors", "frontage_m", "road_width_m"]
MISS_FLAGS = ["floors_missing", "frontage_m_missing", "road_width_m_missing", "bedrooms_missing",
              "bathrooms_missing"]
TEXT_FLAGS = ["kw_car_access", "kw_corner", "kw_business", "kw_lake_view", "kw_near_school",
              "kw_furnished"]
CAT = ["property_type", "city", "legal_group"]

# plain-language names for charts aimed at managers
NICE = {
    "ln_area": "Floor / land area (log)", "bedrooms": "Bedrooms", "bathrooms": "Bathrooms",
    "floors": "Storeys", "frontage_m": "Frontage width (m)", "road_width_m": "Access road width (m)",
    "lat": "Latitude", "lon": "Longitude", "accessibility_index": "Accessibility index (OSM)",
    "ln_dist_metro": "Distance to metro", "ln_dist_bus_stop": "Distance to bus stop",
    "ln_dist_school": "Distance to school", "ln_dist_university": "Distance to university",
    "ln_dist_hospital": "Distance to hospital/clinic", "ln_dist_park": "Distance to park",
    "ln_dist_market": "Distance to market/mall", "ln_dist_water": "Distance to lake/river",
    "ln_dist_cbd": "Distance to city centre", "n_bus_stop_500m": "Bus stops within 500 m",
    "n_school_1000m": "Schools within 1 km", "n_market_1000m": "Markets within 1 km",
    "kw_car_access": "Car access (text)", "kw_corner": "Corner lot (text)",
    "kw_business": "Business use (text)", "kw_lake_view": "Lake/river view (text)",
    "kw_near_school": "Near school (text)", "kw_furnished": "Furnished (text)",
    "kw_owner": "Owner-listed (text)", "kw_urgent": "Urgent sale wording (text)",
    "floors_missing": "Storeys not stated", "frontage_m_missing": "Frontage not stated",
    "road_width_m_missing": "Road width not stated", "bedrooms_missing": "Bedrooms not stated",
    "bathrooms_missing": "Bathrooms not stated", "city_TP. Hồ Chí Minh": "City: HCMC",
    "avm_resid": "AVM overpricing gap (out-of-fold)", "rel_ppm2_district": "Price/m² vs district median",
    "rel_price_district_type": "Price vs district-type median", "district_liquidity": "District median age (others)",
    "ln_district_volume": "District listing volume (log)", "n_images": "Number of photos",
    "ln_desc_len": "Description length (log)", "broker_profile": "Broker profile",
    "agent_tenure_months": "Poster account age (months)", "ln_price": "Asking price (log)",
}


def nice(c):
    if c in NICE:
        return NICE[c]
    from config import TYPE_EN
    if c.startswith("property_type_"):
        return "Type: " + TYPE_EN.get(c[len("property_type_"):], c)
    if c.startswith("legal_group_"):
        return "Legal: " + {"Sổ đỏ/Sổ hồng": "certificate", "Hợp đồng mua bán": "sale contract",
                            "Đang chờ sổ": "certificate pending", "Khác/không rõ": "other/unknown"}.get(
            c[len("legal_group_"):], c)
    for pre, lab in (("property_type_", "Type: "), ("legal_group_", "Legal: "), ("district_", "District: "),
                     ("city_", "City: ")):
        if c.startswith(pre):
            return lab + c[len(pre):]
    return c


def add_log_dist(df):
    for l in DIST_LAYERS:
        df[f"ln_dist_{l}"] = np.log(df[f"dist_{l}_km"] + 0.1)
    df["district_key"] = df["city"].map({"Hà Nội": "HN", "TP. Hồ Chí Minh": "HCM"}) + " | " + df["district"]
    return df


def tree_matrix(df, columns=None):
    """Design matrix for XGBoost: numeric features + one-hot categoricals + district + lat/lon."""
    num = PHYS_NUM + MISS_FLAGS + TEXT_FLAGS + LOG_DIST + DENSITY + ["accessibility_index", "lat", "lon"]
    X = pd.concat([df[num].astype(float),
                   pd.get_dummies(df[CAT + ["district_key"]].rename(columns={"district_key": "district"}),
                                  dtype=float)], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X
