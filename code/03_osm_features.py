"""Phase 3: second dataset, OpenStreetMap points of interest (Overpass API).

1. Download POI layers once per city bounding box (cached in data/external/osm/).
2. For each listing: haversine distance to the nearest POI of each layer, counts
   within 500 m and 1 km, distance to the CBD.
3. Accessibility index = first principal component of standardised inverted distances.
4. Merge into data/processed/listings_model.parquet / .csv.

Usage: python code/03_osm_features.py [--download-only]
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import requests
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import BBOX, CBD, OSM, PROCESSED, TAB, get_logger  # noqa: E402

log = get_logger("03_osm_features")
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter",
             "https://overpass.private.coffee/api/interpreter",
             "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
UA = {"User-Agent": "AdvancedBA-ResearchBot/1.0 (university course project, non-commercial)"}
EARTH_KM = 6371.0088

LAYERS = {
    "metro": """
      node["railway"="station"]["station"~"subway|light_rail|monorail"]({b});
      node["public_transport"="station"]["subway"="yes"]({b});
      node["public_transport"="station"]["light_rail"="yes"]({b});
      node["railway"="halt"]["light_rail"="yes"]({b});
      way["railway"="station"]["station"~"subway|light_rail|monorail"]({b});
    """,
    "bus_stop": 'node["highway"="bus_stop"]({b});',
    "school": 'nwr["amenity"~"^(school|kindergarten)$"]({b});',
    "university": 'nwr["amenity"="university"]({b});',
    "hospital": 'nwr["amenity"~"^(hospital|clinic)$"]({b});',
    "park": 'nwr["leisure"="park"]({b});',
    "market": 'nwr["amenity"="marketplace"]({b}); nwr["shop"~"^(mall|supermarket)$"]({b});',
    "water": 'way["natural"="water"]({b}); relation["natural"="water"]({b});',
}
SLUG = {"Hà Nội": "hanoi", "TP. Hồ Chí Minh": "hcmc"}


def overpass(query, attempts=12):
    for attempt in range(attempts):
        url = ENDPOINTS[0] if attempt < 2 else ENDPOINTS[attempt % len(ENDPOINTS)]
        try:
            r = requests.post(url, data={"data": query}, headers=UA, timeout=300)
            if r.status_code == 200:
                return r.json()
            log.info(f"  overpass {url} HTTP {r.status_code}; retrying")
        except requests.RequestException as e:
            log.info(f"  overpass {url} error {e}; retrying")
        time.sleep(min(15 * (attempt + 1), 90))
    raise RuntimeError(f"Overpass failed after {attempts} attempts on all endpoints")


def tiles(s, w, n, e, k):
    """Split a bounding box into k x k tiles."""
    la, lo = np.linspace(s, n, k + 1), np.linspace(w, e, k + 1)
    return [(la[i], lo[j], la[i + 1], lo[j + 1]) for i in range(k) for j in range(k)]


def download():
    """One query per city bounding box and layer; if the server times out, fall back to 2x2 tiles."""
    for city, box in BBOX.items():
        for layer, body in LAYERS.items():
            out = OSM / f"{SLUG[city]}_{layer}.json"
            if out.exists() and out.stat().st_size > 50:
                continue
            log.info(f"downloading {city} / {layer}")
            q = "[out:json][timeout:240];({body});out center tags;"
            try:
                data = overpass(q.format(body=body.format(b="{},{},{},{}".format(*box))), attempts=4)
            except RuntimeError:
                log.info(f"  full box failed: downloading {city} / {layer} in 2x2 tiles")
                els = {}
                for t in tiles(*box, 2):
                    part = overpass(q.format(body=body.format(b="{:.5f},{:.5f},{:.5f},{:.5f}".format(*t))))
                    for el in part.get("elements", []):
                        els[f"{el['type']}/{el['id']}"] = el
                    time.sleep(5)
                data = {"elements": list(els.values()), "note": "merged from 2x2 tiles"}
            out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            time.sleep(5)


def load_layer(city, layer):
    data = json.loads((OSM / f"{SLUG[city]}_{layer}.json").read_text(encoding="utf-8"))
    rows = []
    for el in data.get("elements", []):
        lat = el.get("lat", el.get("center", {}).get("lat"))
        lon = el.get("lon", el.get("center", {}).get("lon"))
        if lat is None:
            continue
        rows.append({"lat": lat, "lon": lon, "name": el.get("tags", {}).get("name", ""),
                     "osm_id": f"{el['type']}/{el['id']}"})
    df = pd.DataFrame(rows).drop_duplicates("osm_id")
    if layer == "metro" and len(df):
        # keep operating stations only: OSM also holds planned stations named by code
        # (e.g. 'S2.01', 'Ga S10', 'Ga C3'); unnamed nodes are dropped as well
        df = df[(df["name"] != "") & ~df["name"].str.match(r"^(Ga\s+)?[A-Z]\d")]
        # one point per station: merge nodes within 150 m sharing the same name
        df = df.assign(k=df["name"].where(df["name"] != "", df["osm_id"]))
        df = df.groupby("k", as_index=False).agg(lat=("lat", "mean"), lon=("lon", "mean"),
                                                 name=("name", "first"), osm_id=("osm_id", "first"))
    return df


def poi_summary():
    rows = []
    for city in BBOX:
        for layer in LAYERS:
            rows.append({"city": city, "layer": layer, "n_poi": len(load_layer(city, layer))})
    s = pd.DataFrame(rows)
    s.to_csv(TAB / "osm_poi_counts.csv", index=False, encoding="utf-8-sig")
    return s


def add_features(df):
    df = df.copy()
    feats = []
    for city in BBOX:
        idx = df.index[df["city"] == city]
        if len(idx) == 0:
            continue
        pts = np.radians(df.loc[idx, ["lat", "lon"]].to_numpy(float))
        out = pd.DataFrame(index=idx)
        for layer in LAYERS:
            poi = load_layer(city, layer)
            tree = BallTree(np.radians(poi[["lat", "lon"]].to_numpy(float)), metric="haversine")
            d, _ = tree.query(pts, k=1)
            out[f"dist_{layer}_km"] = d[:, 0] * EARTH_KM
            if layer in ("bus_stop", "school", "market"):
                for r_m in (500, 1000):
                    out[f"n_{layer}_{r_m}m"] = tree.query_radius(pts, r=r_m / 1000 / EARTH_KM,
                                                                count_only=True)
        c = np.radians(np.array([CBD[city]]))
        out["dist_cbd_km"] = BallTree(c, metric="haversine").query(pts, k=1)[0][:, 0] * EARTH_KM
        feats.append(out)
    return df.join(pd.concat(feats))


ACC_VARS = ["dist_metro_km", "dist_school_km", "dist_hospital_km", "dist_park_km",
            "dist_market_km", "dist_cbd_km"]


def accessibility(df):
    """First PC of standardised inverted distances, oriented so higher = more accessible.

    Inversion uses -log(distance + 0.1 km) so a few very remote listings do not dominate.
    Standardisation is within city so the index ranks locations inside each market.
    """
    X = -np.log(df[ACC_VARS].to_numpy(float) + 0.1)
    Z = np.zeros_like(X)
    for city in df["city"].unique():
        m = (df["city"] == city).to_numpy()
        Z[m] = StandardScaler().fit_transform(X[m])
    pca = PCA(n_components=1, random_state=42).fit(Z)
    load = pca.components_[0]
    sign = 1 if load.sum() >= 0 else -1
    score = sign * pca.transform(Z)[:, 0]
    loadings = pd.DataFrame({"variable": ACC_VARS, "loading": sign * load})
    loadings["explained_variance_ratio"] = pca.explained_variance_ratio_[0]
    loadings.to_csv(TAB / "accessibility_loadings.csv", index=False, encoding="utf-8-sig")
    return (score - score.mean()) / score.std(), loadings


def main(download_only=False):
    download()
    s = poi_summary()
    log.info("POI counts:\n" + s.pivot(index="layer", columns="city", values="n_poi").to_string())
    if download_only:
        return
    df = pd.read_parquet(PROCESSED / "listings_clean.parquet")
    n0 = len(df)
    geo = df["lat"].notna() & df["lon"].notna()
    out = add_features(df[geo])
    out["accessibility_index"], loadings = accessibility(out)
    log.info(f"merge: {len(out)} of {n0} listings with coordinates got OSM features "
             f"({len(out) / max(geo.sum(), 1):.1%} of geocoded rows)")
    log.info("accessibility loadings:\n" + loadings.to_string(index=False))
    out.to_parquet(PROCESSED / "listings_model.parquet", index=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-only", action="store_true")
    main(ap.parse_args().download_only)
