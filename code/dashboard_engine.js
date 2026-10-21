/* Vietnam Real Estate AVM: in-browser valuation engine.
 * Evaluates the exported XGBoost trees and reproduces the Streamlit app's preprocessing exactly.
 * Used by the standalone HTML dashboard and by the node parity test (code/17_export_dashboard.py).
 *
 * Tree format (one flat array per tree, node id order): internal node = [feature, threshold, yes, missingIsYes],
 * leaf = [-1, value]; "no" child = yes + 1 (checked at export). XGBoost sends a row to "yes" when
 * float32(x) < threshold and to the "missing" child when x is NaN.
 */
(function (root) {
  "use strict";
  var f32 = Math.fround;

  function unpack(tree) {
    // tree: flat list of node records concatenated; record length 2 (leaf) or 4 (split), prefixed by node id order
    var nodes = [], i = 0;
    while (i < tree.length) {
      if (tree[i] === -1) { nodes.push({ leaf: tree[i + 1] }); i += 2; }
      else { nodes.push({ f: tree[i], t: f32(tree[i + 1]), y: tree[i + 2], m: tree[i + 3] }); i += 4; }
    }
    return nodes;
  }

  function Booster(spec) {
    this.cols = spec.cols;
    this.index = {};
    for (var k = 0; k < spec.cols.length; k++) this.index[spec.cols[k]] = k;
    this.base = spec.base;          // regression: base score; classifier: base margin (logit of base_score)
    this.logistic = spec.objective === "binary:logistic";
    this.trees = spec.trees.map(unpack);
  }
  Booster.prototype.margin = function (x) {
    // XGBoost accumulates the base margin and the leaf values in float32, tree by tree; mirror it exactly
    var s = f32(this.base);
    for (var t = 0; t < this.trees.length; t++) {
      var nodes = this.trees[t], n = nodes[0], id = 0;
      while (n.leaf === undefined) {
        var v = x[n.f];
        if (v !== v) id = n.m ? n.y : n.y + 1;          // NaN: default direction
        else id = f32(v) < n.t ? n.y : n.y + 1;
        n = nodes[id];
      }
      s = f32(s + f32(n.leaf));
    }
    return s;
  };
  Booster.prototype.predict = function (x) {
    var m = this.margin(x);
    return this.logistic ? f32(1 / (1 + Math.exp(-m))) : m;
  };
  Booster.prototype.vector = function (obj) {      // obj: {column name: value}; absent columns are 0 (one-hot)
    var x = new Array(this.cols.length);
    for (var k = 0; k < this.cols.length; k++) {
      var v = obj[this.cols[k]];
      x[k] = v === undefined ? 0 : v;
    }
    return x;
  };

  var DIST = ["metro", "bus_stop", "school", "university", "hospital", "park", "market", "water", "cbd"];
  var CITY_CODE = { "Hà Nội": "HN", "TP. Hồ Chí Minh": "HCM" };

  function Engine(D) {
    this.D = D;
    this.avm = new Booster(D.avm);
    this.stale = new Booster(D.stale);
  }

  // location features: ward median when the ward has at least 3 listings, else the district median
  Engine.prototype.location = function (city, district, ward) {
    var d = this.D.loc[city + "|" + district];
    if (ward && d.wards[ward]) return { v: d.wards[ward], level: "ward" };
    return { v: d.district, level: "district" };
  };

  Engine.prototype.value = function (inp) {
    var D = this.D, loc = this.location(inp.city, inp.district, inp.ward), L = loc.v;
    var dk = CITY_CODE[inp.city] + " | " + inp.district;
    var o = {
      ln_area: Math.log(inp.area), bedrooms: inp.bedrooms, bathrooms: inp.bathrooms, floors: inp.floors,
      frontage_m: inp.frontage, road_width_m: inp.road,
      floors_missing: inp.floors_missing || 0, frontage_m_missing: inp.frontage_missing || 0,
      road_width_m_missing: inp.road_missing || 0, bedrooms_missing: 0, bathrooms_missing: 0,
      kw_car_access: 0, kw_corner: 0, kw_business: 0, kw_lake_view: 0, kw_near_school: 0, kw_furnished: 0,
      n_bus_stop_500m: L.n_bus_stop_500m, n_school_1000m: L.n_school_1000m, n_market_1000m: L.n_market_1000m,
      accessibility_index: L.accessibility_index, lat: L.lat, lon: L.lon
    };
    DIST.forEach(function (l) { o["ln_dist_" + l] = Math.log(L["dist_" + l + "_km"] + 0.1); });
    o["property_type_" + inp.type] = 1;
    o["city_" + inp.city] = 1;
    o["legal_group_" + inp.legal] = 1;
    o["district_" + dk] = 1;
    var pred = this.avm.predict(this.avm.vector(o));
    var q = D.conf.q_type[inp.type] !== undefined ? D.conf.q_type[inp.type] : D.conf.q_global;
    var res = { pred: pred, fv: Math.exp(pred) / 1e9, lo: Math.exp(pred - q) / 1e9, hi: Math.exp(pred + q) / 1e9,
                level: loc.level, loc: L, dk: dk };
    if (inp.ask > 0) {
      var gap = Math.log(inp.ask * 1e9) - pred;
      var lk = D.lookup[dk] || D.lookupMedian;
      var dt = D.dtMed[dk + "|" + inp.type];
      if (dt === undefined || dt === null) dt = D.ctMed[inp.city + "|" + inp.type];
      var s = {
        avm_resid: gap, rel_ppm2_district: inp.ask * 1e9 / inp.area / lk.med_ppm2,
        rel_price_district_type: inp.ask * 1e9 / dt, district_liquidity: lk.district_liquidity,
        ln_district_volume: lk.ln_district_volume, ln_area: Math.log(inp.area), bedrooms: inp.bedrooms,
        n_images: D.nImagesMed, ln_desc_len: D.lnDescMed, accessibility_index: L.accessibility_index,
        broker_profile: 1, kw_car_access: 0, kw_corner: 0, kw_business: 0, kw_lake_view: 0, kw_near_school: 0,
        kw_owner: 0, kw_urgent: 0, kw_furnished: 0
      };
      s["property_type_" + inp.type] = 1;
      s["city_" + inp.city] = 1;
      s["legal_group_" + inp.legal] = 1;
      var p = this.stale.predict(this.stale.vector(s));
      res.gap = Math.exp(gap) - 1;
      res.p = p;
      res.light = p >= D.staleCut ? "High" : (p >= D.staleGreen ? "Medium" : "Low");
    }
    return res;
  };

  // five nearest comparable listings of the same city and type (same weights as the Streamlit app)
  Engine.prototype.comparables = function (inp, L, k) {
    var T = this.D.listings, idx = [], i;
    for (i = 0; i < T.city.length; i++)
      if (T.cities[T.city[i]] === inp.city && T.types[T.type[i]] === inp.type) idx.push(i);
    var keys = ["ln_area", "bedrooms", "bathrooms", "lat", "lon"], w = [1.0, 0.5, 0.3, 1.5, 1.5];
    var target = [Math.log(inp.area), inp.bedrooms, inp.bathrooms, L.lat, L.lon];
    var mu = [], sd = [];
    keys.forEach(function (key, j) {
      var a = idx.map(function (r) { return T[key][r]; });
      var m = a.reduce(function (x, y) { return x + y; }, 0) / a.length;
      var v = a.reduce(function (x, y) { return x + (y - m) * (y - m); }, 0) / (a.length - 1);
      mu.push(m); sd.push(Math.sqrt(v) || 1);
    });
    var scored = idx.map(function (r) {
      var d = 0;
      keys.forEach(function (key, j) { var z = (T[key][r] - target[j]) / sd[j]; d += z * z * w[j]; });
      return [Math.sqrt(d), r];
    });
    scored.sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
    return scored.slice(0, k || 5).map(function (s) { return s[1]; });
  };

  root.AVMEngine = { Engine: Engine, Booster: Booster };
  if (typeof module !== "undefined") module.exports = root.AVMEngine;
})(typeof window !== "undefined" ? window : globalThis);
