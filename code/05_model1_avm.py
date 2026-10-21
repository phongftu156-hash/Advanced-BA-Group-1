"""Module 1: hedonic AVM (OLS specification ladder) and XGBoost AVM.

Outputs (result/):
  tables/hedonic_ladder.tex/.csv     specification ladder (clustered SE by district)
  tables/avm_metrics.csv             OLS(4) vs XGBoost on the test set, overall
  tables/avm_metrics_by_segment.csv  same by city x type
  tables/conformal_coverage.csv      90% split-conformal interval coverage
  tables/xgb_avm_params.csv          tuned hyperparameters
  models/xgb_avm.joblib, models/ols_avm.joblib, models/avm_predictions.csv (OOF / test)
  figures/avm_performance, avm_pred_vs_actual, avm_shap_summary, avm_shap_dependence
"""
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (BLUE, DGREY, LGREY, MGREY, MODELS, ORANGE, PROCESSED, SEED, TAB, W1, W2,  # noqa: E402
                    get_logger, savefig, set_style)
from features import DENSITY, LOG_DIST, add_log_dist, nice, tree_matrix  # noqa: E402

log = get_logger("05_model1_avm")
warnings.filterwarnings("ignore", message=".*rank-deficient.*")
warnings.filterwarnings("ignore", message=".*covariance of the constraints.*")
N_ITER = int(os.environ.get("AVM_N_ITER", 40))  # debug runs may lower this; reported runs use 40
set_style()

PHYS = ("ln_area + bedrooms + bathrooms + floors + frontage_m + road_width_m + floors_missing"
        " + frontage_m_missing + road_width_m_missing + C(legal_group, Treatment('Sổ đỏ/Sổ hồng'))"
        " + C(property_type, Treatment('Căn hộ chung cư')) + C(city)")
SPATIAL = " + ".join(LOG_DIST + DENSITY)
SPECS = {
    "(1) Physical": f"ln_price ~ {PHYS}",
    "(2) + District FE": f"ln_price ~ {PHYS} + C(district_key)",
    "(3) + OSM spatial": f"ln_price ~ {PHYS} + C(district_key) + {SPATIAL}",
    "(4) + Interactions": (f"ln_price ~ {PHYS} + C(district_key) + {SPATIAL}"
                           " + C(property_type):C(city) + C(property_type):ln_area"),
}
SHOW = {
    "ln_area": "Log area", "bedrooms": "Bedrooms", "bathrooms": "Bathrooms", "floors": "Storeys",
    "frontage_m": "Frontage (m)", "road_width_m": "Access road width (m)",
    "C(legal_group, Treatment('Sổ đỏ/Sổ hồng'))[T.Hợp đồng mua bán]": "Legal: sale contract",
    "C(legal_group, Treatment('Sổ đỏ/Sổ hồng'))[T.Đang chờ sổ]": "Legal: certificate pending",
    "C(legal_group, Treatment('Sổ đỏ/Sổ hồng'))[T.Khác/không rõ]": "Legal: other/unknown",
    "ln_dist_cbd": "Log distance to CBD", "ln_dist_metro": "Log distance to metro",
    "ln_dist_school": "Log distance to school", "ln_dist_hospital": "Log distance to hospital",
    "ln_dist_park": "Log distance to park", "ln_dist_market": "Log distance to market/mall",
    "ln_dist_water": "Log distance to lake/river", "n_bus_stop_500m": "Bus stops within 500 m",
}


# --------------------------------------------------------------------------- metrics
def avm_metrics(y_log, p_log):
    y, p = np.exp(y_log), np.exp(p_log)
    ape = np.abs(p / y - 1)
    ratio = p / y
    med = np.median(ratio)
    return {
        "N": len(y), "RMSE_log": float(np.sqrt(np.mean((p_log - y_log) ** 2))),
        "MAE_log": float(np.mean(np.abs(p_log - y_log))), "MAPE": float(ape.mean()),
        "MdAPE": float(np.median(ape)), "PPE10": float((ape <= 0.10).mean()),
        "PPE20": float((ape <= 0.20).mean()), "COD": float(100 * np.mean(np.abs(ratio - med)) / med),
        "PRD": float(ratio.mean() / (p.sum() / y.sum())),
    }


def cv_rmse_ols(formula, df, k=5):
    kf = KFold(k, shuffle=True, random_state=SEED)
    errs = []
    for tr, va in kf.split(df):
        m = smf.ols(formula, df.iloc[tr]).fit()
        pv = predict_ols(m, df.iloc[va])
        ok = ~np.isnan(pv)
        errs.append((df["ln_price"].to_numpy()[va][ok] - pv[ok]) ** 2)
    return float(np.sqrt(np.concatenate(errs).mean()))


def predict_ols(m, d):
    """Predict, falling back on the reference district for districts unseen in training."""
    d = d.copy()
    seen = set(m.model.data.frame["district_key"]) if "district_key" in m.model.data.frame else None
    if seen is not None:
        ref = m.model.data.frame["district_key"].mode()[0]
        d.loc[~d["district_key"].isin(seen), "district_key"] = ref
    return np.asarray(m.predict(d), dtype=float)


# --------------------------------------------------------------------------- ladder
def ladder(df):
    groups = df["district_key"].astype("category").cat.codes
    rows, fits = [], {}
    for name, f in SPECS.items():
        m = smf.ols(f, df).fit(cov_type="cluster", cov_kwds={"groups": groups})
        fits[name] = m
        rows.append({"spec": name, "N": int(m.nobs), "R2": m.rsquared, "adj_R2": m.rsquared_adj,
                     "CV_RMSE_log": cv_rmse_ols(f, df)})
        log.info(f"{name}: R2={m.rsquared:.3f} adjR2={m.rsquared_adj:.3f} CV RMSE={rows[-1]['CV_RMSE_log']:.3f}")
    # multi-view check: what does the second dataset add without district identities?
    f_alt = f"ln_price ~ {PHYS} + {SPATIAL}"
    m_alt = smf.ols(f_alt, df).fit()
    rows.append({"spec": "(1) + OSM spatial, no district FE", "N": int(m_alt.nobs), "R2": m_alt.rsquared,
                 "adj_R2": m_alt.rsquared_adj, "CV_RMSE_log": cv_rmse_ols(f_alt, df)})
    log.info(f"(1) + OSM, no FE: R2={m_alt.rsquared:.3f} CV RMSE={rows[-1]['CV_RMSE_log']:.3f}")
    stats = pd.DataFrame(rows)
    stats.to_csv(TAB / "hedonic_ladder_stats.csv", index=False)
    stats = stats.iloc[:4]
    coef = []
    for var, lab in SHOW.items():
        r = {"variable": lab}
        for name, m in fits.items():
            if var in m.params.index:
                b, p, se = m.params[var], m.pvalues[var], m.bse[var]
                r[name] = b
                r[name + "_se"] = se
                r[name + "_p"] = p
        coef.append(r)
    coef = pd.DataFrame(coef)
    coef.to_csv(TAB / "hedonic_ladder.csv", index=False, encoding="utf-8-sig")
    write_ladder_tex(coef, stats)
    return fits, stats


def stars(p):
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.1 else ""


def write_ladder_tex(coef, stats):
    names = list(SPECS)
    L = ["\\begin{tabular}{l" + "c" * len(names) + "}", "\\toprule",
         " & " + " & ".join(f"\\textbf{{{n.split(')')[0]})}}" for n in names) + " \\\\",
         " & " + " & ".join(n.split(") ")[1] for n in names) + " \\\\", "\\midrule"]
    for _, r in coef.iterrows():
        if all(pd.isna(r.get(n)) for n in names):
            continue  # category absent from the data (e.g. no 'certificate pending' listings)
        cells, ses = [], []
        for n in names:
            if pd.notna(r.get(n)):
                cells.append(f"${r[n]:.3f}{stars(r[n + '_p'])}$")
                ses.append(f"\\footnotesize({r[n + '_se']:.3f})")
            else:
                cells.append("")
                ses.append("")
        L.append(r["variable"] + " & " + " & ".join(cells) + " \\\\")
        L.append(" & " + " & ".join(ses) + " \\\\[-1pt]")
    L.append("\\midrule")
    L.append("District fixed effects & No & Yes & Yes & Yes \\\\")
    L.append("Type $\\times$ city, type $\\times$ log area & No & No & No & Yes \\\\")
    L.append("N & " + " & ".join(f"{int(v):,}" for v in stats["N"]) + " \\\\")
    L.append("R$^2$ & " + " & ".join(f"{v:.3f}" for v in stats["R2"]) + " \\\\")
    L.append("Adjusted R$^2$ & " + " & ".join(f"{v:.3f}" for v in stats["adj_R2"]) + " \\\\")
    L.append("5-fold CV RMSE (log) & " + " & ".join(f"{v:.3f}" for v in stats["CV_RMSE_log"]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "hedonic_ladder.tex").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- xgboost
def sample_params(rng):
    return {
        "max_depth": int(rng.choice([3, 4, 5, 6, 7, 8])),
        "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.08, 0.1])),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.4, 1.0)),
        "min_child_weight": float(rng.choice([1, 2, 4, 8, 16])),
        "reg_lambda": float(10 ** rng.uniform(-1, 1.5)),
        "reg_alpha": float(10 ** rng.uniform(-3, 0.5)),
        "gamma": float(rng.choice([0, 0, 0.01, 0.05])),
    }


def fit_xgb(params, X, y, Xv=None, yv=None, n_estimators=3000):
    m = xgb.XGBRegressor(n_estimators=n_estimators, tree_method="hist", random_state=SEED, n_jobs=4,
                         early_stopping_rounds=100 if Xv is not None else None, **params)
    if Xv is not None:
        m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    else:
        m.fit(X, y, verbose=False)
    return m


def tune_xgb(X, y, n_iter=40):
    rng = np.random.default_rng(SEED)
    kf = KFold(5, shuffle=True, random_state=SEED)
    res = []
    for i in range(n_iter):
        p = sample_params(rng)
        rmse, iters = [], []
        for tr, va in kf.split(X):
            m = fit_xgb(p, X.iloc[tr], y[tr], X.iloc[va], y[va])
            iters.append(m.best_iteration)
            rmse.append(np.sqrt(np.mean((m.predict(X.iloc[va]) - y[va]) ** 2)))
        res.append({**p, "cv_rmse": float(np.mean(rmse)), "n_estimators": int(np.mean(iters) * 1.1) + 1})
        log.info(f"  config {i + 1}/{n_iter}: CV RMSE {res[-1]['cv_rmse']:.4f}")
    res = pd.DataFrame(res).sort_values("cv_rmse")
    res.to_csv(TAB / "xgb_avm_search.csv", index=False)
    best = res.iloc[0].to_dict()
    return best


def oof_predictions(params, n_est, X, y):
    kf = KFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, va in kf.split(X):
        m = fit_xgb(params, X.iloc[tr], y[tr], n_estimators=n_est)
        oof[va] = m.predict(X.iloc[va])
    return oof


# --------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    df = add_log_dist(pd.read_parquet(PROCESSED / "listings_model.parquet")).reset_index(drop=True)
    log.info(f"rows: {len(df)}")
    fits, stats = ladder(df)

    strata = df["city"] + "|" + df["property_type"]
    idx_tr, idx_te = train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=strata)
    tr, te = df.loc[idx_tr], df.loc[idx_te]

    # OLS (4) on the training set, evaluated on the test set
    ols = smf.ols(SPECS["(4) + Interactions"], tr).fit()
    p_ols = predict_ols(ols, te)
    joblib.dump({"formula": SPECS["(4) + Interactions"], "params": ols.params}, MODELS / "ols_avm.joblib")

    # XGBoost
    X = tree_matrix(df)
    Xtr, Xte = X.loc[idx_tr], X.loc[idx_te]
    ytr, yte = tr["ln_price"].to_numpy(), te["ln_price"].to_numpy()
    best = tune_xgb(Xtr, ytr, N_ITER)
    params = {k: (int(best[k]) if k == "max_depth" else float(best[k]))
              for k in sample_params(np.random.default_rng(0))}
    n_est = int(best["n_estimators"])
    pd.DataFrame([{**params, "n_estimators": n_est, "cv_rmse": best["cv_rmse"]}]).to_csv(
        TAB / "xgb_avm_params.csv", index=False)
    model = fit_xgb(params, Xtr, ytr, n_estimators=n_est)
    p_xgb = model.predict(Xte)
    oof = oof_predictions(params, n_est, Xtr, ytr)
    joblib.dump({"model": model, "columns": list(X.columns)}, MODELS / "xgb_avm.joblib")

    # OLS out-of-fold predictions on the training set (for the conformal and residual comparison)
    oof_ols = np.full(len(tr), np.nan)
    for a, b in KFold(5, shuffle=True, random_state=SEED).split(tr):
        m = smf.ols(SPECS["(4) + Interactions"], tr.iloc[a]).fit()
        oof_ols[b] = predict_ols(m, tr.iloc[b])

    # metrics
    rows = [{"model": "OLS (4)", **avm_metrics(yte, p_ols)}, {"model": "XGBoost", **avm_metrics(yte, p_xgb)}]
    met = pd.DataFrame(rows)
    met.to_csv(TAB / "avm_metrics.csv", index=False)
    log.info("test metrics:\n" + met.round(3).to_string(index=False))
    seg = []
    for (c, t), g in te.assign(p_ols=p_ols, p_xgb=p_xgb).groupby(["city", "property_type"]):
        for name, col in (("OLS (4)", "p_ols"), ("XGBoost", "p_xgb")):
            seg.append({"city": c, "property_type": t, "model": name,
                        **avm_metrics(g["ln_price"].to_numpy(), g[col].to_numpy())})
    for t, g in te.assign(p_ols=p_ols, p_xgb=p_xgb).groupby("property_type"):
        for name, col in (("OLS (4)", "p_ols"), ("XGBoost", "p_xgb")):
            seg.append({"city": "Both", "property_type": t, "model": name,
                        **avm_metrics(g["ln_price"].to_numpy(), g[col].to_numpy())})
    seg = pd.DataFrame(seg)
    seg.to_csv(TAB / "avm_metrics_by_segment.csv", index=False, encoding="utf-8-sig")
    preferred = "XGBoost" if met.set_index("model").loc["XGBoost", "RMSE_log"] <= \
        met.set_index("model").loc["OLS (4)", "RMSE_log"] else "OLS (4)"
    log.info(f"preferred AVM by test RMSE: {preferred}")

    # split conformal 90%: global and by property type (Mondrian), calibrated on OOF residuals
    r_oof = np.abs(ytr - oof)
    n = len(r_oof)
    q_global = float(np.quantile(r_oof, min(1, np.ceil((n + 1) * 0.9) / n)))
    q_type = {}
    for t, g in pd.DataFrame({"t": tr["property_type"].to_numpy(), "r": r_oof}).groupby("t"):
        k = len(g)
        q_type[t] = float(np.quantile(g["r"], min(1, np.ceil((k + 1) * 0.9) / k)))
    cover_g = np.abs(yte - p_xgb) <= q_global
    qt = te["property_type"].map(q_type).to_numpy()
    cover_t = np.abs(yte - p_xgb) <= qt
    cov = [{"interval": "Global", "property_type": "All", "q_log": q_global,
            "width_pct": float(np.exp(q_global) - np.exp(-q_global)), "coverage": cover_g.mean()},
           {"interval": "By type (Mondrian)", "property_type": "All", "q_log": np.nan,
            "width_pct": np.nan, "coverage": cover_t.mean()}]
    for t in q_type:
        m = (te["property_type"] == t).to_numpy()
        cov.append({"interval": "By type (Mondrian)", "property_type": t, "q_log": q_type[t],
                    "width_pct": float(np.exp(q_type[t]) - np.exp(-q_type[t])), "coverage": cover_t[m].mean()})
    pd.DataFrame(cov).to_csv(TAB / "conformal_coverage.csv", index=False, encoding="utf-8-sig")
    joblib.dump({"q_global": q_global, "q_type": q_type}, MODELS / "conformal.joblib")
    log.info(f"conformal coverage: global {cover_g.mean():.3f}, by type {cover_t.mean():.3f}")

    # per-listing predictions: train rows out-of-fold, test rows from the final model
    pred = pd.DataFrame({"listing_id": df["listing_id"], "split": "train"})
    pred.loc[idx_te, "split"] = "test"
    pred.loc[idx_tr, "avm_pred_ln"] = oof
    pred.loc[idx_te, "avm_pred_ln"] = p_xgb
    pred.loc[idx_tr, "ols_pred_ln"] = oof_ols
    pred.loc[idx_te, "ols_pred_ln"] = p_ols
    pred["ln_price"] = df["ln_price"]
    pred["avm_resid"] = pred["ln_price"] - pred["avm_pred_ln"]
    q = df["property_type"].map(q_type)
    pred["avm_fair_value_bn"] = np.exp(pred["avm_pred_ln"]) / 1e9
    pred["avm_lo90_bn"] = np.exp(pred["avm_pred_ln"] - q) / 1e9
    pred["avm_hi90_bn"] = np.exp(pred["avm_pred_ln"] + q) / 1e9
    pred.to_csv(MODELS / "avm_predictions.csv", index=False)

    figures(met, te, yte, p_xgb, model, Xte, stats)
    log.info(f"done in {time.time() - t0:.0f} s")


def figures(met, te, yte, p_xgb, model, Xte, stats):
    # Exhibit: performance comparison (4 panels: RMSE, MdAPE, PPE10, PPE20)
    fig, axes = plt.subplots(1, 4, figsize=(W2, 4.6 / 2.54))
    panels = [("RMSE_log", "RMSE (log price)", False), ("MdAPE", "Median abs. % error", True),
              ("PPE10", "Within ±10% (PPE10)", True), ("PPE20", "Within ±20% (PPE20)", True)]
    for ax, (k, lab, pct) in zip(axes, panels):
        v = met.set_index("model")[k]
        cols = [LGREY if m == "OLS (4)" else BLUE for m in v.index]
        bars = ax.bar(range(len(v)), v.values, color=cols, width=0.6)
        for b, val in zip(bars, v.values):
            ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.0%}" if pct else f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(range(len(v)), ["OLS (4)", "XGBoost"])
        ax.set_title(lab, fontsize=8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0, v.max() * 1.22)
    fig.tight_layout(w_pad=1.2)
    savefig(fig, "avm_performance")

    # Exhibit: predicted vs actual (hexbin, log scale) with PPE10 band
    fig, ax = plt.subplots(figsize=(W1, W1 * 0.95))
    y, p = np.exp(yte) / 1e9, np.exp(p_xgb) / 1e9
    hb = ax.hexbin(y, p, gridsize=38, xscale="log", yscale="log", cmap="Blues", mincnt=1, linewidths=0.1)
    lo, hi = min(y.min(), p.min()) * 0.8, max(y.max(), p.max()) * 1.2
    xs = np.geomspace(lo, hi, 50)
    ax.fill_between(xs, xs * 0.9, xs * 1.1, color=ORANGE, alpha=0.18, lw=0)
    ax.plot(xs, xs, color=ORANGE, lw=0.8)
    ax.text(0.04, 0.95, "Shaded band: within ±10%", transform=ax.transAxes, fontsize=7, color=ORANGE, va="top")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    ax.set_xlabel("Asking price (VND bn, log scale)")
    ax.set_ylabel("XGBoost estimate (VND bn, log scale)")
    cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Listings", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    savefig(fig, "avm_pred_vs_actual")

    # SHAP
    import shap
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(Xte)
    imp = pd.Series(np.abs(sv).mean(0), index=Xte.columns)
    # aggregate one-hot district dummies into one feature for readability
    dist_cols = [c for c in Xte.columns if c.startswith("district_")]
    imp_agg = imp.drop(dist_cols)
    imp_agg["District (as listed)"] = imp[dist_cols].sum()
    imp_agg.sort_values(ascending=False).to_csv(TAB / "avm_shap_importance.csv", header=["mean_abs_shap"])
    top = imp.drop(dist_cols).sort_values(ascending=False).head(12).index
    fig = plt.figure(figsize=(W1, 8.2 / 2.54))
    shap.summary_plot(sv[:, [Xte.columns.get_loc(c) for c in top]], Xte[top],
                      feature_names=[nice(c) for c in top], show=False, plot_size=None,
                      max_display=12, color_bar_label="Feature value", cmap=plt.get_cmap("coolwarm"))
    ax = plt.gca()
    ax.set_xlabel("SHAP value (impact on log asking price)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    for t in fig.axes[-1].get_yticklabels():
        t.set_fontsize(6.5)
    fig.axes[-1].set_ylabel("Feature value", fontsize=7)
    savefig(fig, "avm_shap_summary")

    fig, axes = plt.subplots(2, 1, figsize=(W1, 9.0 / 2.54), sharey=True)
    for ax, c, lab in ((axes[0], "ln_dist_cbd", "Distance to city centre (km)"),
                       (axes[1], "ln_dist_metro", "Distance to metro station (km)")):
        x = np.exp(Xte[c].to_numpy()) - 0.1
        s = sv[:, Xte.columns.get_loc(c)]
        hcm = Xte["city_TP. Hồ Chí Minh"].to_numpy() == 1 if "city_TP. Hồ Chí Minh" in Xte else np.zeros(len(x), bool)
        ax.scatter(x[~hcm], s[~hcm], s=3, color=BLUE, alpha=0.35, lw=0, label="Hanoi")
        ax.scatter(x[hcm], s[hcm], s=3, color=ORANGE, alpha=0.35, lw=0, label="HCMC")
        ax.axhline(0, color=MGREY, lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].set_ylabel("SHAP (log price)")
    axes[1].set_ylabel("SHAP (log price)")
    axes[0].text(0.97, 0.95, "Hanoi", color=BLUE, transform=axes[0].transAxes, ha="right", va="top", fontsize=7.5)
    axes[0].text(0.97, 0.85, "HCMC", color=ORANGE, transform=axes[0].transAxes, ha="right", va="top", fontsize=7.5)
    fig.tight_layout()
    savefig(fig, "avm_shap_dependence")
    _ = DGREY


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
