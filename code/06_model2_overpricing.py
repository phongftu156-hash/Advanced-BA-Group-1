"""Module 2: stale-listing risk model (overpricing classifier).

Label: stale = 1 if the listing has been on the site longer than the threshold (default 90 days).
Primary age measure: listing age implied by the listing ID (creation order), calibrated to
dates with photo-upload dates (see 02_clean.py and DECISIONS.md D13): the posted date shown on
mogi.vn is refreshed when a listing is pushed, so it measures time since the last push.
The posted-date label is reported as a robustness check.

No leakage: listing age, posted/updated/photo dates, listing ID, search page and sample part
are excluded. The AVM residual is out-of-fold (Module 1). District liquidity and relative-price
features are computed from training rows only (leave-one-out inside the training set).
"""
import os
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (BLUE, LGREY, MGREY, MODELS, ORANGE, PROCESSED, SEED, TAB, W1, W2,  # noqa: E402
                    get_logger, savefig, set_style)
from features import add_log_dist, nice  # noqa: E402

log = get_logger("06_model2_overpricing")
set_style()
RULE_CUT = np.log(1.15)
N_ITER = int(os.environ.get("STALE_N_ITER", 30))  # debug runs may lower this; reported runs use 30
NUM = ["avm_resid", "rel_ppm2_district", "rel_price_district_type", "district_liquidity",
       "ln_district_volume", "ln_area", "bedrooms", "n_images", "ln_desc_len", "accessibility_index",
       "broker_profile", "kw_car_access", "kw_corner", "kw_business",
       "kw_lake_view", "kw_near_school", "kw_owner", "kw_urgent", "kw_furnished"]
CAT = ["property_type", "city", "legal_group"]
EXCLUDED = ["listing_age_days", "age_id_days", "age_created_days", "posted", "posted_date",
            "updated_date", "card_date", "img_first_date", "img_last_date", "listing_id",
            "search_page", "sample_part",
            "agent_tenure_months"]  # account age bounds listing age mechanically (an account cannot post before it exists)


def choose_label(age):
    share = (age > 90).mean()
    log.info(f"positive share at 90 days: {share:.1%}")
    if 0.08 <= share <= 0.70:
        return 90, share
    shares = {t: (age > t).mean() for t in (60, 90, 120, 180)}
    t = min(shares, key=lambda k: abs(shares[k] - 0.25))
    log.info(f"share outside [8%, 70%] -> threshold switched to {t} days (shares: "
             + ", ".join(f"{k}d={v:.1%}" for k, v in shares.items()) + ")")
    return t, shares[t]


def group_features(df, is_tr, age_col):
    """Relative-price and district-liquidity features from training rows only."""
    tr = df[is_tr]
    out = pd.DataFrame(index=df.index)
    med_ppm2 = tr.groupby("district_key")["price_per_m2"].median()
    city_ppm2 = tr.groupby("city")["price_per_m2"].median()
    out["rel_ppm2_district"] = df["price_per_m2"] / df["district_key"].map(med_ppm2).fillna(
        df["city"].map(city_ppm2))
    dt = tr.groupby(["district_key", "property_type"])["price_vnd"].median()
    ct = tr.groupby(["city", "property_type"])["price_vnd"].median()
    k = pd.MultiIndex.from_frame(df[["district_key", "property_type"]])
    base = pd.Series(dt.reindex(k).to_numpy(), index=df.index)
    base = base.fillna(pd.Series(ct.reindex(pd.MultiIndex.from_frame(df[["city", "property_type"]])).to_numpy(),
                                 index=df.index))
    out["rel_price_district_type"] = df["price_vnd"] / base
    # district liquidity: median age of OTHER training listings in the district
    city_med = tr.groupby("city")[age_col].median()
    liq = pd.Series(np.nan, index=df.index)
    for d, g in df.groupby("district_key"):
        trg = tr.loc[tr["district_key"] == d, age_col]
        for i in g.index:
            others = trg.drop(i, errors="ignore")
            liq[i] = others.median() if len(others) >= 5 else np.nan
    out["district_liquidity"] = liq.fillna(df["city"].map(city_med))
    vol = tr.groupby("district_key").size()
    out["ln_district_volume"] = np.log1p(df["district_key"].map(vol).fillna(0))
    return out


def design(df):
    X = pd.concat([df[NUM].astype(float), pd.get_dummies(df[CAT], drop_first=True, dtype=float)], axis=1)
    return X


def boot_auc(y, s, n=1000):
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def tune_logit(X, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1, 3, 10):
        aucs = []
        for a, b in skf.split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
            m.fit(X.iloc[a], y[a])
            aucs.append(roc_auc_score(y[b], m.predict_proba(X.iloc[b])[:, 1]))
        if best is None or np.mean(aucs) > best[1]:
            best = (C, np.mean(aucs))
    log.info(f"logistic: best C={best[0]} CV AUC={best[1]:.3f}")
    return best[0]


def xgb_params(rng):
    return {"max_depth": int(rng.choice([2, 3, 4, 5, 6])),
            "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.1])),
            "subsample": float(rng.uniform(0.6, 1.0)), "colsample_bytree": float(rng.uniform(0.4, 1.0)),
            "min_child_weight": float(rng.choice([1, 4, 8, 16, 32])),
            "reg_lambda": float(10 ** rng.uniform(-1, 1.5)), "gamma": float(rng.choice([0, 0.1, 0.5, 1]))}


def fit_xgbc(p, X, y, spw, Xv=None, yv=None, n_est=2000):
    m = xgb.XGBClassifier(n_estimators=n_est, tree_method="hist", random_state=SEED, n_jobs=4,
                          scale_pos_weight=spw, eval_metric="auc",
                          early_stopping_rounds=100 if Xv is not None else None, **p)
    if Xv is not None:
        m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    else:
        m.fit(X, y, verbose=False)
    return m


def tune_xgbc(X, y, spw, n_iter=30):
    rng = np.random.default_rng(SEED)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res = []
    for i in range(n_iter):
        p = xgb_params(rng)
        aucs, its = [], []
        for a, b in skf.split(X, y):
            m = fit_xgbc(p, X.iloc[a], y[a], spw, X.iloc[b], y[b])
            its.append(m.best_iteration)
            aucs.append(roc_auc_score(y[b], m.predict_proba(X.iloc[b])[:, 1]))
        res.append({**p, "cv_auc": float(np.mean(aucs)), "n_estimators": int(np.mean(its) * 1.1) + 1})
    res = pd.DataFrame(res).sort_values("cv_auc", ascending=False)
    res.to_csv(TAB / "xgb_stale_search.csv", index=False)
    best = res.iloc[0].to_dict()
    log.info(f"xgboost classifier: best CV AUC={best['cv_auc']:.3f}")
    return ({k: (int(best[k]) if k == "max_depth" else float(best[k])) for k in xgb_params(np.random.default_rng(0))},
            int(best["n_estimators"]))


def oof_proba(make, X, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    o = np.zeros(len(y))
    for a, b in skf.split(X, y):
        o[b] = make().fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]
    return o


def run_label(df, age_col, threshold, tag, full=False):
    """Fit rule / logistic / XGBoost for one label definition. Returns metrics (and objects if full)."""
    y_all = (df[age_col] > threshold).astype(int).to_numpy()
    idx_tr, idx_te = train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=y_all)
    is_tr = df.index.isin(idx_tr)
    idx_tr, idx_te = df.index[is_tr], df.index[~is_tr]  # keep row order aligned with y_all
    d = df.join(group_features(df, is_tr, age_col))
    X = design(d)
    Xtr, Xte = X.loc[idx_tr], X.loc[idx_te]
    ytr, yte = y_all[is_tr], y_all[~is_tr]
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    C = tune_logit(Xtr, ytr)
    logit = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(Xtr, ytr)
    p, n_est = tune_xgbc(Xtr, ytr, spw, n_iter=N_ITER if full else min(12, N_ITER))
    xgbc = fit_xgbc(p, Xtr, ytr, spw, n_est=n_est)
    scores = {"Rule: AVM gap > +15%": d.loc[idx_te, "avm_resid"].to_numpy(),
              "Logistic regression": logit.predict_proba(Xte)[:, 1],
              "XGBoost": xgbc.predict_proba(Xte)[:, 1]}
    rows = []
    for name, s in scores.items():
        lo, hi = boot_auc(yte, s, 1000 if full else 200)
        r = {"label": tag, "threshold_days": threshold, "model": name, "pos_share": yte.mean(),
             "AUC": roc_auc_score(yte, s), "AUC_lo": lo, "AUC_hi": hi,
             "PR_AUC": average_precision_score(yte, s), "N_test": len(yte)}
        if name.startswith("Rule"):
            pred = (s > RULE_CUT).astype(int)
            r["Brier"] = np.nan
        else:
            r["Brier"] = brier_score_loss(yte, s)
        rows.append(r)
    res = pd.DataFrame(rows)
    if not full:
        return res
    return res, dict(d=d, X=X, idx_tr=idx_tr, idx_te=idx_te, ytr=ytr, yte=yte, logit=logit, xgbc=xgbc,
                     scores=scores, p=p, n_est=n_est, spw=spw, C=C)


def main():
    t0 = time.time()
    df = add_log_dist(pd.read_parquet(PROCESSED / "listings_model.parquet")).reset_index(drop=True)
    avm = pd.read_csv(MODELS / "avm_predictions.csv")
    df = df.merge(avm[["listing_id", "avm_resid"]], on="listing_id", how="left")
    log.info(f"rows {len(df)}; AVM residual available for {df['avm_resid'].notna().mean():.1%}")
    log.info(f"excluded (leakage): {', '.join(EXCLUDED)}")
    age = "age_id_days"
    df = df[df[age].notna()].reset_index(drop=True)
    thr, share = choose_label(df[age])

    res, o = run_label(df, age, thr, "Creation age (listing ID)", full=True)
    d, yte, ytr, scores = o["d"], o["yte"], o["ytr"], o["scores"]
    log.info("test results:\n" + res.round(3).to_string(index=False))

    # threshold tuned on training out-of-fold probabilities (max F1), applied to the test set
    Xtr = o["X"].loc[o["idx_tr"]]
    oof_x = oof_proba(lambda: xgb.XGBClassifier(n_estimators=o["n_est"], tree_method="hist",
                                                random_state=SEED, n_jobs=4, scale_pos_weight=o["spw"],
                                                **o["p"]), Xtr, ytr)
    pr, rc, th = precision_recall_curve(ytr, oof_x)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    t_star = float(th[np.argmax(f1[:-1])])
    cm = confusion_matrix(yte, (scores["XGBoost"] >= t_star).astype(int))
    tn, fp, fn, tp = cm.ravel()
    res.loc[res["model"] == "XGBoost", "threshold"] = t_star
    res.loc[res["model"] == "XGBoost", ["TN", "FP", "FN", "TP"]] = [tn, fp, fn, tp]
    res.loc[res["model"] == "XGBoost", "precision"] = tp / max(tp + fp, 1)
    res.loc[res["model"] == "XGBoost", "recall"] = tp / max(tp + fn, 1)
    rule_pred = (scores["Rule: AVM gap > +15%"] > RULE_CUT).astype(int)
    rtn, rfp, rfn, rtp = confusion_matrix(yte, rule_pred).ravel()
    res.loc[res["model"].str.startswith("Rule"), ["threshold", "TN", "FP", "FN", "TP"]] = \
        [RULE_CUT, rtn, rfp, rfn, rtp]
    res.loc[res["model"].str.startswith("Rule"), "precision"] = rtp / max(rtp + rfp, 1)
    res.loc[res["model"].str.startswith("Rule"), "recall"] = rtp / max(rtp + rfn, 1)
    # top-decile capture (lift)
    for name, s in scores.items():
        k = int(np.ceil(0.1 * len(s)))
        top = np.argsort(-s)[:k]
        res.loc[res["model"] == name, "top_decile_capture"] = yte[top].sum() / yte.sum()
        res.loc[res["model"] == name, "top_decile_precision"] = yte[top].mean()
    res.to_csv(TAB / "stale_model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"threshold_days": thr, "positive_share": share, "xgb_threshold": t_star,
                   "logit_C": o["C"], "scale_pos_weight": o["spw"], **o["p"],
                   "n_estimators": o["n_est"]}]).to_csv(TAB / "stale_model_params.csv", index=False)

    # cost-based thresholds (illustrative costs; FP = 1 unit)
    grid = np.linspace(0.02, 0.98, 97)
    cost_rows = []
    for ratio in (1, 3, 5):
        for t in grid:
            p_tr = (oof_x >= t).astype(int)
            p_te = (scores["XGBoost"] >= t).astype(int)
            c_tr = ((p_tr == 1) & (ytr == 0)).sum() + ratio * ((p_tr == 0) & (ytr == 1)).sum()
            c_te = ((p_te == 1) & (yte == 0)).sum() + ratio * ((p_te == 0) & (yte == 1)).sum()
            cost_rows.append({"fn_cost_ratio": ratio, "threshold": t, "cost_per_listing_train": c_tr / len(ytr),
                              "cost_per_listing_test": c_te / len(yte),
                              "flag_rate_test": p_te.mean()})
    cost = pd.DataFrame(cost_rows)
    cost.to_csv(TAB / "stale_cost_curves.csv", index=False)
    opt = cost.loc[cost.groupby("fn_cost_ratio")["cost_per_listing_train"].idxmin()]
    opt.to_csv(TAB / "stale_cost_optimal.csv", index=False)
    log.info("cost-optimal thresholds (chosen on training OOF):\n" + opt.round(3).to_string(index=False))

    # robustness: alternative thresholds and the posted-date label
    rob = [res[["label", "threshold_days", "model", "pos_share", "AUC", "AUC_lo", "AUC_hi", "PR_AUC"]]]
    for t in (60, 90, 120, 180):
        if t != thr:
            rob.append(run_label(df, age, t, "Creation age (listing ID)")[
                ["label", "threshold_days", "model", "pos_share", "AUC", "AUC_lo", "AUC_hi", "PR_AUC"]])
    for t in (60, 90):
        dd = df[df["listing_age_days"].notna()].reset_index(drop=True)
        rob.append(run_label(dd, "listing_age_days", t, "Posted date (refreshed)")[
            ["label", "threshold_days", "model", "pos_share", "AUC", "AUC_lo", "AUC_hi", "PR_AUC"]])
    rob = pd.concat(rob, ignore_index=True)
    rob.to_csv(TAB / "stale_robustness.csv", index=False, encoding="utf-8-sig")
    log.info("robustness:\n" + rob.round(3).to_string(index=False))

    # scores for every listing: training rows out-of-fold, test rows from the final model
    out = pd.DataFrame({"listing_id": df["listing_id"], "split": "train", "stale_label": 0})
    out.loc[o["idx_te"], "split"] = "test"
    out.loc[o["idx_tr"], "p_stale"] = oof_x
    out.loc[o["idx_te"], "p_stale"] = scores["XGBoost"]
    out.loc[o["idx_tr"], "stale_label"] = ytr
    out.loc[o["idx_te"], "stale_label"] = yte
    # operational flag: top quartile of risk scores (review capacity of one listing in four),
    # cut-off taken from the training out-of-fold score distribution
    q75 = float(np.quantile(oof_x, 0.75))
    out["high_risk"] = (out["p_stale"] >= q75).astype(int)
    pd.DataFrame([{"high_risk_cutoff": q75}]).to_csv(TAB / "stale_high_risk_cutoff.csv", index=False)
    out["avm_resid"] = d["avm_resid"]
    out.to_csv(MODELS / "overpricing_scores.csv", index=False)
    joblib.dump({"model": o["xgbc"], "columns": list(o["X"].columns), "threshold": t_star, "high_risk_cutoff": q75,
                 "label_days": thr, "logit": o["logit"]}, MODELS / "xgb_stale.joblib")
    # group-level lookups for the app (training rows only)
    d.loc[o["idx_tr"]].groupby("district_key").agg(
        district_liquidity=("district_liquidity", "median"), ln_district_volume=("ln_district_volume", "first"),
        med_ppm2=("price_per_m2", "median")).to_csv(MODELS / "district_lookup.csv", encoding="utf-8-sig")

    figures(res, scores, yte, cost, opt, o)
    log.info(f"done in {time.time() - t0:.0f} s")


def figures(res, scores, yte, cost, opt, o):
    colors = {"Rule: AVM gap > +15%": MGREY, "Logistic regression": LGREY, "XGBoost": BLUE}
    colors["Logistic regression"] = "#8FB8DE"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 6.6 / 2.54))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(yte, s)
        auc = res.set_index("model").loc[name, "AUC"]
        a1.plot(fpr, tpr, color=colors[name], lw=1.6 if name == "XGBoost" else 1.0,
                label=f"{name.split(':')[0]} (AUC {auc:.2f})")
        pr, rc, _ = precision_recall_curve(yte, s)
        ap = res.set_index("model").loc[name, "PR_AUC"]
        a2.plot(rc, pr, color=colors[name], lw=1.6 if name == "XGBoost" else 1.0,
                label=f"{name.split(':')[0]} (PR-AUC {ap:.2f})")
    a1.plot([0, 1], [0, 1], color=LGREY, lw=0.6, ls="--")
    a2.axhline(yte.mean(), color=LGREY, lw=0.6, ls="--")
    a2.text(0.98, yte.mean() - 0.02, f"Base rate {yte.mean():.0%}", ha="right", va="top", fontsize=7, color=MGREY)
    a1.set_xlabel("False positive rate")
    a1.set_ylabel("True positive rate")
    a2.set_xlabel("Recall")
    a2.set_ylabel("Precision")
    a1.set_title("ROC curve")
    a2.set_title("Precision-recall curve")
    for a in (a1, a2):
        a.legend(loc="lower right" if a is a1 else "lower left", fontsize=6.8, handlelength=1.2)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1.02)
    fig.tight_layout()
    savefig(fig, "stale_roc_pr")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 6.2 / 2.54))
    for name, s in scores.items():
        order = np.argsort(-s)
        gain = np.cumsum(yte[order]) / yte.sum()
        x = np.arange(1, len(s) + 1) / len(s)
        a1.plot(x, gain, color=colors[name], lw=1.6 if name == "XGBoost" else 1.0, label=name.split(":")[0])
    a1.plot([0, 1], [0, 1], color=LGREY, ls="--", lw=0.6)
    cap = res.set_index("model").loc["XGBoost", "top_decile_capture"]
    a1.axvline(0.1, color=ORANGE, lw=0.6)
    a1.text(0.98, 0.03, f"Top 10% of scores capture\n{cap:.0%} of stale listings", fontsize=6.8, color=ORANGE,
            ha="right", va="bottom", transform=a1.transAxes)
    a1.set_xlabel("Share of listings reviewed (highest risk first)")
    a1.set_ylabel("Share of stale listings captured")
    a1.set_title("Cumulative gains")
    a1.legend(loc="upper left", fontsize=6.8)
    shades = {1: LGREY, 3: "#8FB8DE", 5: BLUE}
    for ratio, g in cost.groupby("fn_cost_ratio"):
        a2.plot(g["threshold"], g["cost_per_listing_test"], color=shades[ratio], lw=1.3)
        r = opt[opt["fn_cost_ratio"] == ratio].iloc[0]
        yv = g.loc[(g["threshold"] - r["threshold"]).abs().idxmin(), "cost_per_listing_test"]
        a2.plot(r["threshold"], yv, "o", color=ORANGE, ms=3.5)
        a2.text(1.0, g["cost_per_listing_test"].iloc[-1], f" 1:{ratio}", fontsize=6.8,
                color=shades[ratio] if ratio > 1 else MGREY, ha="left", va="center")
    a2.set_xlim(0, 1.12)
    a2.set_xlabel("Probability threshold for flagging")
    a2.set_ylabel("Expected cost per listing (FP = 1)")
    a2.set_title("Threshold vs illustrative cost (labels: FP:FN cost ratio)")
    fig.tight_layout()
    savefig(fig, "stale_gains_cost")

    fig, ax = plt.subplots(figsize=(W1, W1 * 0.8))
    for name in ("Logistic regression", "XGBoost"):
        fr, mp = calibration_curve(yte, scores[name], n_bins=10, strategy="quantile")
        ax.plot(mp, fr, "o-", ms=2.5, lw=1, color=colors[name], label=name)
    ax.plot([0, 1], [0, 1], ls="--", color=LGREY, lw=0.6)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed stale share")
    ax.legend(fontsize=6.8)
    fig.tight_layout()
    savefig(fig, "stale_calibration")

    import shap
    Xte = o["X"].loc[o["idx_te"]]
    sv = shap.TreeExplainer(o["xgbc"]).shap_values(Xte)
    imp = pd.Series(np.abs(sv).mean(0), index=Xte.columns).sort_values(ascending=False)
    imp.to_csv(TAB / "stale_shap_importance.csv", header=["mean_abs_shap"])
    top = imp.head(12).index
    fig = plt.figure(figsize=(W1, 8.2 / 2.54))
    shap.summary_plot(sv[:, [Xte.columns.get_loc(c) for c in top]], Xte[top], feature_names=[nice(c) for c in top],
                      show=False, plot_size=None, max_display=12, cmap=plt.get_cmap("coolwarm"))
    ax = plt.gca()
    ax.set_xlabel("SHAP value (impact on log-odds of stale)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    for t in fig.axes[-1].get_yticklabels():
        t.set_fontsize(6.5)
    fig.axes[-1].set_ylabel("Feature value", fontsize=7)
    savefig(fig, "stale_shap_summary")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
