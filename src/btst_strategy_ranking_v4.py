from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from btst_composite import FEATURES, add_candidate_scores, add_features, build_folds, load_config, make_regime_labels, metrics
from btst_composite_btst import load_universe, proxy_market, strict_net

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = LGBMRegressor = None


def archetype_features(x: pd.DataFrame) -> list[str]:
    """Continuous stock-state descriptors used to let the router distinguish archetypes."""
    cols = [
        "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60", "gap",
        "range_pct", "close_location", "body_pct", "atr_pct", "vol20",
        "volume_z", "sma20_gap", "sma50_gap", "mkt_ret_1", "mkt_ret_5",
        "mkt_vol20", "mkt_trend", "breadth", "rank_ret_5", "rank_ret_20",
        "rank_gap", "rank_close_location", "rank_volume_z", "rank_sma20_gap",
    ]
    return [c for c in cols if c in x.columns]


def build_design(train: pd.DataFrame, families: list[str]):
    dparts = []
    for fam in families:
        z = train.dropna(subset=["btst_return", f"score_{fam}"]).copy()
        if z.empty:
            continue
        z["family"] = fam
        dparts.append(z)
    if not dparts:
        return None, None
    d = pd.concat(dparts, ignore_index=True)
    base = archetype_features(d)
    for f in families:
        d[f"family_{f}"] = (d.family == f).astype(float)
    regime_values = sorted(pd.Series(d.regime).dropna().unique().tolist())
    for r in regime_values:
        d[f"regime_{int(r)}"] = (d.regime == r).astype(float)
    # Explicit family x regime interactions; tree models can then separate conditional edges.
    for f in families:
        for r in regime_values:
            d[f"fxr_{f}_{int(r)}"] = ((d.family == f) & (d.regime == r)).astype(float)
    cols = base + [f"score_{f}" for f in families]
    cols += [f"family_{f}" for f in families]
    cols += [f"regime_{int(r)}" for r in regime_values]
    cols += [f"fxr_{f}_{int(r)}" for f in families for r in regime_values]
    return d, cols


def fit_models(train: pd.DataFrame, families: list[str], seed: int):
    d, cols = build_design(train, families)
    if d is None:
        return None
    X = d[cols].astype(float).fillna(0)
    y = d.btst_return.astype(float)
    ybin = (y > 0).astype(int)
    if ybin.nunique() < 2:
        return None
    if LGBMClassifier is not None:
        clf = LGBMClassifier(
            n_estimators=220, learning_rate=.025, num_leaves=15, max_depth=5,
            min_child_samples=100, subsample=.8, colsample_bytree=.8,
            reg_lambda=3.0, random_state=seed, verbosity=-1,
        )
        reg = LGBMRegressor(
            n_estimators=220, learning_rate=.025, num_leaves=15, max_depth=5,
            min_child_samples=100, subsample=.8, colsample_bytree=.8,
            reg_lambda=3.0, random_state=seed + 1000, verbosity=-1,
        )
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=220, learning_rate=.035, max_leaf_nodes=15,
            min_samples_leaf=100, l2_regularization=3.0, random_state=seed,
        )
        reg = HistGradientBoostingRegressor(
            max_iter=220, learning_rate=.035, max_leaf_nodes=15,
            min_samples_leaf=100, l2_regularization=3.0, random_state=seed + 1000,
        )
    clf.fit(X, ybin)
    reg.fit(X, y.clip(-.25, .25))
    return {"classifier": clf, "regressor": reg, "cols": cols, "families": families}


def predict(model_pack, frame: pd.DataFrame) -> pd.DataFrame:
    clf, reg, cols, families = model_pack["classifier"], model_pack["regressor"], model_pack["cols"], model_pack["families"]
    parts = []
    regime_cols = [c for c in cols if c.startswith("regime_") and not c.startswith("regime_x")]
    interaction_cols = [c for c in cols if c.startswith("fxr_")]
    for fam in families:
        z = frame.copy()
        z["family"] = fam
        for f in families:
            z[f"family_{f}"] = float(f == fam)
        for rc in regime_cols:
            rid = int(rc.split("_")[-1])
            z[rc] = (z.regime == rid).astype(float)
        for ic in interaction_cols:
            bits = ic.split("_")
            rid = int(bits[-1])
            ff = "_".join(bits[1:-1])
            z[ic] = ((z.regime == rid) & (z.family == ff)).astype(float)
        X = z[cols].astype(float).fillna(0)
        z["p_positive"] = clf.predict_proba(X)[:, 1]
        z["predicted_return"] = reg.predict(X)
        # Conservative utility: require both probability and magnitude to agree.
        z["edge_score"] = z.p_positive * np.maximum(z.predicted_return, 0.0)
        z["candidate_score"] = z[f"score_{fam}"]
        parts.append(z[["date", "symbol", "regime", "family", "candidate_score", "p_positive", "predicted_return", "edge_score", "btst_return", "next_open"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def select_validation_policy(vp: pd.DataFrame, max_positions: int):
    """Choose thresholds using validation only; includes an explicit abstain/no-trade state."""
    if vp.empty:
        return {"min_prob": 1.0, "min_pred": np.inf, "min_score": .60}
    best = None
    probs = np.unique(np.quantile(vp.p_positive.dropna(), [0.55, .65, .75, .80, .85, .90, .95]))
    preds = np.unique(np.quantile(vp.predicted_return.dropna(), [0.50, .65, .80, .90]))
    for p in probs:
        for pr in preds:
            z = vp[(vp.candidate_score >= .60) & (vp.p_positive >= p) & (vp.predicted_return >= pr)].copy()
            if z.empty:
                continue
            z = z.sort_values(["date", "predicted_return"], ascending=[True, False]).drop_duplicates(["date", "symbol"])
            z = pd.concat([g.nlargest(max_positions, "edge_score") for _, g in z.groupby("date")], ignore_index=True)
            if z.empty:
                continue
            daily = z.groupby("date").btst_return.mean()
            if len(daily) < 20:
                continue
            # Favor positive mean with stability; do not reward high trade count by itself.
            score = float(daily.mean() - .50 * daily.std())
            positive_days = float((daily > 0).mean())
            score += .002 * positive_days
            if best is None or score > best[0]:
                best = (score, float(p), float(pr))
    if best is None or best[0] <= 0:
        return {"min_prob": 1.0, "min_pred": np.inf, "min_score": .60}
    return {"min_prob": best[1], "min_pred": best[2], "min_score": .60}


def execute(selected: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    max_pos = int(cfg["portfolio"].get("max_positions", 10))
    gross = float(cfg["portfolio"].get("max_gross_exposure", .95))
    cap = float(cfg["portfolio"].get("max_position_weight", .15))
    rows = []
    for date, g in selected.groupby("date"):
        g = g.nlargest(max_pos, "edge_score")
        n = len(g)
        if not n:
            continue
        weight = min(cap, gross / n)
        for _, r in g.iterrows():
            if not np.isfinite(r.next_open) or not np.isfinite(r.btst_return):
                continue
            rows.append({
                "date": date, "symbol": r.symbol, "family": r.family, "regime": r.regime,
                "candidate_score": r.candidate_score, "p_positive": r.p_positive,
                "predicted_return": r.predicted_return, "edge_score": r.edge_score,
                "net_return": r.btst_return, "weight": weight,
                "weighted_return": r.btst_return * weight,
            })
    return pd.DataFrame(rows)


def run(config_path: str):
    cfg = load_config(config_path)
    dc = cfg["data"]
    families = cfg["strategies"]["enabled"]
    df = load_universe(dc["daily_glob"], dc.get("max_symbols"))
    df = df[(df.date >= pd.Timestamp(dc["start_date"])) & (df.date <= pd.Timestamp(dc["end_date"]))]
    market = None
    if dc.get("market_glob"):
        from glob import glob
        if glob(dc["market_glob"], recursive=True):
            market = load_universe(dc["market_glob"], 1)
    if market is None or market.empty:
        market = proxy_market(df)
        print("Using cross-sectional proxy market index (no NIFTY 50 file supplied).")
    x = add_features(df, market)
    x["btst_return"] = x.apply(lambda r: strict_net(r, cfg)[0], axis=1)
    x = add_candidate_scores(x, families)
    folds = build_folds(pd.DatetimeIndex(sorted(x.date.unique())), cfg)
    seed = int(cfg["research"].get("random_state", 42))
    all_trades, predictions, tuning = [], [], []
    for fid, fold in enumerate(folds, 1):
        train = x[(x.date >= fold.train_start) & (x.date <= fold.train_end)].copy()
        val = x[(x.date >= fold.validation_start) & (x.date <= fold.validation_end)].copy()
        test = x[(x.date >= fold.test_start) & (x.date <= fold.test_end)].copy()
        if train.empty or val.empty or test.empty:
            continue
        trreg, _, _ = make_regime_labels(train, train, int(cfg["research"].get("n_clusters", 6)), seed)
        fwreg, _, _ = make_regime_labels(train, pd.concat([val, test]), int(cfg["research"].get("n_clusters", 6)), seed)
        train["regime"] = train.date.map(trreg)
        val["regime"] = val.date.map(fwreg)
        test["regime"] = test.date.map(fwreg)
        pack = fit_models(train, families, seed + fid)
        if pack is None:
            continue
        vp = predict(pack, val)
        policy = select_validation_policy(vp, int(cfg["portfolio"].get("max_positions", 10)))
        tuning.append({"fold": fid, **policy})
        tp = predict(pack, test)
        tp = tp[(tp.candidate_score >= policy["min_score"]) & (tp.p_positive >= policy["min_prob"]) & (tp.predicted_return >= policy["min_pred"])].copy()
        if not tp.empty:
            tp = tp.sort_values(["date", "edge_score"], ascending=[True, False]).drop_duplicates(["date", "symbol"])
            predictions.append(tp[["date", "symbol", "regime", "family", "candidate_score", "p_positive", "predicted_return", "edge_score"]])
        trades = execute(tp, cfg)
        if not trades.empty:
            trades["fold"] = fid
            all_trades.append(trades)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    out = Path("docs"); out.mkdir(exist_ok=True)
    if not trades.empty:
        trades.to_csv(out / "btst_strategy_ranking_v4_trades.csv", index=False)
        daily = trades.groupby("date").weighted_return.sum().sort_index()
        monthly = ((1 + daily).groupby(daily.index.to_period("M")).prod() - 1).rename("monthly_return").reset_index()
        monthly.to_csv(out / "btst_strategy_ranking_v4_monthly.csv", index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(out / "btst_strategy_ranking_v4_predictions.csv", index=False)
    pd.DataFrame(tuning).to_csv(out / "btst_strategy_ranking_v4_tuning.csv", index=False)
    result = metrics(trades)
    pd.DataFrame([result]).to_csv(out / "btst_strategy_ranking_v4_metrics.csv", index=False)
    print(result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config/composite.yaml"); run(ap.parse_args().config)
