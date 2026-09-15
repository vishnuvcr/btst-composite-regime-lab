from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from btst_composite import FEATURES, add_candidate_scores, add_features, build_folds, load_config, make_regime_labels, metrics
from btst_composite_btst import load_universe, proxy_market, strict_net

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


def fit_rank_model(train: pd.DataFrame, families: list[str], seed: int):
    rows = []
    for fam in families:
        z = train.dropna(subset=FEATURES + ["btst_return", f"score_{fam}"]).copy()
        if z.empty:
            continue
        z["family"] = fam
        rows.append(z)
    if not rows:
        return None, None
    d = pd.concat(rows, ignore_index=True)
    fam_cols = [f"family_{f}" for f in families]
    for f in families:
        d[f"family_{f}"] = (d.family == f).astype(float)

    regime_values = sorted(pd.Series(d["regime"]).dropna().unique().tolist())
    regime_cols = [f"regime_{int(r)}" for r in regime_values]
    for r in regime_values:
        d[f"regime_{int(r)}"] = (d.regime == r).astype(float)

    cols = FEATURES + [f"score_{f}" for f in families] + fam_cols + regime_cols
    X = d[cols].astype(float).fillna(0)
    y = d.btst_return.clip(-.25, .25).astype(float)
    if LGBMRegressor is not None:
        model = LGBMRegressor(
            n_estimators=260, learning_rate=.025, num_leaves=15, max_depth=5,
            min_child_samples=80, subsample=.8, colsample_bytree=.8,
            reg_lambda=2.0, random_state=seed, verbosity=-1,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=260, learning_rate=.035, max_leaf_nodes=15,
            min_samples_leaf=80, l2_regularization=2.0, random_state=seed,
        )
    model.fit(X, y)
    return model, cols


def predict_family_rows(model, cols, frame: pd.DataFrame, families: list[str]):
    parts = []
    execution_cols = [
        "date", "symbol", "regime", "family", "candidate_score", "predicted_return",
        "next_open", "btst_return",
    ]
    regime_cols = [c for c in cols if c.startswith("regime_")]
    for fam in families:
        z = frame.copy()
        z["family"] = fam
        for f in families:
            z[f"family_{f}"] = (f == fam)
        for rc in regime_cols:
            rid = int(rc.split("_")[-1])
            z[rc] = (z["regime"] == rid).astype(float)
        z["predicted_return"] = model.predict(z[cols].astype(float).fillna(0))
        z["candidate_score"] = z[f"score_{fam}"]
        parts.append(z[execution_cols])
    return pd.concat(parts, ignore_index=True)


def tune_thresholds(val: pd.DataFrame, families: list[str], max_positions: int):
    if val.empty:
        return {"min_pred": 0.0, "min_score": .60, "max_per_regime": 1}

    # Never use a negative predicted edge as a reason to trade.  The old tuner
    # could select a negative threshold, forcing trades even when the model
    # expected a loss.  Thresholds are selected only from non-negative model
    # predictions and scored on executable validation returns.
    preds = val.predicted_return.dropna()
    preds = preds[preds >= 0]
    if preds.empty:
        return {"min_pred": np.inf, "min_score": .60, "max_per_regime": 1}

    candidates = np.unique(np.quantile(preds, [0.50, .60, .70, .80, .90, .95]))
    best = None
    for thr in candidates:
        z = val[(val.candidate_score >= .60) & (val.predicted_return >= thr)].copy()
        if z.empty:
            continue
        z = z.sort_values(["date", "symbol", "predicted_return"], ascending=[True, True, False])
        z = z.drop_duplicates(["date", "symbol"])
        z = select_top_by_date(z, max_positions)
        daily = z.groupby("date").btst_return.mean()
        if daily.empty:
            continue
        score = float(daily.mean() - 0.5 * daily.std())
        coverage = len(z) / max(1, len(val.date.unique()) * max_positions)
        score -= max(0.0, 0.01 - coverage) * 0.2
        if best is None or score > best[0]:
            best = (score, float(thr))
    return {"min_pred": best[1] if best else np.inf, "min_score": .60, "max_per_regime": 1}


def select_top_by_date(frame: pd.DataFrame, max_positions: int) -> pd.DataFrame:
    """Select top predicted setups per date while preserving all columns."""
    if frame.empty:
        return frame.copy()
    required = {"date", "predicted_return"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"select_top_by_date missing columns: {sorted(missing)}")
    parts = []
    for _, group in frame.groupby("date", sort=False):
        parts.append(group.nlargest(max_positions, "predicted_return"))
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()


def execute(selected: pd.DataFrame, cfg: dict):
    if selected.empty:
        return pd.DataFrame()
    max_pos = int(cfg["portfolio"].get("max_positions", 10))
    gross = float(cfg["portfolio"].get("max_gross_exposure", .95))
    cap = float(cfg["portfolio"].get("max_position_weight", .15))
    rows = []
    for date, g in selected.groupby("date"):
        g = g.nlargest(max_pos, "predicted_return")
        n = len(g)
        if n == 0:
            continue
        weight = min(cap, gross / n)
        for _, r in g.iterrows():
            if not np.isfinite(r["next_open"]):
                continue
            net = float(r["btst_return"])
            if not np.isfinite(net):
                continue
            rows.append({
                "date": date, "symbol": r["symbol"], "family": r["family"], "regime": r["regime"],
                "candidate_score": r["candidate_score"], "predicted_return": r["predicted_return"],
                "net_return": net, "reason": "precomputed_strict_net", "weight": weight,
                "weighted_return": net * weight,
            })
    return pd.DataFrame(rows)


def run(config_path: str):
    cfg = load_config(config_path)
    dc = cfg["data"]
    families = cfg["strategies"]["enabled"]
    df = load_universe(dc["daily_glob"], dc.get("max_symbols"))
    df = df[(df.date >= pd.Timestamp(dc["start_date"])) & (df.date <= pd.Timestamp(dc["end_date"]))]
    print(f"Loaded {len(df):,} rows across {df.symbol.nunique():,} symbols, {df.date.min().date()} to {df.date.max().date()}")
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
    all_trades, pred_rows, tuning_rows = [], [], []

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

        model, cols = fit_rank_model(train, families, seed + fid)
        if model is None:
            continue
        vp = predict_family_rows(model, cols, val, families)
        params = tune_thresholds(vp, families, int(cfg["portfolio"].get("max_positions", 10)))
        tuning_rows.append({"fold": fid, **params})

        tp = predict_family_rows(model, cols, test, families)
        tp = tp[(tp.candidate_score >= params["min_score"]) & (tp.predicted_return >= params["min_pred"])].copy()
        tp = tp.sort_values(["date", "symbol", "predicted_return"], ascending=[True, True, False])
        tp = tp.drop_duplicates(["date", "symbol"])
        if not tp.empty:
            tp = select_top_by_date(tp, int(cfg["portfolio"].get("max_positions", 10)))
            pred_rows.append(tp[["date", "symbol", "regime", "family", "candidate_score", "predicted_return"]])
        t = execute(tp, cfg)
        if not t.empty:
            t["fold"] = fid
            all_trades.append(t)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    out = Path("docs")
    out.mkdir(exist_ok=True)
    if not trades.empty:
        trades.to_csv(out / "btst_strategy_ranking_v2_trades.csv", index=False)
        daily = trades.groupby("date").weighted_return.sum().sort_index()
        monthly = ((1 + daily).groupby(daily.index.to_period("M")).prod() - 1).rename("monthly_return").reset_index()
        monthly.to_csv(out / "btst_strategy_ranking_v2_monthly.csv", index=False)
    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_csv(out / "btst_strategy_ranking_v2_predictions.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(out / "btst_strategy_ranking_v2_tuning.csv", index=False)
    pd.DataFrame([metrics(trades)]).to_csv(out / "btst_strategy_ranking_v2_metrics.csv", index=False)
    print(metrics(trades))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/composite.yaml")
    run(ap.parse_args().config)
