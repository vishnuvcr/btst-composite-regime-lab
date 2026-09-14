from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from btst_composite import add_candidate_scores, add_features, build_folds, load_config
from btst_composite_btst import load_universe, proxy_market


def strict_net(row, cfg):
    sl = float(cfg["costs"].get("slippage_bps_per_side", 5)) / 10000
    tc = float(cfg["costs"].get("transaction_cost_bps_per_side", 8)) / 10000
    sm = float(cfg["execution"].get("stop_atr_mult", 1.5))
    tm = float(cfg["execution"].get("target_atr_mult", 2.0))
    entry = float(row.close) * (1 + sl)
    atr = float(np.clip(row.atr_pct, .005, .20))
    stop = entry * (1 - sm * atr)
    target = entry * (1 + tm * atr)
    if row.next_low <= stop:
        exit_px, reason = stop, "stop"
    elif row.next_high >= target:
        exit_px, reason = target, "target"
    else:
        exit_px, reason = float(row.next_open), "next_open"
    exit_px *= 1 - sl
    return exit_px / entry - 1 - 2 * tc, reason


def execute(picks, cfg, label):
    if picks.empty:
        return pd.DataFrame()
    max_pos = int(cfg["portfolio"].get("max_positions", 10))
    gross = float(cfg["portfolio"].get("max_gross_exposure", .95))
    cap = float(cfg["portfolio"].get("max_position_weight", 1.0))
    rows = []
    for date, g in picks.groupby("date"):
        g = g.nlargest(max_pos, "selection_score").copy()
        n = len(g)
        if not n:
            continue
        weight = min(cap, gross / n)
        for _, r in g.iterrows():
            if not np.isfinite(r.next_open):
                continue
            net, reason = strict_net(r, cfg)
            rows.append({"date": date, "symbol": r.symbol, "method": label, "family": r.get("family", label),
                         "net_return": net, "weight": weight, "weighted_return": net * weight, "reason": reason})
    return pd.DataFrame(rows)


def method_metrics(trades):
    if trades.empty:
        return {"method": "", "trades": 0, "total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
                "win_rate": 0.0, "profit_factor": 0.0, "best_month": 0.0, "worst_month": 0.0,
                "months_ge_10pct": 0, "months_ge_20pct": 0, "months_ge_30pct": 0}
    daily = trades.groupby("date").weighted_return.sum().sort_index()
    eq = (1 + daily).cumprod()
    dd = eq / eq.cummax() - 1
    monthly = ((1 + daily).groupby(daily.index.to_period("M")).prod() - 1)
    loss = abs(trades.loc[trades.net_return <= 0, "net_return"].sum())
    pf = trades.loc[trades.net_return > 0, "net_return"].sum() / loss if loss else float("inf")
    return {"method": str(trades.method.iloc[0]), "trades": int(len(trades)),
            "total_return": float(eq.iloc[-1] - 1), "max_drawdown": float(dd.min()),
            "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() else 0.0,
            "win_rate": float((trades.net_return > 0).mean()), "profit_factor": float(pf),
            "best_month": float(monthly.max()), "worst_month": float(monthly.min()),
            "months_ge_10pct": int((monthly >= .10).sum()), "months_ge_20pct": int((monthly >= .20).sum()),
            "months_ge_30pct": int((monthly >= .30).sum())}


def main(config_path):
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
    x = add_candidate_scores(add_features(df, market), families)
    x["btst_return"] = x.next_open / x.close - 1
    folds = build_folds(pd.DatetimeIndex(sorted(x.date.unique())), cfg)
    family_trades = {f: [] for f in families}
    ensemble_trades, oracle_trades = [], []

    for fold in folds:
        test = x[(x.date >= fold.test_start) & (x.date <= fold.test_end)].copy()
        if test.empty:
            continue
        for fam in families:
            z = test.copy()
            z["family"] = fam
            z["selection_score"] = z[f"score_{fam}"]
            z = z[z.selection_score >= .60]
            t = execute(z, cfg, fam)
            if not t.empty:
                family_trades[fam].append(t)

        # Equal-weight signal ensemble: average the 11 bounded family scores per stock/date.
        score_cols = [f"score_{f}" for f in families]
        ens = test.copy()
        ens["selection_score"] = ens[score_cols].mean(axis=1)
        ens["family"] = "equal_weight_ensemble"
        t = execute(ens[ens.selection_score >= .60], cfg, "equal_weight_ensemble")
        if not t.empty:
            ensemble_trades.append(t)

        # Oracle is deliberately look-ahead and is only an upper-bound diagnostic.
        oracle_rows = []
        for fam in families:
            z = test.copy()
            z["family"] = fam
            z["selection_score"] = z[f"score_{fam}"]
            z = z[z.selection_score >= .60].copy()
            if z.empty:
                continue
            vals = z.apply(lambda r: strict_net(r, cfg)[0], axis=1)
            z["realized_net"] = vals
            oracle_rows.append(z)
        if oracle_rows:
            oz = pd.concat(oracle_rows, ignore_index=True)
            # One stock can appear under several families; keep its best realized setup.
            oz = oz.sort_values("realized_net", ascending=False).drop_duplicates(["date", "symbol"], keep="first")
            oz["selection_score"] = oz["realized_net"]
            t = execute(oz, cfg, "oracle_upper_bound")
            if not t.empty:
                oracle_trades.append(t)

    out = Path("docs")
    out.mkdir(exist_ok=True)
    combined = []
    for fam, chunks in family_trades.items():
        if chunks:
            combined.append(pd.concat(chunks, ignore_index=True))
    if ensemble_trades:
        combined.append(pd.concat(ensemble_trades, ignore_index=True))
    if oracle_trades:
        combined.append(pd.concat(oracle_trades, ignore_index=True))
    all_trades = pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()
    if not all_trades.empty:
        all_trades.to_csv(out / "btst_benchmark_trades.csv", index=False)
        monthly = []
        for method, g in all_trades.groupby("method"):
            daily = g.groupby("date").weighted_return.sum().sort_index()
            m = ((1 + daily).groupby(daily.index.to_period("M")).prod() - 1).rename("monthly_return").reset_index()
            m["method"] = method
            monthly.append(m)
        pd.concat(monthly, ignore_index=True).to_csv(out / "btst_benchmark_monthly.csv", index=False)
    metrics_rows = [method_metrics(pd.concat(v, ignore_index=True)) for v in family_trades.values() if v]
    if ensemble_trades:
        metrics_rows.append(method_metrics(pd.concat(ensemble_trades, ignore_index=True)))
    if oracle_trades:
        metrics_rows.append(method_metrics(pd.concat(oracle_trades, ignore_index=True)))
    pd.DataFrame(metrics_rows).sort_values("total_return", ascending=False).to_csv(out / "btst_benchmark_metrics.csv", index=False)
    print(pd.DataFrame(metrics_rows).sort_values("total_return", ascending=False).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/composite.yaml")
    main(ap.parse_args().config)
