from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from btst_composite import (
    FEATURES, add_candidate_scores, add_features, build_folds, fit_family_models,
    load_config, make_regime_labels, metrics,
)
from btst_composite_regime import choose_regimes


def _norm(c):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(c).strip()).strip("_")


def _resolve_col(cols, candidates, contains=()):
    norm = {_norm(c): c for c in cols}
    for c in candidates:
        if c in norm:
            return norm[c]
    for c in cols:
        n = _norm(c)
        if any(n.startswith(x) or n.endswith(x) for x in contains):
            return c
    return None


def _read_robust(fp):
    try:
        x = pd.read_csv(fp, encoding="utf-8-sig")
    except Exception:
        try:
            x = pd.read_csv(fp, encoding_errors="ignore")
        except Exception:
            return None

    def map_columns(frame):
        cols = list(frame.columns)
        date_col = _resolve_col(cols, ["date", "datetime", "timestamp", "time", "trade_date", "trading_date", "date_of_trade"], ("date", "timestamp", "datetime", "trade"))
        open_col = _resolve_col(cols, ["open", "open_price", "opening_price", "openprice"], ("open", "opening"))
        high_col = _resolve_col(cols, ["high", "high_price", "highprice"], ("high", "highest"))
        low_col = _resolve_col(cols, ["low", "low_price", "lowprice"], ("low", "lowest"))
        close_col = _resolve_col(cols, ["close", "close_price", "closing_price", "closing", "adj_close", "adjusted_close", "price", "ltp"], ("close", "closing", "adjclose", "price", "ltp"))
        vol_col = _resolve_col(cols, ["volume", "vol", "volume_traded", "total_volume", "shares_traded"], ("volume", "vol", "shares"))
        return date_col, open_col, high_col, low_col, close_col, vol_col

    date_col, open_col, high_col, low_col, close_col, vol_col = map_columns(x)
    if not all([date_col, open_col, high_col, low_col, close_col]):
        try:
            raw = pd.read_csv(fp, header=None, nrows=20, encoding="utf-8-sig")
            for i in range(len(raw)):
                row = raw.iloc[i].astype(str).tolist()
                joined = " ".join(_norm(v) for v in row)
                if sum(k in joined for k in ["date", "open", "high", "low", "close"]) >= 4:
                    candidate = pd.read_csv(fp, header=i, encoding="utf-8-sig")
                    dc, oc, hc, lc, cc, vc = map_columns(candidate)
                    if all([dc, oc, hc, lc, cc]):
                        x = candidate
                        date_col, open_col, high_col, low_col, close_col, vol_col = dc, oc, hc, lc, cc, vc
                        break
        except Exception:
            pass
    if not date_col:
        for c in x.columns:
            p = pd.to_datetime(x[c], errors="coerce", format="mixed")
            if len(p) and p.notna().mean() >= 0.70:
                date_col = c
                break
    if not all([date_col, open_col, high_col, low_col, close_col]):
        return None

    y = pd.DataFrame({
        "date": pd.to_datetime(x[date_col], errors="coerce", format="mixed").dt.normalize(),
        "open": pd.to_numeric(x[open_col], errors="coerce"),
        "high": pd.to_numeric(x[high_col], errors="coerce"),
        "low": pd.to_numeric(x[low_col], errors="coerce"),
        "close": pd.to_numeric(x[close_col], errors="coerce"),
    })
    if vol_col:
        y["volume"] = pd.to_numeric(x[vol_col], errors="coerce")
    y = y.dropna(subset=["date", "open", "high", "low", "close"])
    if y.empty:
        return None

    sym_col = _resolve_col(x.columns, ["symbol", "ticker", "stock", "security", "instrument", "company"], ("symbol", "ticker"))
    if sym_col:
        y["symbol"] = x.loc[y.index, sym_col].astype(str).str.upper().str.replace("-", "_", regex=False)
    else:
        y["symbol"] = Path(fp).stem.upper().replace("-", "_")
    y = y[y["symbol"].notna() & (y["symbol"].str.len() > 0)]
    if y.empty:
        return None
    return y[["date", "symbol", "open", "high", "low", "close"] + (["volume"] if "volume" in y else [])]


def load_universe(pattern, max_files=None):
    files = sorted(glob.glob(pattern, recursive=True))
    if max_files:
        files = files[:max_files]
    frames, rejected = [], []
    for fp in files:
        z = _read_robust(fp)
        if z is None:
            rejected.append(Path(fp).name)
        else:
            frames.append(z)
    if not frames:
        raise RuntimeError(f"No usable OHLCV files matched {pattern}. Rejected {len(rejected)} files. Samples: {', '.join(rejected[:10])}")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True)


def proxy_market(df):
    z = df.sort_values(["symbol", "date"]).copy()
    z["r"] = z.groupby("symbol").close.pct_change()
    r = z.groupby("date").r.median().fillna(0.0)
    close = (1 + r).cumprod() * 1000
    return pd.DataFrame({"date": close.index, "close": close.values})


def execute_btst(selected, cfg):
    if selected.empty:
        return pd.DataFrame()
    sl = float(cfg["costs"].get("slippage_bps_per_side", 5)) / 10000
    tc = float(cfg["costs"].get("transaction_cost_bps_per_side", 8)) / 10000
    max_pos = int(cfg["portfolio"].get("max_positions", 10))
    gross = float(cfg["portfolio"].get("max_gross_exposure", .95))
    cap = float(cfg["portfolio"].get("max_position_weight", 1.0))
    sm = float(cfg["execution"].get("stop_atr_mult", 1.5))
    tm = float(cfg["execution"].get("target_atr_mult", 2.0))
    rows = []
    for date, g in selected.groupby("date"):
        picks = g.nlargest(max_pos, "predicted_return")
        n = len(picks)
        if n == 0:
            continue
        weight = min(cap, gross / n)
        for _, r in picks.iterrows():
            if not np.isfinite(r.next_open):
                continue
            entry = float(r.close) * (1 + sl)
            atr = float(np.clip(r.atr_pct, .005, .20))
            stop = entry * (1 - sm * atr)
            target = entry * (1 + tm * atr)
            if r.next_low <= stop:
                exit_px, reason = stop, "stop"
            elif r.next_high >= target:
                exit_px, reason = target, "target"
            else:
                exit_px, reason = float(r.next_open), "next_open"
            exit_px *= 1 - sl
            net = exit_px / entry - 1 - 2 * tc
            rows.append({"date": date, "symbol": r.symbol, "family": r.family, "regime": r.regime,
                         "predicted_return": r.predicted_return, "net_return": net, "reason": reason,
                         "weight": weight, "weighted_return": net * weight})
    return pd.DataFrame(rows)


def run(config_path):
    cfg = load_config(config_path)
    dc = cfg["data"]
    families = cfg["strategies"]["enabled"]
    df = load_universe(dc["daily_glob"], dc.get("max_symbols"))
    df = df[(df.date >= pd.Timestamp(dc["start_date"])) & (df.date <= pd.Timestamp(dc["end_date"]))]
    print(f"Loaded {len(df):,} rows across {df.symbol.nunique():,} symbols, {df.date.min().date()} to {df.date.max().date()}")
    market = None
    if glob.glob(dc.get("market_glob", ""), recursive=True):
        market = load_universe(dc["market_glob"], 1)
    if market is None or market.empty:
        market = proxy_market(df)
        print("Using cross-sectional proxy market index (no NIFTY 50 file supplied).")

    x = add_features(df, market)
    x["btst_return"] = x["next_open"] / x["close"] - 1
    x = add_candidate_scores(x, families)
    folds = build_folds(pd.DatetimeIndex(sorted(x.date.unique())), cfg)
    seed = int(cfg["research"].get("random_state", 42))
    q = float(cfg["research"].get("strategy_selection_quantile", .65))
    maxpr = int(cfg["research"].get("max_families_per_regime", 1))
    all_trades, route_rows, pred_rows = [], [], []

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
        models = fit_family_models(train, families, FEATURES, seed + fid)
        vp = []
        for fam, (model, cols) in models.items():
            z = val.copy()
            z["family"] = fam
            z["candidate_score"] = z[f"score_{fam}"]
            z["predicted_return"] = model.predict(z[cols].fillna(0))
            vp.append(z[["date", "symbol", "regime", "family", "candidate_score", "predicted_return", "btst_return"]])
        if not vp:
            continue
        vp = pd.concat(vp, ignore_index=True)
        routing, _ = choose_regimes(vp, families, q, maxpr)
        fallback = vp[vp.candidate_score >= .60].groupby("family").btst_return.mean().sort_values(ascending=False).head(maxpr).index.tolist() or [families[0]]
        for regime, chosen in routing.items():
            route_rows.append({"fold": fid, "regime": regime, "families": ",".join(chosen)})
        parts = []
        for fam, (model, cols) in models.items():
            z = test.copy()
            z["family"] = fam
            z["candidate_score"] = z[f"score_{fam}"]
            z["predicted_return"] = model.predict(z[cols].fillna(0))
            parts.append(z)
        if not parts:
            continue
        pred = pd.concat(parts, ignore_index=True)
        pred = pd.concat([g[g.family.isin(routing.get(reg, fallback))] for reg, g in pred.groupby("regime", dropna=False)], ignore_index=True)
        pred = pred[(pred.candidate_score >= .60) & (pred.predicted_return > 0)]
        if not pred.empty:
            pred_rows.append(pred[["date", "symbol", "regime", "family", "candidate_score", "predicted_return"]])
        t = execute_btst(pred, cfg)
        if not t.empty:
            t["fold"] = fid
            all_trades.append(t)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    out = Path("docs")
    out.mkdir(exist_ok=True)
    if not trades.empty:
        trades.to_csv(out / "btst_composite_btst_oos_trades.csv", index=False)
        daily = trades.groupby("date").weighted_return.sum().sort_index()
        monthly = ((1 + daily).groupby(daily.index.to_period("M")).prod() - 1).rename("monthly_return").reset_index()
        monthly.to_csv(out / "btst_composite_btst_monthly.csv", index=False)
    pd.DataFrame(route_rows).to_csv(out / "btst_composite_btst_routing.csv", index=False)
    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_csv(out / "btst_composite_btst_predictions.csv", index=False)
    pd.DataFrame([metrics(trades)]).to_csv(out / "btst_composite_btst_metrics.csv", index=False)
    print(metrics(trades))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/composite.yaml")
    run(ap.parse_args().config)
