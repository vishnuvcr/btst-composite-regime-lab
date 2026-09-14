from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

FAMILIES = ["momentum", "mean_reversion", "gap_continuation", "gap_fade", "closing_strength",
            "breakout", "volatility_expansion", "relative_strength", "trend", "volume_anomaly", "hybrid"]
FEATURES = ["ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60", "gap", "range_pct",
            "close_location", "body_pct", "atr_pct", "vol20", "volume_z", "sma20_gap", "sma50_gap",
            "mkt_ret_1", "mkt_ret_5", "mkt_vol20", "mkt_trend", "breadth", "rank_ret_5", "rank_ret_20",
            "rank_gap", "rank_close_location", "rank_volume_z", "rank_sma20_gap"]

@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def norm_col(c: str) -> str:
    return str(c).replace("\ufeff", "").strip().lower().replace(" ", "_").replace("-", "_")


def parse_dates(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.replace("\ufeff", "", regex=False).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    compact = x.str.fullmatch(r"\d{8}")
    if compact.any():
        out.loc[compact] = pd.to_datetime(x.loc[compact], format="%Y%m%d", errors="coerce")
    rem = out.isna()
    if rem.any():
        out.loc[rem] = pd.to_datetime(x.loc[rem], errors="coerce", format="mixed")
    return out.dt.normalize()


def read_csv_universe(pattern: str, max_files: int | None = None) -> pd.DataFrame:
    files = sorted(glob.glob(pattern, recursive=True))
    if max_files:
        files = files[:max_files]
    frames = []
    for fp in files:
        try:
            x = pd.read_csv(fp)
        except Exception:
            continue
        x.columns = [norm_col(c) for c in x.columns]
        date_col = next((c for c in ("date", "datetime", "timestamp", "time") if c in x.columns), None)
        if date_col is None:
            continue
        aliases = {"open": ["o"], "high": ["h"], "low": ["l"], "close": ["adj_close", "price", "c"], "volume": ["vol", "v"]}
        rename = {}
        for target, alts in aliases.items():
            if target not in x.columns:
                for alt in alts:
                    if alt in x.columns and alt != date_col:
                        rename[alt] = target
                        break
        x = x.rename(columns=rename)
        if not all(c in x.columns for c in ("open", "high", "low", "close")):
            continue
        x["date"] = parse_dates(x[date_col])
        for c in ("open", "high", "low", "close", "volume"):
            if c in x.columns:
                x[c] = pd.to_numeric(x[c], errors="coerce")
        x = x.dropna(subset=["date", "open", "high", "low", "close"])
        if x.empty:
            continue
        x["symbol"] = Path(fp).stem.upper().replace("-", "_")
        keep = ["date", "symbol", "open", "high", "low", "close"] + (["volume"] if "volume" in x else [])
        frames.append(x[keep])
    if not frames:
        raise RuntimeError(f"No usable OHLCV files matched {pattern}")
    return (pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
            .drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True))


def add_features(df: pd.DataFrame, market: pd.DataFrame | None = None) -> pd.DataFrame:
    x = df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    g = x.groupby("symbol", group_keys=False)
    x["ret_1"] = g.close.pct_change()
    for n in (3, 5, 10, 20, 60):
        x[f"ret_{n}"] = g.close.pct_change(n)
    prev = g.close.shift(1)
    x["gap"] = x.open / prev - 1
    x["range_pct"] = (x.high - x.low) / x.close.replace(0, np.nan)
    x["close_location"] = (x.close - x.low) / (x.high - x.low).replace(0, np.nan)
    x["body_pct"] = (x.close - x.open) / x.open.replace(0, np.nan)
    x["atr_pct"] = g.range_pct.transform(lambda s: s.rolling(14, min_periods=10).mean())
    x["sma20_gap"] = x.close / g.close.transform(lambda s: s.rolling(20, min_periods=15).mean()) - 1
    x["sma50_gap"] = x.close / g.close.transform(lambda s: s.rolling(50, min_periods=30).mean()) - 1
    x["vol20"] = g.ret_1.transform(lambda s: s.rolling(20, min_periods=15).std())
    if "volume" in x:
        vm = g.volume.transform(lambda s: s.rolling(20, min_periods=15).mean())
        vs = g.volume.transform(lambda s: s.rolling(20, min_periods=15).std())
        x["volume_z"] = (x.volume - vm) / vs.replace(0, np.nan)
    else:
        x["volume_z"] = 0.0
    for c in ("ret_5", "ret_20", "gap", "close_location", "volume_z", "sma20_gap"):
        x[f"rank_{c}"] = x.groupby("date")[c].rank(pct=True)
    if market is not None and not market.empty:
        m = market.sort_values("date").drop_duplicates("date").copy()
        m["mkt_ret_1"] = m.close.pct_change()
        m["mkt_ret_5"] = m.close.pct_change(5)
        m["mkt_vol20"] = m.mkt_ret_1.rolling(20, min_periods=15).std()
        m["mkt_trend"] = m.close / m.close.rolling(50, min_periods=30).mean() - 1
        x = x.merge(m[["date", "mkt_ret_1", "mkt_ret_5", "mkt_vol20", "mkt_trend"]], on="date", how="left")
    else:
        x[["mkt_ret_1", "mkt_ret_5", "mkt_vol20", "mkt_trend"]] = 0.0
    x["breadth"] = x.groupby("date").ret_5.transform(lambda s: (s > 0).mean())
    x["next_open"] = g.open.shift(-1)
    x["next_high"] = g.high.shift(-1)
    x["next_low"] = g.low.shift(-1)
    x["next_close"] = g.close.shift(-1)
    x["btst_return"] = x.next_close / x.next_open - 1
    return x.replace([np.inf, -np.inf], np.nan)


def volatility_rank(x):
    return x.groupby("date").range_pct.rank(pct=True)


def strategy_score(x: pd.DataFrame, family: str) -> pd.Series:
    r5, r20, rg, rc, rv, rs = x.rank_ret_5, x.rank_ret_20, x.rank_gap, x.rank_close_location, x.rank_volume_z, x.rank_sma20_gap
    vr = volatility_rank(x)
    if family == "momentum": return .55*r5 + .25*r20 + .20*rc
    if family == "mean_reversion": return .50*(1-rs) + .30*(1-r5) + .20*rc
    if family == "gap_continuation": return .45*r5 + .25*r20 + .30*rg
    if family == "gap_fade": return .55*(1-rg) + .25*(1-r5) + .20*rc
    if family == "closing_strength": return .70*rc + .30*r5
    if family == "breakout": return .45*r20 + .30*rc + .25*vr
    if family == "volatility_expansion": return .50*vr + .30*rv + .20*rc
    if family == "relative_strength": return .65*r20 + .35*r5
    if family == "trend": return .55*x.sma50_gap.groupby(x.date).rank(pct=True) + .25*r20 + .20*rc
    if family == "volume_anomaly": return .55*rv + .25*r5 + .20*rc
    if family == "hybrid": return .35*r5 + .25*r20 + .20*rc + .10*rv + .10*(1-rg)
    raise ValueError(f"Unknown strategy family: {family}")


def add_candidate_scores(x: pd.DataFrame, families: list[str]) -> pd.DataFrame:
    y = x.copy()
    for f in families:
        y[f"score_{f}"] = strategy_score(y, f)
    return y


def make_regime_labels(train: pd.DataFrame, forward: pd.DataFrame, n_clusters: int, seed: int):
    cols = ["mkt_ret_5", "mkt_vol20", "mkt_trend", "breadth", "vol20"]
    daily = train.groupby("date")[cols].mean().dropna()
    if len(daily) < max(30, n_clusters * 4):
        return pd.Series(dtype="float64"), None, None
    scaler = StandardScaler().fit(daily)
    km = KMeans(n_clusters=min(n_clusters, len(daily)), random_state=seed, n_init=20).fit(scaler.transform(daily))
    fday = forward.groupby("date")[cols].mean().fillna(daily.mean())
    return pd.Series(km.predict(scaler.transform(fday)), index=fday.index), scaler, km


def build_folds(dates: pd.DatetimeIndex, cfg: dict) -> list[Fold]:
    start, end = dates.min(), dates.max()
    r = cfg["research"]
    train_years, val_months, test_months, step = map(int, (r["train_years"], r["validation_months"], r["test_months"], r["step_months"]))
    embargo = int(r.get("embargo_days", 5))
    folds = []
    test_start = start + pd.DateOffset(years=train_years, months=val_months) + pd.Timedelta(days=embargo)
    while test_start <= end:
        val_end = test_start - pd.Timedelta(days=embargo)
        val_start = val_end - pd.DateOffset(months=val_months) + pd.Timedelta(days=1)
        train_end = val_start - pd.Timedelta(days=1)
        train_start = train_end - pd.DateOffset(years=train_years) + pd.Timedelta(days=1)
        test_end = min(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), end)
        if train_start >= start and test_end >= test_start:
            folds.append(Fold(train_start, train_end, val_start, val_end, test_start, test_end))
        test_start += pd.DateOffset(months=step)
    return folds


def fit_family_models(train: pd.DataFrame, families: list[str], features: list[str], seed: int):
    models = {}
    for j, fam in enumerate(families):
        d = train.dropna(subset=features + ["btst_return", f"score_{fam}"]).copy()
        if len(d) < 100:
            continue
        cols = features + [f"score_{fam}"]
        X, y = d[cols].astype(float), d.btst_return.clip(-.25, .25).astype(float)
        if LGBMRegressor is not None:
            model = LGBMRegressor(n_estimators=180, learning_rate=.035, num_leaves=15, max_depth=5,
                                  subsample=.8, colsample_bytree=.8, random_state=seed+j, verbosity=-1)
        else:
            model = HistGradientBoostingRegressor(max_iter=180, learning_rate=.04, max_leaf_nodes=15,
                                                  l2_regularization=1.0, random_state=seed+j)
        model.fit(X, y)
        models[fam] = (model, cols)
    return models


def portfolio_backtest(selected: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(columns=["date", "symbol", "family", "predicted_return", "net_return", "weight"])
    sl = float(cfg["costs"].get("slippage_bps_per_side", 5)) / 10000
    tc = float(cfg["costs"].get("transaction_cost_bps_per_side", 8)) / 10000
    max_pos = int(cfg["portfolio"].get("max_positions", 10))
    gross = float(cfg["portfolio"].get("max_gross_exposure", .95))
    stop_mult = float(cfg["execution"].get("stop_atr_mult", 1.5))
    target_mult = float(cfg["execution"].get("target_atr_mult", 2.0))
    rows = []
    for date, d in selected.groupby("date"):
        d = d.sort_values("predicted_return", ascending=False).head(max_pos)
        for _, r in d.iterrows():
            atr = float(np.clip(r.atr_pct, .005, .20))
            entry = r.next_open * (1 + sl)
            stop, target = entry * (1-stop_mult*atr), entry * (1+target_mult*atr)
            if r.next_low <= stop:
                exit_px, reason = stop, "stop"
            elif r.next_high >= target:
                exit_px, reason = target, "target"
            else:
                exit_px, reason = r.next_close, "close"
            exit_px *= 1-sl
            ret = exit_px / entry - 1 - 2*tc
            rows.append({"date": date, "symbol": r.symbol, "family": r.family, "predicted_return": r.predicted_return,
                         "net_return": ret, "reason": reason})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["weight"] = gross / out.groupby("date").symbol.transform("count").clip(lower=1)
    out["weighted_return"] = out.net_return * out.weight
    return out


def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0, "total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "win_rate": 0.0, "profit_factor": 0.0}
    daily = trades.groupby("date").weighted_return.sum().sort_index()
    eq = (1+daily).cumprod()
    dd = eq/eq.cummax()-1
    loss = abs(trades.loc[trades.net_return <= 0, "net_return"].sum())
    pf = trades.loc[trades.net_return > 0, "net_return"].sum()/loss if loss else float("inf")
    return {"trades": int(len(trades)), "total_return": float(eq.iloc[-1]-1), "max_drawdown": float(dd.min()),
            "sharpe": float(daily.mean()/daily.std()*math.sqrt(252)) if daily.std() else 0.0,
            "win_rate": float((trades.net_return > 0).mean()), "profit_factor": float(pf)}


def run(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    dc = cfg["data"]
    df = read_csv_universe(dc["daily_glob"], dc.get("max_symbols"))
    df = df[(df.date >= pd.Timestamp(dc["start_date"])) & (df.date <= pd.Timestamp(dc["end_date"]))]
    market = None
    if glob.glob(dc.get("market_glob", ""), recursive=True):
        market = read_csv_universe(dc["market_glob"], 1)
    x = add_candidate_scores(add_features(df, market), cfg["strategies"]["enabled"])
    folds = build_folds(pd.DatetimeIndex(sorted(x.date.unique())), cfg)
    families = cfg["strategies"]["enabled"]
    all_trades, diagnostics, regime_rows = [], [], []
    seed = int(cfg["research"].get("random_state", 42))

    for i, fold in enumerate(folds, 1):
        train = x[(x.date >= fold.train_start) & (x.date <= fold.train_end)].copy()
        val = x[(x.date >= fold.validation_start) & (x.date <= fold.validation_end)].copy()
        test = x[(x.date >= fold.test_start) & (x.date <= fold.test_end)].copy()
        if train.empty or val.empty or test.empty:
            continue
        regimes, _, _ = make_regime_labels(train, pd.concat([val, test]), int(cfg["research"].get("n_clusters", 6)), seed)
        val["regime"] = val.date.map(regimes)
        test["regime"] = test.date.map(regimes)

        models = fit_family_models(train, families, FEATURES, seed+i)
        if not models:
            continue
        # Validation is the only place where the family set is chosen.
        vrows = []
        for fam, (model, cols) in models.items():
            z = val.copy()
            z["family"] = fam
            z["predicted_return"] = model.predict(z[cols].fillna(0))
            vrows.append(z[["date", "symbol", "family", "predicted_return", "btst_return", "regime"]])
        vp = pd.concat(vrows, ignore_index=True)
        top10 = vp.sort_values(["date", "predicted_return"], ascending=[True, False]).groupby("family").head(20)
        quality = top10.groupby("family").btst_return.mean().sort_values(ascending=False)
        q = float(cfg["research"].get("strategy_selection_quantile", .65))
        threshold = quality.quantile(q)
        selected_families = quality[quality >= threshold].index.tolist() or [quality.index[0]]
        diagnostics.append({"fold": i, "train_end": str(fold.train_end.date()), "test_start": str(fold.test_start.date()),
                            "selected_families": ",".join(selected_families), "validation_best_family": str(quality.index[0]),
                            "validation_best_return": float(quality.iloc[0])})

        preds = []
        for fam in selected_families:
            model, cols = models[fam]
            z = test.copy()
            z["family"] = fam
            z["predicted_return"] = model.predict(z[cols].fillna(0))
            z["candidate_score"] = z[f"score_{fam}"]
            z = z[(z.candidate_score >= .60) & (z.predicted_return > 0)]
            preds.append(z)
            if not z.empty:
                regime_rows.append(z[["date", "symbol", "family", "regime", "candidate_score", "predicted_return"]])
        pred = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
        trades = portfolio_backtest(pred, cfg)
        if not trades.empty:
            trades["fold"] = i
            all_trades.append(trades)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    out = Path("docs"); out.mkdir(exist_ok=True)
    if not trades.empty:
        trades.to_csv(out/"btst_composite_oos_trades.csv", index=False)
        monthly = trades.assign(month=pd.to_datetime(trades.date).dt.to_period("M")).groupby("month").weighted_return.sum().reset_index()
        monthly.to_csv(out/"btst_composite_monthly.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out/"btst_composite_fold_diagnostics.csv", index=False)
    if regime_rows:
        pd.concat(regime_rows, ignore_index=True).to_csv(out/"btst_composite_strategy_regime.csv", index=False)
    result = metrics(trades)
    pd.DataFrame([result]).to_csv(out/"btst_composite_metrics.csv", index=False)
    manifest = {"folds": len(folds), "metrics": result, "families": families,
                "anti_leakage": {"chronological_walk_forward": True, "validation_only_selection": True,
                                  "test_returns_used_for_selection": False, "signal_uses_same_day_information_only": True},
                "warning": "30% monthly return is a discovery target, not a guarantee. Validate costs, survivorship, execution and unseen periods before deployment."}
    (out/"btst_composite_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config/composite.yaml"); run(ap.parse_args().config)
