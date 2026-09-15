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

BASE = [c for c in FEATURES if c in FEATURES]
ARCH = ["ret_1","ret_3","ret_5","ret_10","ret_20","ret_60","gap","range_pct","close_location","body_pct","atr_pct","vol20","volume_z","sma20_gap","sma50_gap","mkt_ret_1","mkt_ret_5","mkt_vol20","mkt_trend","breadth","rank_ret_5","rank_ret_20","rank_gap","rank_close_location","rank_volume_z","rank_sma20_gap"]


def add_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy()
    # Stock archetypes are deliberately coarse and deterministic; thresholds are fixed, not fitted on test.
    x["archetype"]="neutral"
    x.loc[(x.ret_20 > .08) & (x.sma20_gap > 0), "archetype"]="strong_trend"
    x.loc[(x.ret_20 < -.08) & (x.sma20_gap < 0), "archetype"]="weak_trend"
    x.loc[(x.gap > .015) & (x.close_location > .65), "archetype"]="gap_up_strength"
    x.loc[(x.gap < -.015) & (x.close_location < .35), "archetype"]="gap_down_weakness"
    x.loc[(x.volume_z > 1.5) & (x.range_pct > x.atr_pct), "archetype"]="volume_expansion"
    x.loc[(x.atr_pct > x.atr_pct.rolling(60, min_periods=20).median()), "archetype"] = x.loc[(x.atr_pct > x.atr_pct.rolling(60, min_periods=20).median()), "archetype"].where(x.archetype!="neutral", "volatility")
    return x


def design(frame, families, archetypes, regimes):
    z=frame.copy()
    for f in families:
        z[f"family_{f}"]=(z.family==f).astype(float)
    for a in archetypes:
        z[f"arch_{a}"]=(z.archetype==a).astype(float)
    for r in regimes:
        z[f"reg_{int(r)}"]=(z.regime==r).astype(float)
    for f in families:
        for r in regimes:
            z[f"fxr_{f}_{int(r)}"]=((z.family==f)&(z.regime==r)).astype(float)
        for a in archetypes:
            z[f"fxa_{f}_{a}"]=((z.family==f)&(z.archetype==a)).astype(float)
    cols=[c for c in ARCH if c in z.columns]
    cols += [f"score_{f}" for f in families]
    cols += [f"family_{f}" for f in families]+[f"arch_{a}" for a in archetypes]+[f"reg_{int(r)}" for r in regimes]
    cols += [f"fxr_{f}_{int(r)}" for f in families for r in regimes]
    cols += [f"fxa_{f}_{a}" for f in families for a in archetypes]
    return z, cols


def fit(train, families, seed):
    archs=sorted(train.archetype.dropna().unique().tolist())
    regs=sorted(train.regime.dropna().unique().tolist())
    parts=[]
    for f in families:
        q=train.dropna(subset=["btst_return",f"score_{f}"]).copy(); q["family"]=f
        if len(q)>=100: parts.append(q)
    if not parts: return None
    d=pd.concat(parts,ignore_index=True); d,cols=design(d,families,archs,regs)
    X=d[cols].astype(float).fillna(0); y=d.btst_return.astype(float); yb=(y>0).astype(int)
    if yb.nunique()<2:return None
    if LGBMClassifier is not None:
        clf=LGBMClassifier(n_estimators=180,learning_rate=.03,num_leaves=15,max_depth=5,min_child_samples=120,subsample=.8,colsample_bytree=.8,reg_lambda=5,random_state=seed,verbosity=-1)
        reg=LGBMRegressor(n_estimators=180,learning_rate=.03,num_leaves=15,max_depth=5,min_child_samples=120,subsample=.8,colsample_bytree=.8,reg_lambda=5,random_state=seed+1000,verbosity=-1)
    else:
        clf=HistGradientBoostingClassifier(max_iter=180,learning_rate=.04,max_leaf_nodes=15,min_samples_leaf=120,l2_regularization=5,random_state=seed)
        reg=HistGradientBoostingRegressor(max_iter=180,learning_rate=.04,max_leaf_nodes=15,min_samples_leaf=120,l2_regularization=5,random_state=seed+1000)
    clf.fit(X,yb); reg.fit(X,y.clip(-.25,.25))
    return {"clf":clf,"reg":reg,"cols":cols,"families":families,"archs":archs,"regs":regs}


def predict(pack, frame):
    out=[]
    for f in pack["families"]:
        z=frame.copy(); z["family"]=f; z,_=design(z,pack["families"],pack["archs"],pack["regs"]); X=z[pack["cols"]].astype(float).fillna(0)
        z["p_raw"]=pack["clf"].predict_proba(X)[:,1]; z["predicted_return"]=pack["reg"].predict(X)
        z["edge_score"]=z.p_raw*np.maximum(z.predicted_return,0)
        z["candidate_score"]=z[f"score_{f}"]
        out.append(z[["date","symbol","regime","archetype","family","candidate_score","p_raw","predicted_return","edge_score","btst_return"]])
    return pd.concat(out,ignore_index=True)


def calibrate_probability(vp):
    # Isotonic calibration is fit on validation only, then applied to the held-out test predictions.
    y=(vp.btst_return>0).astype(int); p=vp.p_raw.clip(0.001,.999)
    if y.nunique()<2 or len(vp)<100:return vp.p_raw.clip(.01,.99)
    iso=IsotonicRegression(out_of_bounds="clip").fit(p,y)
    return iso.predict(p)


def policy(vp,max_positions):
    if vp.empty:return {"min_prob":.99,"min_pred":0.0,"min_edge":0.0}
    v=vp.copy(); v["p_cal"] = calibrate_probability(v)
    probs=np.unique(np.quantile(v.p_cal,[.50,.60,.70,.80,.85,.90,.95]))
    preds=np.unique(np.quantile(v.predicted_return,[.45,.55,.65,.75,.85]))
    edges=np.unique(np.quantile(v.edge_score,[.50,.65,.75,.85,.90]))
    best=None
    for p in probs:
      for pr in preds:
       for e in edges:
        z=v[(v.candidate_score>=.60)&(v.p_cal>=p)&(v.predicted_return>=pr)&(v.edge_score>=e)].copy()
        if z.empty:continue
        z=z.sort_values(["date","edge_score"],ascending=[True,False]).drop_duplicates(["date","symbol"])
        z=pd.concat([g.nlargest(max_positions,"edge_score") for _,g in z.groupby("date")],ignore_index=True)
        if len(z)<50:continue
        daily=z.groupby("date").btst_return.mean(); m=daily.mean(); sd=daily.std()
        # Validation objective rewards positive edge and stability, but does not optimize total test return.
        sc=float(m-.35*sd+.0015*(daily>0).mean())
        if best is None or sc>best[0]:best=(sc,float(p),float(pr),float(e))
    if best is None:return {"min_prob":.99,"min_pred":0.0,"min_edge":np.inf}
    return {"min_prob":best[1],"min_pred":best[2],"min_edge":best[3]}


def execute(sel,cfg):
    if sel.empty:return pd.DataFrame()
    maxp=int(cfg["portfolio"].get("max_positions",10)); gross=float(cfg["portfolio"].get("max_gross_exposure",.95)); cap=float(cfg["portfolio"].get("max_position_weight",.15)); rows=[]
    for d,g in sel.groupby("date"):
        g=g.nlargest(maxp,"edge_score"); n=len(g); w=min(cap,gross/n) if n else 0
        for _,r in g.iterrows():rows.append({"date":d,"symbol":r.symbol,"family":r.family,"regime":r.regime,"archetype":r.archetype,"candidate_score":r.candidate_score,"p_positive":r.p_cal,"predicted_return":r.predicted_return,"edge_score":r.edge_score,"net_return":r.btst_return,"weight":w,"weighted_return":r.btst_return*w})
    return pd.DataFrame(rows)


def run(path):
    cfg=load_config(path); dc=cfg["data"]; families=cfg["strategies"]["enabled"]
    df=load_universe(dc["daily_glob"],dc.get("max_symbols")); df=df[(df.date>=pd.Timestamp(dc["start_date"]))&(df.date<=pd.Timestamp(dc["end_date"]))]
    market=None
    if dc.get("market_glob"):
      from glob import glob
      if glob(dc["market_glob"],recursive=True):market=load_universe(dc["market_glob"],1)
    if market is None or market.empty:market=proxy_market(df); print("Using cross-sectional proxy market index (no NIFTY 50 file supplied).")
    x=add_features(df,market); x=add_archetypes(x); x["btst_return"]=x.apply(lambda r:strict_net(r,cfg)[0],axis=1); x=add_candidate_scores(x,families)
    folds=build_folds(pd.DatetimeIndex(sorted(x.date.unique())),cfg); seed=int(cfg["research"].get("random_state",42)); trades_all=[]; pred_all=[]; tune=[]
    for fid,fold in enumerate(folds,1):
      train=x[(x.date>=fold.train_start)&(x.date<=fold.train_end)].copy(); val=x[(x.date>=fold.validation_start)&(x.date<=fold.validation_end)].copy(); test=x[(x.date>=fold.test_start)&(x.date<=fold.test_end)].copy()
      if train.empty or val.empty or test.empty:continue
      trreg,_,_=make_regime_labels(train,train,int(cfg["research"].get("n_clusters",6)),seed); fwreg,_,_=make_regime_labels(train,pd.concat([val,test]),int(cfg["research"].get("n_clusters",6)),seed)
      train["regime"]=train.date.map(trreg); val["regime"]=val.date.map(fwreg); test["regime"]=test.date.map(fwreg)
      pack=fit(train,families,seed+fid)
      if pack is None:continue
      vp=predict(pack,val); pol=policy(vp,int(cfg["portfolio"].get("max_positions",10))); tune.append({"fold":fid,**pol})
      tp=predict(pack,test); tp["p_cal"]=tp.p_raw.clip(.01,.99)
      # Probability calibration must be fit from validation; fit a fresh calibrator on validation then transform test.
      vv=vp.copy(); yy=(vv.btst_return>0).astype(int); pp=vv.p_raw.clip(.001,.999)
      if yy.nunique()>=2 and len(vv)>=100: tp["p_cal"]=IsotonicRegression(out_of_bounds="clip").fit(pp,yy).predict(tp.p_raw.clip(.001,.999))
      tp=tp[(tp.candidate_score>=.60)&(tp.p_cal>=pol["min_prob"])&(tp.predicted_return>=pol["min_pred"])&(tp.edge_score>=pol["min_edge"])].copy()
      if not tp.empty:
        tp=tp.sort_values(["date","edge_score"],ascending=[True,False]).drop_duplicates(["date","symbol"]); pred_all.append(tp[["date","symbol","regime","archetype","family","candidate_score","p_cal","predicted_return","edge_score"]])
      t=execute(tp,cfg)
      if not t.empty:t["fold"]=fid; trades_all.append(t)
    trades=pd.concat(trades_all,ignore_index=True) if trades_all else pd.DataFrame(); out=Path("docs"); out.mkdir(exist_ok=True)
    if not trades.empty:
      trades.to_csv(out/"btst_strategy_ranking_v5_trades.csv",index=False); daily=trades.groupby("date").weighted_return.sum().sort_index(); monthly=((1+daily).groupby(daily.index.to_period("M")).prod()-1).rename("monthly_return").reset_index(); monthly.to_csv(out/"btst_strategy_ranking_v5_monthly.csv",index=False)
    if pred_all:pd.concat(pred_all,ignore_index=True).to_csv(out/"btst_strategy_ranking_v5_predictions.csv",index=False)
    pd.DataFrame(tune).to_csv(out/"btst_strategy_ranking_v5_tuning.csv",index=False); result=metrics(trades); pd.DataFrame([result]).to_csv(out/"btst_strategy_ranking_v5_metrics.csv",index=False); print(result)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config/composite.yaml"); run(ap.parse_args().config)
