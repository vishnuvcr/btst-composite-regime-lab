from __future__ import annotations
import argparse, csv, glob
from pathlib import Path
import numpy as np
import pandas as pd
from btst_composite import FEATURES, add_candidate_scores, add_features, build_folds, fit_family_models, load_config, make_regime_labels, metrics
from btst_composite_regime import choose_regimes


def _norm(c):
    return str(c).replace('\ufeff','').strip().lower().replace(' ','_').replace('-','_')


def _read_robust(fp):
    try:
        x=pd.read_csv(fp)
    except Exception:
        try: x=pd.read_csv(fp,sep=None,engine='python')
        except Exception: return None
    x.columns=[_norm(c) for c in x.columns]
    date_alias=['date','datetime','timestamp','time','trade_date','trading_date']
    aliases={'open':['o','open_price','opening_price'],'high':['h','high_price'],'low':['l','low_price'],'close':['c','adj_close','price','close_price','closing_price'],'volume':['vol','v','volume_traded']}
    date_col=next((c for c in date_alias if c in x.columns),None)
    if date_col is None or not all(k in x.columns or any(a in x.columns for a in v) for k,v in aliases.items() if k!='volume'):
        # Some yfinance/NSE exports have a metadata/header row before the real header.
        try:
            raw=pd.read_csv(fp,header=None,nrows=12)
            header=None
            for i,row in raw.iterrows():
                vals={_norm(v) for v in row.astype(str)}
                if ('date' in vals or 'timestamp' in vals or 'datetime' in vals) and sum(any(a in vals for a in [k]+v) for k,v in aliases.items() if k!='volume')>=4:
                    header=i; break
            if header is not None:
                x=pd.read_csv(fp,header=header); x.columns=[_norm(c) for c in x.columns]
                date_col=next((c for c in date_alias if c in x.columns),None)
        except Exception: pass
    if date_col is None: return None
    ren={}
    for target,alts in aliases.items():
        if target not in x.columns:
            for a in alts:
                if a in x.columns: ren[a]=target; break
    x=x.rename(columns=ren)
    if not all(c in x.columns for c in ('open','high','low','close')): return None
    x['date']=pd.to_datetime(x[date_col].astype(str).str.strip(),errors='coerce',format='mixed').dt.normalize()
    for c in ('open','high','low','close','volume'):
        if c in x: x[c]=pd.to_numeric(x[c],errors='coerce')
    x=x.dropna(subset=['date','open','high','low','close'])
    if x.empty:return None
    sym_col=next((c for c in ('symbol','ticker','stock','security','instrument') if c in x.columns),None)
    if sym_col:
        x['symbol']=x[sym_col].astype(str).str.upper().str.replace('-','_',regex=False)
    else:
        x['symbol']=Path(fp).stem.upper().replace('-','_')
    keep=['date','symbol','open','high','low','close']+(['volume'] if 'volume' in x else [])
    return x[keep]


def load_universe(pattern,max_files=None):
    files=sorted(glob.glob(pattern,recursive=True))
    if max_files: files=files[:max_files]
    frames=[]
    for fp in files:
        z=_read_robust(fp)
        if z is not None: frames.append(z)
    if not frames: raise RuntimeError(f'No usable OHLCV files matched {pattern}')
    return pd.concat(frames,ignore_index=True).sort_values(['symbol','date']).drop_duplicates(['symbol','date'],keep='last').reset_index(drop=True)


def proxy_market(df):
    z=df.sort_values(['symbol','date']).copy(); z['r']=z.groupby('symbol').close.pct_change()
    r=z.groupby('date').r.median().fillna(0.0); close=(1+r).cumprod()*1000
    return pd.DataFrame({'date':close.index,'close':close.values})


def execute_btst(selected,cfg):
    if selected.empty:return pd.DataFrame()
    sl=float(cfg['costs'].get('slippage_bps_per_side',5))/10000; tc=float(cfg['costs'].get('transaction_cost_bps_per_side',8))/10000
    max_pos=int(cfg['portfolio'].get('max_positions',10)); gross=float(cfg['portfolio'].get('max_gross_exposure',.95)); sm=float(cfg['execution'].get('stop_atr_mult',1.5)); tm=float(cfg['execution'].get('target_atr_mult',2.0)); rows=[]
    for date,g in selected.groupby('date'):
        for _,r in g.nlargest(max_pos,'predicted_return').iterrows():
            entry=float(r.close)*(1+sl); atr=float(np.clip(r.atr_pct,.005,.20)); stop=entry*(1-sm*atr); target=entry*(1+tm*atr)
            if r.next_low<=stop: exit_px,reason=stop,'stop'
            elif r.next_high>=target: exit_px,reason=target,'target'
            else: exit_px,reason=float(r.next_open),'next_open'
            exit_px*=1-sl; net=exit_px/entry-1-2*tc
            rows.append({'date':date,'symbol':r.symbol,'family':r.family,'regime':r.regime,'predicted_return':r.predicted_return,'net_return':net,'reason':reason})
    out=pd.DataFrame(rows)
    if out.empty:return out
    out['weight']=gross/out.groupby('date').symbol.transform('count').clip(lower=1); out['weighted_return']=out.net_return*out.weight
    return out


def run(config_path):
    cfg=load_config(config_path); dc=cfg['data']; families=cfg['strategies']['enabled']
    df=load_universe(dc['daily_glob'],dc.get('max_symbols')); df=df[(df.date>=pd.Timestamp(dc['start_date']))&(df.date<=pd.Timestamp(dc['end_date']))]
    market=load_universe(dc['market_glob'],1) if glob.glob(dc.get('market_glob',''),recursive=True) else proxy_market(df)
    x=add_features(df,market); x['btst_return']=x['next_open']/x['close']-1; x=add_candidate_scores(x,families)
    folds=build_folds(pd.DatetimeIndex(sorted(x.date.unique())),cfg); seed=int(cfg['research'].get('random_state',42)); q=float(cfg['research'].get('strategy_selection_quantile',.65)); maxpr=int(cfg['research'].get('max_families_per_regime',1))
    all_trades=[]; route_rows=[]; pred_rows=[]
    for fid,fold in enumerate(folds,1):
        train=x[(x.date>=fold.train_start)&(x.date<=fold.train_end)].copy(); val=x[(x.date>=fold.validation_start)&(x.date<=fold.validation_end)].copy(); test=x[(x.date>=fold.test_start)&(x.date<=fold.test_end)].copy()
        if train.empty or val.empty or test.empty: continue
        trreg,_,_=make_regime_labels(train,train,int(cfg['research'].get('n_clusters',6)),seed); fwreg,_,_=make_regime_labels(train,pd.concat([val,test]),int(cfg['research'].get('n_clusters',6)),seed)
        train['regime']=train.date.map(trreg); val['regime']=val.date.map(fwreg); test['regime']=test.date.map(fwreg)
        models=fit_family_models(train,families,FEATURES,seed+fid); vp=[]
        for fam,(model,cols) in models.items():
            z=val.copy(); z['family']=fam; z['candidate_score']=z[f'score_{fam}']; z['predicted_return']=model.predict(z[cols].fillna(0)); vp.append(z[['date','symbol','regime','family','candidate_score','predicted_return','btst_return']])
        if not vp: continue
        vp=pd.concat(vp,ignore_index=True); routing,rstats=choose_regimes(vp,families,q,maxpr); fallback=vp[vp.candidate_score>=.60].groupby('family').btst_return.mean().sort_values(ascending=False).head(maxpr).index.tolist() or [families[0]]
        for regime,chosen in routing.items(): route_rows.append({'fold':fid,'regime':regime,'families':','.join(chosen)})
        parts=[]
        for fam,(model,cols) in models.items():
            z=test.copy(); z['family']=fam; z['candidate_score']=z[f'score_{fam}']; z['predicted_return']=model.predict(z[cols].fillna(0)); parts.append(z)
        pred=pd.concat(parts,ignore_index=True); pred=pd.concat([g[g.family.isin(routing.get(reg,fallback))] for reg,g in pred.groupby('regime',dropna=False)],ignore_index=True); pred=pred[(pred.candidate_score>=.60)&(pred.predicted_return>0)]
        if not pred.empty: pred_rows.append(pred[['date','symbol','regime','family','candidate_score','predicted_return']])
        t=execute_btst(pred,cfg)
        if not t.empty:t['fold']=fid; all_trades.append(t)
    trades=pd.concat(all_trades,ignore_index=True) if all_trades else pd.DataFrame(); out=Path('docs'); out.mkdir(exist_ok=True)
    if not trades.empty:
        trades.to_csv(out/'btst_composite_btst_oos_trades.csv',index=False); daily=trades.groupby('date').weighted_return.sum().sort_index(); ((1+daily).groupby(daily.index.to_period('M')).prod()-1).rename('monthly_return').reset_index().to_csv(out/'btst_composite_btst_monthly.csv',index=False)
    pd.DataFrame(route_rows).to_csv(out/'btst_composite_btst_routing.csv',index=False)
    if pred_rows: pd.concat(pred_rows,ignore_index=True).to_csv(out/'btst_composite_btst_predictions.csv',index=False)
    pd.DataFrame([metrics(trades)]).to_csv(out/'btst_composite_btst_metrics.csv',index=False); print(metrics(trades))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/composite.yaml'); run(ap.parse_args().config)
