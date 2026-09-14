from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd
from btst_composite import FEATURES, add_candidate_scores, add_features, build_folds, fit_family_models, load_config, make_regime_labels, metrics, read_csv_universe
from btst_composite_regime import choose_regimes


def execute_btst(selected, cfg):
    if selected.empty: return pd.DataFrame()
    sl=float(cfg['costs'].get('slippage_bps_per_side',5))/10000; tc=float(cfg['costs'].get('transaction_cost_bps_per_side',8))/10000
    max_pos=int(cfg['portfolio'].get('max_positions',10)); gross=float(cfg['portfolio'].get('max_gross_exposure',.95))
    sm=float(cfg['execution'].get('stop_atr_mult',1.5)); tm=float(cfg['execution'].get('target_atr_mult',2.0)); rows=[]
    for date,g in selected.groupby('date'):
        for _,r in g.nlargest(max_pos,'predicted_return').iterrows():
            entry=float(r.close)*(1+sl); atr=float(np.clip(r.atr_pct,.005,.20)); stop=entry*(1-sm*atr); target=entry*(1+tm*atr)
            if r.next_low<=stop: exit_px,reason=stop,'stop'
            elif r.next_high>=target: exit_px,reason=target,'target'
            else: exit_px,reason=float(r.next_open),'next_open'
            exit_px*=1-sl; net=exit_px/entry-1-2*tc
            rows.append({'date':date,'symbol':r.symbol,'family':r.family,'predicted_return':r.predicted_return,'net_return':net,'reason':reason})
    out=pd.DataFrame(rows)
    if out.empty:return out
    out['weight']=gross/out.groupby('date').symbol.transform('count').clip(lower=1); out['weighted_return']=out.net_return*out.weight
    return out


def run(config_path):
    cfg=load_config(config_path); dc=cfg['data']; families=cfg['strategies']['enabled']
    df=read_csv_universe(dc['daily_glob'],dc.get('max_symbols')); df=df[(df.date>=pd.Timestamp(dc['start_date']))&(df.date<=pd.Timestamp(dc['end_date']))]
    market=read_csv_universe(dc['market_glob'],1) if glob.glob(dc.get('market_glob',''),recursive=True) else None
    x=add_candidate_scores(add_features(df,market),families); folds=build_folds(pd.DatetimeIndex(sorted(x.date.unique())),cfg)
    seed=int(cfg['research'].get('random_state',42)); q=float(cfg['research'].get('strategy_selection_quantile',.65)); maxpr=int(cfg['research'].get('max_families_per_regime',1))
    all_trades=[]; route_rows=[]; pred_rows=[]
    for fid,fold in enumerate(folds,1):
        train=x[(x.date>=fold.train_start)&(x.date<=fold.train_end)].copy(); val=x[(x.date>=fold.validation_start)&(x.date<=fold.validation_end)].copy(); test=x[(x.date>=fold.test_start)&(x.date<=fold.test_end)].copy()
        if train.empty or val.empty or test.empty: continue
        trreg,_,_=make_regime_labels(train,train,int(cfg['research'].get('n_clusters',6)),seed); fwreg,_,_=make_regime_labels(train,pd.concat([val,test]),int(cfg['research'].get('n_clusters',6)),seed)
        train['regime']=train.date.map(trreg); val['regime']=val.date.map(fwreg); test['regime']=test.date.map(fwreg)
        models=fit_family_models(train,families,FEATURES,seed+fid); vp=[]
        for fam,(model,cols) in models.items():
            z=val.copy(); z['family']=fam; z['candidate_score']=z[f'score_{fam}']; z['predicted_return']=model.predict(z[cols].fillna(0)); vp.append(z[['date','symbol','regime','family','candidate_score','predicted_return','btst_return']])
        vp=pd.concat(vp,ignore_index=True); routing,rstats=choose_regimes(vp,families,q,maxpr)
        fallback=vp[vp.candidate_score>=.60].groupby('family').btst_return.mean().sort_values(ascending=False).head(maxpr).index.tolist() or [families[0]]
        for regime,chosen in routing.items(): route_rows.append({'fold':fid,'regime':regime,'families':','.join(chosen)})
        parts=[]
        for fam,(model,cols) in models.items():
            z=test.copy(); z['family']=fam; z['candidate_score']=z[f'score_{fam}']; z['predicted_return']=model.predict(z[cols].fillna(0)); parts.append(z)
        pred=pd.concat(parts,ignore_index=True); pred=pd.concat([g[g.family.isin(routing.get(reg,fallback))] for reg,g in pred.groupby('regime',dropna=False)],ignore_index=True); pred=pred[(pred.candidate_score>=.60)&(pred.predicted_return>0)]
        if not pred.empty: pred_rows.append(pred[['date','symbol','regime','family','candidate_score','predicted_return']])
        t=execute_btst(pred,cfg)
        if not t.empty: t['fold']=fid; all_trades.append(t)
    trades=pd.concat(all_trades,ignore_index=True) if all_trades else pd.DataFrame(); out=Path('docs'); out.mkdir(exist_ok=True)
    if not trades.empty:
        trades.to_csv(out/'btst_composite_btst_oos_trades.csv',index=False); daily=trades.groupby('date').weighted_return.sum().sort_index(); ((1+daily).groupby(daily.index.to_period('M')).prod()-1).rename('monthly_return').reset_index().to_csv(out/'btst_composite_btst_monthly.csv',index=False)
    pd.DataFrame(route_rows).to_csv(out/'btst_composite_btst_routing.csv',index=False)
    if pred_rows: pd.concat(pred_rows,ignore_index=True).to_csv(out/'btst_composite_btst_predictions.csv',index=False)
    pd.DataFrame([metrics(trades)]).to_csv(out/'btst_composite_btst_metrics.csv',index=False); print(metrics(trades))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/composite.yaml'); run(ap.parse_args().config)
