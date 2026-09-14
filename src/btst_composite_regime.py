from __future__ import annotations
import argparse, glob
from pathlib import Path
import pandas as pd
from btst_composite import FEATURES, add_candidate_scores, add_features, build_folds, fit_family_models, load_config, make_regime_labels, metrics, portfolio_backtest, read_csv_universe


def choose_regimes(vp, families, quantile=0.65, max_per_regime=1):
    rows=[]
    for regime,g in vp.groupby('regime',dropna=False):
        for fam in families:
            z=g[(g.family==fam)&(g.candidate_score>=0.60)]
            if len(z)<20: continue
            z=z.nlargest(max(20,int(len(z)*0.10)),'predicted_return')
            rows.append({'regime':regime,'family':fam,'n':len(z),'mean_return':z.btst_return.mean(),'win_rate':(z.btst_return>0).mean()})
    q=pd.DataFrame(rows)
    if q.empty: return {},q
    routing={}
    for regime,g in q.groupby('regime',dropna=False):
        cutoff=g.mean_return.quantile(quantile)
        routing[regime]=g[g.mean_return>=cutoff].sort_values('mean_return',ascending=False).head(max_per_regime).family.tolist()
    return routing,q


def run(config_path):
    cfg=load_config(config_path); dc=cfg['data']; families=cfg['strategies']['enabled']
    df=read_csv_universe(dc['daily_glob'],dc.get('max_symbols'))
    df=df[(df.date>=pd.Timestamp(dc['start_date']))&(df.date<=pd.Timestamp(dc['end_date']))]
    market=read_csv_universe(dc['market_glob'],1) if glob.glob(dc.get('market_glob',''),recursive=True) else None
    x=add_candidate_scores(add_features(df,market),families)
    folds=build_folds(pd.DatetimeIndex(sorted(x.date.unique())),cfg); seed=int(cfg['research'].get('random_state',42))
    quantile=float(cfg['research'].get('strategy_selection_quantile',0.65)); maxpr=int(cfg['research'].get('max_families_per_regime',1))
    trades_all=[]; routes=[]; predictions=[]
    for fold_id,fold in enumerate(folds,1):
        train=x[(x.date>=fold.train_start)&(x.date<=fold.train_end)].copy(); val=x[(x.date>=fold.validation_start)&(x.date<=fold.validation_end)].copy(); test=x[(x.date>=fold.test_start)&(x.date<=fold.test_end)].copy()
        if train.empty or val.empty or test.empty: continue
        train_reg,_,_=make_regime_labels(train,train,int(cfg['research'].get('n_clusters',6)),seed)
        forward_reg,_,_=make_regime_labels(train,pd.concat([val,test]),int(cfg['research'].get('n_clusters',6)),seed)
        train['regime']=train.date.map(train_reg); val['regime']=val.date.map(forward_reg); test['regime']=test.date.map(forward_reg)
        models=fit_family_models(train,families,FEATURES,seed+fold_id)
        vp=[]
        for fam,(model,cols) in models.items():
            z=val.copy(); z['family']=fam; z['candidate_score']=z[f'score_{fam}']; z['predicted_return']=model.predict(z[cols].fillna(0)); vp.append(z[['date','symbol','regime','family','candidate_score','predicted_return','btst_return']])
        vp=pd.concat(vp,ignore_index=True); routing,rstats=choose_regimes(vp,families,quantile,maxpr)
        fallback=vp[vp.candidate_score>=.60].groupby('family').btst_return.mean().sort_values(ascending=False).head(maxpr).index.tolist() or [families[0]]
        for regime,chosen in routing.items(): routes.append({'fold':fold_id,'regime':regime,'families':','.join(chosen)})
        parts=[]
        for fam,(model,cols) in models.items():
            z=test.copy(); z['family']=fam; z['candidate_score']=z[f'score_{fam}']; z['predicted_return']=model.predict(z[cols].fillna(0)); parts.append(z)
        pred=pd.concat(parts,ignore_index=True)
        pred=pd.concat([g[g.family.isin(routing.get(regime,fallback))] for regime,g in pred.groupby('regime',dropna=False)],ignore_index=True)
        pred=pred[(pred.candidate_score>=.60)&(pred.predicted_return>0)]
        if not pred.empty: predictions.append(pred[['date','symbol','regime','family','candidate_score','predicted_return']])
        t=portfolio_backtest(pred,cfg)
        if not t.empty: t['fold']=fold_id; trades_all.append(t)
    trades=pd.concat(trades_all,ignore_index=True) if trades_all else pd.DataFrame(); out=Path('docs'); out.mkdir(exist_ok=True)
    if not trades.empty:
        trades.to_csv(out/'btst_composite_regime_oos_trades.csv',index=False)
        daily=trades.groupby('date').weighted_return.sum().sort_index(); monthly=((1+daily).groupby(daily.index.to_period('M')).prod()-1).rename('monthly_return').reset_index(); monthly.to_csv(out/'btst_composite_regime_monthly.csv',index=False)
    pd.DataFrame(routes).to_csv(out/'btst_composite_regime_routing.csv',index=False)
    if predictions: pd.concat(predictions,ignore_index=True).to_csv(out/'btst_composite_regime_predictions.csv',index=False)
    pd.DataFrame([metrics(trades)]).to_csv(out/'btst_composite_regime_metrics.csv',index=False)
    print(metrics(trades))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/composite.yaml'); run(ap.parse_args().config)
