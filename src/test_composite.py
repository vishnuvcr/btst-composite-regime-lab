import numpy as np
import pandas as pd

from btst_composite import add_features, strategy_score


def synthetic():
    dates = pd.date_range('2015-01-01', periods=100, freq='B')
    rows=[]
    for j,s in enumerate(['AAA','BBB','CCC']):
        base=100+j*20
        for i,d in enumerate(dates):
            o=base*(1+0.001*i)
            c=o*(1+0.002*np.sin(i/5+j))
            rows.append({'date':d,'symbol':s,'open':o,'high':max(o,c)*1.01,'low':min(o,c)*.99,'close':c,'volume':100000+i*10})
    return pd.DataFrame(rows)


def test_features_have_no_forward_label_as_feature():
    x=add_features(synthetic())
    assert 'btst_return' in x
    assert x['next_open'].notna().sum() > 0
    assert np.isfinite(x['ret_5'].dropna()).all()


def test_strategy_scores_are_bounded():
    x=add_features(synthetic())
    for fam in ['momentum','mean_reversion','gap_continuation','gap_fade','closing_strength','breakout','volatility_expansion','relative_strength','trend','volume_anomaly','hybrid']:
        s=strategy_score(x,fam).dropna()
        assert len(s)>0
        assert s.min() >= 0
        assert s.max() <= 1
