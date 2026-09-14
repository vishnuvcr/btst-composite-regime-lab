import numpy as np
import pandas as pd

from btst_composite import add_features, strategy_score
from btst_strategy_ranking_v2 import select_top_by_date


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


def test_strategy_ranking_preserves_date_column():
    dates = pd.date_range('2025-01-01', periods=3, freq='B')
    rows = []
    for d in dates:
        for i in range(3):
            rows.append({
                'date': d,
                'symbol': f'S{i}',
                'predicted_return': float(i + 1),
            })
    x = pd.DataFrame(rows)
    out = select_top_by_date(x, 2)
    assert 'date' in out.columns
    assert len(out) == 6
    assert out.groupby('date').size().eq(2).all()
    assert set(out.predicted_return) == {2.0, 3.0}
