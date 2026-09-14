# BTST Composite Regime Lab

A separate research engine for **conditional BTST strategy selection**.

## Research thesis

The earlier BTST experiments showed an important asymmetry: individual strategy families could have exceptional months even when their unconditional OOS performance was negative. This project tests whether those conditional edges can be harvested by learning **which strategy works best for which market/stock state** rather than blindly combining all strategies.

Pipeline:

`daily OHLCV -> state/features -> strategy candidates -> regime/archetype -> meta-model predicts strategy return -> select strategy/stock -> portfolio -> OOS metrics`

## Candidate BTST families

- momentum
- mean reversion
- gap continuation
- gap fade
- closing strength
- breakout
- volatility expansion
- relative strength
- trend
- volume anomaly
- hybrid momentum/quality

## Anti-leakage design

- Features are known at the signal close only.
- BTST entry is the **same-session close**; exit is next-session open by default.
- Alternative next-session-close evaluation is retained as a diagnostic, not mixed into selection.
- Strategy outcomes are computed only after the signal date.
- Walk-forward train/validation/test splits are chronological.
- Regime scaler/clustering is fitted on train only and transformed forward.
- Meta-model selection/tuning occurs on validation only.
- Test windows are untouched until final scoring.
- Portfolio selection never uses realized test returns.
- Simultaneous positions share capital; observations are never compounded sequentially.

## Objective

The research target is to investigate whether a robust composite can approach or exceed **30% in individual months** while maintaining positive long-run expectancy and controlled drawdown. 30% monthly return is a target for discovery, **not a hard-coded optimization objective or a guaranteed outcome**.

## Data

The default workflow downloads the same genuine NSE daily source used in the prior BTST research:

` s iddharthhirvaniya/nifty-50-stocks-data-01-jan-2015-to-01-sept-2026 `

and optionally augments it with Yahoo Finance for current/extended symbols. The workflow records the universe and survivorship limitations in its manifest.

## Outputs

The Actions workflow publishes:

- `btst_composite_metrics.csv`
- `btst_composite_monthly.csv`
- `btst_composite_oos_trades.csv`
- `btst_composite_fold_diagnostics.csv`
- `btst_composite_strategy_regime.csv`
- `btst_composite_manifest.json`

## Run

```bash
pip install -r requirements.txt
python src/btst_composite.py --config config/composite.yaml
```

Results must be judged from untouched OOS artifacts and then subjected to robustness tests before any claim of deployability.
