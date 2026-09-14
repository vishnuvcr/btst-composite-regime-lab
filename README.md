# BTST Composite Regime Lab

A separate research engine for **conditional BTST strategy selection**. It is intentionally separate from the earlier BTST Strategy Lab so we can test the composite idea without contaminating the original experiments.

## Core idea

The earlier experiments showed that individual BTST strategy families could have extraordinary months while being negative over the full sample. A simple average of those strategies is not the objective. This project asks a stronger question:

> **Given today's market regime and the stock's current state, which strategy family has historically worked best in comparable conditions?**

The router therefore chooses a strategy rather than blindly combining all signals.

`OHLCV -> causal features -> candidate strategy scores -> market regime -> family-specific ML models -> validation-only regime router -> stock selection -> BTST portfolio -> untouched OOS metrics`

## Strategy families

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
- hybrid

## Regime model

Market regimes are learned with K-means using training-period information only. State variables include market momentum, market volatility, market trend, breadth and stock volatility. Stock-state features include short/medium-term returns, gap, candle location/body, ATR, volume anomaly, moving-average distance and cross-sectional ranks.

For each walk-forward fold, the validation period determines which strategy family is preferred **within each regime**. That routing table is then frozen and applied to the next untouched test period.

## Strict BTST execution

The main composite runner uses the conventional overnight BTST interpretation:

- signal and entry: day-t close
- exit: day-t+1 open
- optional ATR stop/target: evaluated using the next session high/low
- if both stop and target are touched in the same daily bar, stop is assumed first
- slippage and transaction costs are charged on both sides
- simultaneous positions share the configured gross exposure

Daily OHLC cannot reveal the exact intraday order of stop/target events; later 5-minute execution refinement is therefore required before any live-use conclusion.

## Anti-leakage rules

- Features use information available by the signal close only.
- Realized BTST returns are labels, never features.
- Regime fitting is train-only and transformed forward.
- Family models are trained only on the historical training window.
- Strategy/regime routing is selected on validation only.
- The test period is never used to choose families, thresholds or model parameters.
- Monthly returns are compounded from daily portfolio returns.
- Survivorship bias and corporate-action/data-quality limitations remain explicit research risks.

## 30% target

The research target is to discover whether the conditional composite can produce **30%+ individual months** without simply overfitting those months. 30% per month is **not guaranteed and is not hard-coded into the model**. A strategy that achieves 30% in-sample and fails untouched OOS is considered a failure.

The real acceptance test is stronger: positive OOS expectancy, acceptable drawdown, stability across years/regimes, cost sensitivity, and performance that survives an unseen final period.

## Run

```bash
pip install -r requirements.txt
python src/btst_composite_btst.py --config config/composite.yaml
```

## Key outputs

- `docs/btst_composite_btst_metrics.csv`
- `docs/btst_composite_btst_monthly.csv`
- `docs/btst_composite_btst_oos_trades.csv`
- `docs/btst_composite_btst_routing.csv`
- `docs/btst_composite_btst_predictions.csv`

The older `src/btst_composite.py` remains as a general composite research engine; the `btst_composite_btst.py` runner is the one to use for the strict overnight BTST experiment.
