# Methods

This document expands on the statistical methods, feature engineering, model training, and evaluation protocol used in the notebooks.

## Statistical tests

- **Normality**: Jarque–Bera, Shapiro–Wilk  
  - $\text{JB} = \frac{n}{6} (S^2 + \frac{(K-3)^2}{4})$ where $S$ is skewness and $K$ is kurtosis.  
- **Sharpe ratio** (risk‑free rate assumed 0 unless noted):  
  - $\text{Sharpe} = \frac{\mathbb{E}[R]}{\sigma(R)}$  
- **Deflated Sharpe ratio**: accounts for multiple testing/strategy selection.  
- **Volatility & tails**: rolling volatility comparisons; kurtosis to assess fat tails.  
- **Hypothesis testing**: fold‑wise comparisons and significance checks on metric deltas.

## Feature engineering

- **Price features**: returns (1d, multi‑horizon), log returns, price ratios.  
- **Volatility**: realized/rolling volatility; optional GARCH components.  
- **Volume/microstructure**: volume changes, VWAP, OBV‑style summaries.  
- **Technical**: RSI, MACD, Bollinger Bands, Stochastic, MAs, crossovers.  
- **Statistical**: z‑scores, rolling moments, entropy/proxy indicators.

**Selection**: mutual information, **RFE**, and **SHAP** to improve stability and interpretability.

## Modeling

- **Gradient boosting**: XGBoost / LightGBM / CatBoost (with tuned hyperparameters).  
- **Tree ensembles**: Random Forest, Extra Trees (for baseline comparisons).  
- **SVM**: non‑linear baseline for margin‑based separation.

## Evaluation protocol

- **Time‑series cross‑validation**: expanding/rolling windows; no shuffling.  
- **Walk‑forward**: train on past, test on the next segment; repeat across the sample.  
- **Leakage control**: strict cutoffs for target horizon; indicators computed using past‑only windows.  
- **Cost model**: transaction cost slippage in backtests for strategy fairness.

## Risk management in backtests

- Position sizing rules; stop‑losses/drawdown guards (where applicable).  
- Turnover and trade frequency monitoring.  
- Post‑cost returns reported alongside pre‑cost metrics.

See notebooks 03–06 for full, executable detail.
