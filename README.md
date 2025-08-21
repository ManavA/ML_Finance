# Market Strategy & ML Comparison — Crypto vs Equities

This repository contains a fully reproducible research pipeline comparing **baseline trading strategies** and **machine‑learning models** across **cryptocurrency** and **traditional equities**. It includes data ingestion and validation, exploratory analysis, statistical testing, feature engineering, model training, and a final synthesis of results.

> Author: **Manav Agarwal** · 2025

---

## 🔎 What this project does

- Builds a **clean data pipeline** with caching and fallbacks (Polygon, CoinMarketCap, yfinance, etc.).  
- Performs **EDA** and distributional checks (skew/kurtosis, normality tests).  
- Runs **advanced statistical analysis** (Sharpe & deflated Sharpe, hypothesis tests).  
- Engineers **rich feature sets** (price/volatility/volume/technical/statistical).  
- Benchmarks **baseline trading strategies** for cross‑market comparison.  
- Trains **ML models** (XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, SVM) with **time‑series aware CV** and **walk‑forward** evaluation.  
- Summarizes results and takeaways in a final report.

All notebooks are exported to **PDFs** for quick reading.

---

## 📂 Repository layout

```
├── Jupyter Notebooks
│
 
 
 
├
─
─
 
0
0
_
d
a
t
a
_
t
e
s
t
i
n
g
.
i
p
y
n
b


│
 
 
 
├
─
─
 
0
1
_
d
a
t
a
_
a
n
a
l
y
s
i
s
.
i
p
y
n
b


│
 
 
 
├
─
─
 
0
2
_
a
d
v
a
n
c
e
d
_
s
t
a
t
i
s
t
i
c
a
l
_
a
n
a
l
y
s
i
s
.
i
p
y
n
b


│
 
 
 
├
─
─
 
0
3
_
f
e
a
t
u
r
e
_
e
n
g
i
n
e
e
r
i
n
g
_
o
p
t
i
m
i
z
a
t
i
o
n
.
i
p
y
n
b


│
 
 
 
├
─
─
 
0
4
_
b
a
s
e
l
i
n
e
_
a
n
a
l
y
s
i
s
.
i
p
y
n
b


│
 
 
 
├
─
─
 
0
4
_
b
a
s
e
l
i
n
e
_
m
o
d
e
l
s
.
i
p
y
n
b


│
 
 
 
├
─
─
 
0
5
_
m
l
_
m
o
d
e
l
s
_
f
i
n
a
l
.
i
p
y
n
b


│
 
 
 
└
─
─
 
0
6
_
f
i
n
a
l
_
s
u
m
m
a
r
y
.
i
p
y
n
b
├── 00_data_testing.pdf
├── 01_data_analysis.pdf
├── 02_advanced_statistical_analysis.pdf
├── 03_feature_engineering_optimization.pdf
├── 04_baseline_analysis.pdf
├── 05_ml_models_final.pdf
├── 06_final_summary.pdf
└── Appendix - References.pdf
```

### Notebooks (in order)

| Notebook | Title |
|---|---|
| `00_data_testing.ipynb` | Comparison of Strategies and ML Methsd for Cryptocurrency and Traditional Equities. |
| `01_data_analysis.ipynb` | 01. COMPREHENSIVE CRYPTOCURRENCY VS EQUITY MARKET ANALYSIS |
| `02_advanced_statistical_analysis.ipynb` | STATISTICAL ANALYSIS |
| `03_feature_engineering_optimization.ipynb` | 03: Advanced Feature Engineering Optimization for Multi-Asset Trading |
| `04_baseline_analysis.ipynb` | 04. Baseline Trading Strategies & Cross-Market Analysis |
| `04_baseline_models.ipynb` | Baseline Trading Dashboard — Clean & Compact |
| `05_ml_models_final.ipynb` | 05. Machine Learning Models |
| `06_final_summary.ipynb` | 06. Strategy Comparison & Statistical Analysis |

PDF exports of the notebooks are in the repository root for convenience.

---

## 🧰 Environment & setup

> Python ≥ 3.10 recommended

**Core libraries**  
`pandas`, `numpy`, `matplotlib`, `scikit-learn`, `scipy`, `statsmodels`

**Modeling (tree ensembles / boosting)**  
`xgboost`, `lightgbm`, `catboost`

**Time‑series & features**  
`ta` (technical indicators), `shap` (feature attributions), `mlxtend` (RFE and helpers)

**Data**  
`yfinance`, and optionally `requests`‑based clients for Polygon and CoinMarketCap

You can start with:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install pandas numpy matplotlib scikit-learn scipy statsmodels xgboost lightgbm catboost yfinance ta shap mlxtend
```

If you use APIs (Polygon/CMC), place your keys in a local **`.env`** (not committed):

```
POLYGONIO_KEY=...
COINMARKETCAP_KEY=...
```

---

## ▶️ How to run

1. **00_data_testing.ipynb** — verify data sources, caching, and API connectivity.  
2. **01_data_analysis.ipynb** — run exploratory analysis and generate intermediate artifacts.  
3. **02_advanced_statistical_analysis.ipynb** — run hypothesis tests and summary stats.  
4. **03_feature_engineering_optimization.ipynb** — build features and run selection.  
5. **04_baseline_analysis.ipynb** / **04_baseline_models.ipynb** — benchmark baseline strategies.  
6. **05_ml_models_final.ipynb** — train and evaluate ML models with walk‑forward CV.  
7. **06_final_summary.ipynb** — compile comparisons and produce final figures/tables.

Each notebook saves intermediate outputs for the next step (see inline paths).

---

## 📊 Methods (high level)

- **Statistical tests**: normality (Jarque–Bera/Shapiro), volatility & tail behavior, Sharpe & deflated Sharpe.  
- **Cross‑market framing**: crypto (24/7) vs equities (RTH) handled consistently.  
- **Walk‑forward evaluation** with **time‑ordered folds** to avoid look‑ahead.  
- **Feature engineering** across price/vol/volume/technicals/statistics; selection via **mutual information**, **RFE**, and **SHAP**.  
- **Models**: gradient boosting (XGBoost/LightGBM/CatBoost), tree ensembles (RF/ExtraTrees), and SVM baselines.  
- **Risk controls** in strategy backtests: transaction costs, position sizing, drawdown checks.

For complete details, see the PDFs and notebooks (03–06).

---

## ✅ Findings (where to look)

This repo includes a **final synthesis** in `06_final_summary.ipynb` and the exported PDF `06_final_summary.pdf`. Those materials summarize:
- Comparative performance of baseline strategies vs ML models
- Distributional & regime differences between crypto and equities
- Effect of feature sets and selection methods on out‑of‑sample results
- Robustness checks via walk‑forward and statistical testing

> Numbers and plots are preserved in the notebooks and PDFs. If you want these summarized into a single markdown report, see `docs/RESULTS.md` (generated below).

---

## 🔁 Reproducibility

- Deterministic seeds are set where supported.  
- Data pulls are cached; APIs have fallbacks.  
- Time‑based splits ensure no leakage.  
- Each notebook documents its inputs/outputs so you can re‑run end‑to‑end.

---

## 📦 Data

Data comes from public APIs (yfinance) and, optionally, paid APIs (Polygon/CMC). **Do not commit raw data**. Use `.gitignore` to keep local data artifacts out of version control.

---

## 🧭 Roadmap & extensions

- Add **Prophet**/**SARIMAX** baselines for seasonality.  
- Add **deep learning** baselines (LSTM/Temporal Convolution/Transformer) with proper walk‑forward.  
- Expand **risk metrics** (Sortino, Calmar, Omega) and **bayesian** comparisons.  
- Integrate a small **CLI** to reproduce figures from raw data.

---

## 📄 License

Choose a license that matches your goals (e.g., MIT for open research). Add a `LICENSE` file at the repo root.

---

## ✨ Acknowledgments

Thanks to open‑source contributors and data providers that made this work possible.
