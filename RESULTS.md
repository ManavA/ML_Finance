# Results & Takeaways

This document points directly to the artifacts that contain the quantitative results and offers a concise narrative so readers can orient quickly. It intentionally **does not duplicate** every figure/table from the notebooks.

## Where the numbers live

- `06_final_summary.ipynb` — master comparison notebook
- `06_final_summary.pdf` — exported report for quick reading
- Intermediate figures/tables are produced by notebooks 04–05

## Reading guide

1. **Baselines vs ML**: start with the tables comparing risk‑adjusted metrics for baselines and ML models. Note walk‑forward fold averages and dispersion.  
2. **Market differences**: compare crypto vs equities plots to observe variance, tail behavior, and drawdown profiles.  
3. **Feature ablations**: look at feature importance/SHAP and RFE results; note stability across folds.  
4. **Robustness**: review statistical tests and any p‑values/CI to understand which differences are statistically meaningful.

## Key patterns to look for (analysis checklist)

- Are ML models consistently ahead **out‑of‑sample**, or only in specific regimes?  
- Do improvements persist **after costs** and **with risk controls**?  
- Which features recur across top models? Are they **market‑specific**?  
- How sensitive are results to **window lengths** and **CV scheme**?  
- Do **deflated Sharpe** or additional risk metrics change the story?

> If you would like me to extract specific tables/plots into this markdown (e.g., top‑k models per market with metric values), let me know and I can bake them in here.
