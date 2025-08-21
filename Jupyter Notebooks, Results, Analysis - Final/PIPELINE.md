# Pipeline Overview

```
[00] Data Testing & Interfaces  -->  [01] EDA & Summaries
                                         |
                                         v
                                 [02] Statistical Analysis
                                         |
                                         v
                             [03] Feature Engineering & Selection
                              /                     \
                             v                       v
                [04] Baseline Strategies]       [05] ML Models]
                              \                     /
                               v                   v
                                 [06] Final Summary]
```

Each notebook documents its inputs and saves outputs for the next step (e.g., pickles or CSV artifacts in a local `artifacts/` or `cache/` directory that you should keep out of version control).

**Reproducibility tips**

- Keep API keys in `.env` and use local caching.  
- Set RNG seeds where supported.  
- Use the same time‑based splits for apples‑to‑apples comparisons.  
- Pin library versions when publishing results.
