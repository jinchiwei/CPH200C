# cph200c

Coursework for **CPH 200C: Computational Precision Health Cornerstone** (UC Berkeley, Spring 2026) — Prof. Irene Chen.

## Layout

```
cph200c/
├── data/        # raw inputs (gitignored; see data/README.md to refetch)
├── src/         # importable modules — preprocessing, models, eval
├── exp/         # dated experiment logs: {YYYYMMDD}_{descriptor}/{experimentN}/
├── results/     # milestone handoffs / report PDFs
└── README.md
```

## Problem sets

- **PS1 + PS2 — Risk Stratification** (combined). Predict 30-day readmission on UCI Diabetes 130-US-Hospitals. Linear / tree / NN + LLM eval + distribution shift.

## Environment

```bash
conda create -n cph python=3.12 -y
conda activate cph
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm confidenceinterval visidata duckdb
```
