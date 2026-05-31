# cph200c

Coursework for **CPH 200C: Computational Precision Health Cornerstone** (UC Berkeley, Spring 2026) — Prof. Irene Chen.

## Layout

Organized by problem set.

```
cph200c/
├── ps1-2/       # Risk Stratification (combined PS1 + PS2)
│   ├── data/    # raw inputs (gitignored; see data/README.md to refetch)
│   ├── src/     # importable modules — preprocessing, models, eval
│   ├── code/    # SLURM batch scripts (FAC cluster)
│   ├── docs/    # branded-report build scripts
│   ├── exp/     # dated experiment logs (gitignored): {YYYYMMDD}_{descriptor}/
│   └── results/ # milestone handoffs / report deliverables
└── ps3/         # Recent Advances in Deployed Clinical Models (presentation)
```

## Problem sets

- **PS1 + PS2 — Risk Stratification** (combined) → `ps1-2/`. Predict 30-day readmission on UCI Diabetes 130-US-Hospitals. Linear / tree / NN + LLM eval + distribution shift.
- **PS3 — Recent Advances in Deployed Clinical Models** → `ps3/`. A 15-min talk: "Three Governing Systems for (Clinical) AI" — how AI deployment differs across the US, Europe, and Asia (policy / funding / technical / landscape), with deployed clinical AI as the test case. Authors: Curtis Chambers · Jinchi Wei. See `ps3/README.md`.

## Environment

```bash
conda create -n cph python=3.12 -y
conda activate cph
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm confidenceinterval visidata duckdb
```
