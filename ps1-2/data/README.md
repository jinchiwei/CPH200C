# `data/`

Raw inputs for problem sets. **Contents are gitignored** — fetch locally with the commands below.

## Problem Set 2 — UCI Diabetes 130-US-Hospitals (1999–2008)

Public benchmark dataset of clinical care at 130 US hospitals + integrated delivery networks for patients diagnosed with diabetes. Goal: predict 30-day hospital readmission.

```bash
cd data/
curl -sSL -o dataset_diabetes.zip \
  https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip
unzip dataset_diabetes.zip
# produces:
#   data/dataset_diabetes/diabetic_data.csv   (101,766 rows × 50 cols)
#   data/dataset_diabetes/IDs_mapping.csv     (decodes 3 ID columns)
```

Reference paper: Strack et al., *BioMed Research International* (2014) — "Impact of HbA1c measurement on hospital readmission rates."

## Course PDFs

Problem-set PDFs (e.g., `pset2.pdf`) are kept locally for reference but excluded from the repo to respect Berkeley's course material policies.
