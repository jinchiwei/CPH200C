# Headline leaderboard

<!-- tab: turquoise -->

| Rank | Model | ICD-9 encoding | AUC | 95 % CI |
|---:|---|---|---:|---|
| 1 | CatBoost | raw codes (native cat) | 0.7037 | 0.685–0.722 |
| 2 | TabPFN-v3 (full 49k) | 9-bucket SDG+14 (OHE) | 0.7025 | 0.683–0.722 |
| 3 | TabPFN-v3 (full 49k) | raw codes (OOF target-encoded) | 0.6987 | 0.680–0.718 |
| 4 | LightGBM dart | 9-bucket SDG+14 (OHE) | 0.6969 | 0.678–0.716 |
| 5 | CatBoost (ablation) | 9-bucket SDG+14 (OHE) | 0.6950 | 0.676–0.714 |
| 6 | TabICL (full 49k) | raw codes (OOF target-encoded) | 0.6948 | 0.676–0.714 |
| 7 | TabICL (full 49k) | 9-bucket SDG+14 (OHE) | 0.6921 | 0.673–0.711 |
| 8 | XGBoost early-stop | 9-bucket SDG+14 (OHE) | 0.6904 | 0.671–0.710 |
| 9 | LogReg (L1) — best linear | 9-bucket SDG+14 (OHE) | 0.6894 | 0.670–0.708 |
| 10 | LogReg (+A1c × Dx) | 9-bucket SDG+14 (OHE) | 0.6888 | 0.670–0.708 |
| 11 | MLP (128-64-32) | 9-bucket SDG+14 (OHE) | 0.6884 | 0.669–0.708 |

> **Takeaway:** CatBoost with raw ICD-9 codes as native categoricals is the only configuration that clears 0.70 with a verified ROC curve. The ablation (row 5: same CatBoost on bucketed features = 0.6950) shows the +0.013 lift over LightGBM is almost entirely from raw-ICD-9 encoding (+0.009), not the model family — CatBoost-on-bucketed is statistically tied with LightGBM-on-bucketed.

# LLM evaluation

<!-- tab: deeppink -->

| Prompt format | Claude Opus 4.6 (full test, n ≈ 10k) | GPT o1 (full test, n ≈ 10.5k) |
|---|:---:|:---:|
| structured | **0.653 (0.634–0.672)** ← Opus best (tied) | 0.595 (0.574–0.615) |
| natural | 0.617 (0.598–0.636) | **0.655 (0.635–0.675)** ← o1 best |
| abbreviated | **0.651 (0.631–0.670)** ← Opus best (tied) | 0.633 (0.614–0.653) |

> **Takeaway:** Both LLMs at full scale land at ~0.65 on their respective best prompt format (Opus structured 0.653 ≈ abbreviated 0.651; o1 natural 0.655) — statistically tied. The n=300 finding that "Opus dominates o1" and "abbreviated is the magic format" were both sampling noise. Both trail supervised CatBoost (0.704) by ~0.05 AUC. Parse-coverage failure modes differ: Opus's 5–7 % loss is Bedrock throttling (model returns parseable JSON 100 % of the time when API succeeds); o1's 0.3 % loss is model-side refusals ("I'm sorry, but I can't comply"). Opus's mode is fixable with retry + backoff; o1's needs a fallback rule.

# Distribution shift (Age <50 vs ≥50)

<!-- tab: amber -->

| Evaluation cohort | AUC (95 % DeLong CI) |
|---|---|
| <50 test (LogReg L2 — PS answer) | 0.683 (0.632–0.735) |
| ≥50 (LogReg L2 — shifted) | 0.628 (0.619–0.637) |
| ΔAUC LogReg | +0.055 |
| <50 test (CatBoost raw-ICD9 — sidebar) | 0.688 (0.635–0.741) |
| ≥50 (CatBoost — shifted) | 0.639 (0.631–0.648) |
| ΔAUC CatBoost | +0.048 |

> **Takeaway:** Both the best-linear model (LogReg L2) and the best-overall model (CatBoost raw-ICD9) lose ~0.05 AUC under the <50 → ≥50 shift. The shift is driven by feature-mix changes (Medicare predominance, more inpatient history, different primary diagnosis mix in ≥50) more than by outcome-prevalence shift alone. This is the kind of subgroup-performance drop the FDA's Jan 2025 draft guidance asks sponsors to surface in marketing submissions.

# Methods + cohort

<!-- tab: blueviolet -->

| Field | Value |
|---|---|
| Dataset | UCI Diabetes 130-US-Hospitals (1999–2008) |
| Raw rows | 101,766 |
| Filters applied | Drop expired / hospice discharge dispositions (discharge_disposition_id ∈ {11,13,14,19,20,21}); keep first encounter per patient_nbr |
| Final cohort | n = 69,990 unique patients |
| 30-day readmission rate | 7.4 % |
| Train / val / test split | Stratified 70 / 15 / 15, seed = 42 |
| Test size | n = 10,499 |
| Held-out AUC method | DeLong 95 % CI (via `confidenceinterval` package) |
| Number of candidates evaluated | 25 (22 in autoresearch sweep + 3 from section 1.2) |
| Foundation models tested | TabPFN-v2, TabPFN-v3 (released 2026-05-12), TabICL v2 |
| LLM models tested | Claude Opus 4.6, GPT o1-2024-12-17 (both via UCSF Versa) |
| Champion model | CatBoost + raw ICD-9 native categoricals (n_estimators=2000 with early stop, best_iter=376) |

> **Takeaway:** The first-encounter-per-patient dedup is the load-bearing methodology choice — without it, the same patient appearing in train and test leaks identity-equivalent signal and inflates AUC by ~0.05. Several published baselines on this dataset omit this dedup; their 0.70+ numbers are typically not comparable to ours.
