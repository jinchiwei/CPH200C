---
title: "CPH 200C — Problem Set 1 + 2"
subtitle: "Risk Stratification for 30-Day Diabetic Readmission"
author: "Jinchi Wei"
date: "2026-05-14"
---

# Overview

This problem set develops a 30-day hospital-readmission risk model on the UCI **Diabetes 130-US-Hospitals (1999–2008)** dataset, evaluates two frontier LLMs (Claude Opus 4.6 and GPT o1 via the UCSF Versa API) under three prompt formats each, and probes the behavior of the best linear model under an age-based distribution shift. A separate written section discusses the FDA's January 2025 draft guidance on AI-enabled device software functions and one selected public comment.

All experiment artifacts live in `exp/20260512_pset2_full/` and `exp/20260513_auc_ceiling/`; importable modules live in `src/`. Code: `git@github.com:jinchiwei/CPH200C.git`.

# 1.1 — Data Exploration

**Cohort.** Starting from 101,766 encounter records, we apply the SDG+14-canonical filters: (a) drop expired / hospice discharge dispositions (`discharge_disposition_id ∈ {11,13,14,19,20,21}`), and (b) keep one row per patient (first encounter) to prevent leakage when the same patient recurs. This yields **n = 69,990 unique patients**, with a 30-day-readmission positive rate of **7.4 %**.

**Findings.** Figure 1 plots the 30-day readmission rate (with Wilson 95 % CIs) stratified by age, gender, and race.

- **Age** is the dominant driver. The rate rises monotonically from **0.7 %** in `[0–10)` to **9.1 %** in `[80–90)`, then falls slightly in `[90–100)` (likely a survival / discharge-disposition artifact: very-elderly patients with poor short-term prognoses often discharge to hospice and are removed by the SDG+14 filter).
- **Gender** shows essentially no effect: 7.4 % male vs 7.4 % female (the `Unknown/Invalid` cell is `n = 3` and ignored).
- **Race** is mostly flat after accounting for CI overlap: Caucasian and Asian are highest (~7.7 %), AfricanAmerican 6.9 %, Hispanic 6.6 %, Other 5.5 %, missing (`?`) 5.3 %. Sample sizes for Asian (n = 495), Hispanic (n = 1,493), and Other (n = 1,143) make their differences imprecise.

**Headline correlate.** Older age is by far the cleanest signal; race and gender carry little univariate signal in this cohort.

![Figure 1. 30-day readmission rates by age, gender, race (Wilson 95% CIs).](figs/01_eda_readmission_rates.png)

# 1.2 — Model Development

**Setup.** We trained an initial three-family suite (linear, gradient-boosted trees, MLP) on a deduplicated 70 / 15 / 15 train / val / test split. The held-out test set has **n = 10,499**, sampled to match the overall 7.4 % positive prevalence. The split is deterministic (`random_state=42`) and patient-grouped by construction since `clean_and_filter` retains the first encounter per `patient_nbr` (the dedupe means each row is a unique patient — group-aware splitting reduces to a simple stratified split). Hyperparameters were selected on val; held-out test AUC is the primary metric, reported with **DeLong 95 % CIs** [CM04]. We additionally ran a broader autoresearch sweep across 22 candidate configurations to characterise the dataset's modeling ceiling.

**Preprocessing.** Numeric features (age midpoint, length of stay, utilization counts) are kept as-is. Low-cardinality categoricals (race, gender, age bin, three admission/discharge IDs, payer code, medical specialty, max-glucose, A1c, change, diabetesMed, 21 medication-change columns) are one-hot encoded. The diagnosis codes `diag_1`, `diag_2`, `diag_3` are collapsed to the nine SDG+14 buckets (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other) for the baseline matrix; one variant (the headline model) keeps raw ICD-9 codes as native categoricals to test whether the SDG+14 aggregation discards predictive signal.

**Headline result.**

Only 3 ICD-9 encodings are used across the whole leaderboard: **9-bucket SDG+14 (OHE)** — the default for 7 of 10 rows; **raw codes as native categoricals** (CatBoost's specialty); and **raw codes, OOF target-encoded** (the foundation-model-friendly version). Pairing rows by encoding lets you read the encoding effect cleanly: TabPFN-v3 is 0.004 *higher* on bucketed than on TE'd, and TabICL is 0.003 *higher* on TE'd than on bucketed.

| Rank | Model | ICD-9 encoding | Test AUC | 95 % DeLong CI |
|---:|---|---|---:|---|
| **1** | **CatBoost** | **raw codes (native cat)** | **0.7037** | 0.685–0.722 |
| 2 | TabPFN-v3 (full 49k) | 9-bucket SDG+14 (OHE) | 0.7025 | 0.683–0.722 |
| 3 | TabPFN-v3 (full 49k) | raw codes (OOF target-encoded) | 0.6987 | 0.680–0.718 |
| 4 | LightGBM dart | 9-bucket SDG+14 (OHE) | 0.6969 | 0.678–0.716 |
| 5 | CatBoost (ablation) | 9-bucket SDG+14 (OHE) | 0.6950 | 0.676–0.714 |
| 6 | TabICL (full 49k) | raw codes (OOF target-encoded) | 0.6948 | 0.676–0.714 |
| 7 | TabICL (full 49k) | 9-bucket SDG+14 (OHE) | 0.6921 | 0.673–0.711 |
| 8 | XGBoost early-stop | 9-bucket SDG+14 (OHE) | 0.6904 | 0.671–0.710 |
| 9 | LogReg (L1) — best linear | 9-bucket SDG+14 (OHE) | 0.6894 | 0.670–0.708 |
| 10 | LogReg (L1, + A1c × diag_1) | 9-bucket SDG+14 (OHE) | 0.6888 | 0.670–0.708 |
| 11 | MLP (128-64-32) | 9-bucket SDG+14 (OHE) | 0.6884 | 0.669–0.708 |

**Ablation — is the win the model or the encoding?** Rows 1 and 5 isolate this question by running the same CatBoost recipe on the two different feature representations. CatBoost on bucketed features (0.6950) is essentially **tied with LightGBM dart on bucketed features (0.6969)** — overlapping CIs, no statistically distinguishable gap. **The +0.013 AUC lift the champion gets over LightGBM is almost entirely from the raw-ICD-9 encoding (+0.009), not from CatBoost-the-algorithm.** The champion is a CatBoost run because CatBoost's *categorical handling* is uniquely mature, not because gradient-boosted trees beat anything here.

**Best overall: CatBoost with raw ICD-9 codes as native categoricals**, test AUC = **0.704** (95 % CI 0.685–0.722). This is the only configuration in our 22-candidate sweep that crosses 0.70 with a verified ROC curve.

![Figure 2. CatBoost + raw ICD-9 — ROC curve with 95 % bootstrap CI band (turquoise = champion).](../20260513_auc_ceiling/figs/roc_4panel_catboost.png)

![Figure 3. TabPFN-v3 (full 49k) — ROC curve with 95 % bootstrap CI band (deeppink = runner-up).](../20260513_auc_ceiling/figs/roc_4panel_tabpfn_v3.png)

![Figure 4. All 26 candidates — test AUC with 95 % DeLong CIs. Top model (turquoise), runner-up (deeppink), third (gold).](../20260513_auc_ceiling/figs/forest_all.png)

**Interpretation — three findings worth naming.**

1. **The ICD-9 representation is the binding constraint, not the model family.** Replacing the 9-bucket aggregation with raw ICD-9 codes lifted CatBoost from the 0.69 cluster to 0.7037 (+0.013). The same data shift hurt XGBoost (`enable_categorical=True`, AUC 0.6643, −0.026) and LightGBM (native cat, AUC 0.6612, −0.036) — both have less mature high-cardinality support than CatBoost's ordered target-encoding. The signal is in the codes; it only surfaces when the encoder is mature.
2. **Foundation models are competitive but don't dominate at this scale.** TabPFN-v3 (0.7012) trails CatBoost by 0.003 with overlapping CIs. TabICL benefits from extra training rows; TabPFN-v3 ignores its pretraining cap with `ignore_pretraining_limits=True` and uses the full 49k context. On a recent medRxiv benchmark across 12 clinical tabular tasks, TabPFN beat the best classical baseline only ~17 % of the time — consistent with what we see here.
3. **MLP / LogReg / bucketed-XGBoost are bunched tightly between 0.688 and 0.691** — the bucketed-feature ceiling, with no statistically distinguishable winner inside that band.

# 1.3 — LLM Evaluation (Claude Opus 4.6 and GPT o1 via Versa)

**Setup.** Each model receives one user turn containing a clinical chart snapshot and an instruction to reply with `{"p": <float 0-1>}`. The float we parse back is the predicted readmission probability; AUC + 95 % CI follows from standard scoring against the same patient labels. Failed parses (model returns prose instead of JSON) are dropped.

**Both LLMs are evaluated on the full section-1.2 test set (n = 10,499 patients)** by chunking into SLURM arrays of 50 (Opus) and 100 (o1) parallel jobs, each running its slice through all three prompt formats. After a second retry pass to recover Bedrock-throttled calls from the first attempt, parse coverage:
- **Claude Opus 4.6:** **93–95 % per format** (9,769–9,996 valid responses out of 10,499). The losses are Bedrock infrastructure errors (transient throttling under our 50-way parallelism) — when Opus's API call succeeds, the model returns a parseable `{"p": <float>}` on **100 % of patients** (zero refusals across 26 k successful calls).
- **GPT o1 (2024-12-17):** **99.7 % per format** (10,459–10,476 valid responses out of 10,499). o1's 0.3 % loss is purely model-side: terse refusals like `"I'm sorry, but I can't comply with that."` or longer hedged disclaimers. Note that this is a *model behavior* failure, not an infrastructure failure — retries wouldn't recover these.

Both AUCs therefore have tight DeLong CIs (~±0.01). The supervised-vs-LLM head-to-head table that follows still uses the n = 300 sub-sample for visual continuity with the earlier sub-sample analysis, but the full-test LLM AUCs are the headline numbers in the table below.

**Three prompt formats:**

1. **Structured** — JSON dict of `{feature: value}` pairs covering 24 fields (demographics, encounter, utilization counts, three primary diagnoses with raw ICD-9 codes, A1c / max-glucose results, key medications).
2. **Natural-language** — narrative paragraph describing the same patient ("A {race} {gender} aged {age_bin} was admitted under the {admission_type} pathway to the {service} service for {LOS} day(s)…").
3. **Abbreviated** — clinical shorthand (`Pt: F, age 60-70 / LOS: 5d / A1c: >8 | MaxGlu: None / Dx1: 250.83 / Meds: ins Steady, met No; Rx change: Ch / Util: 13 meds, 9 dx, IP 1 | ER 0 | OP 0`).

**Results.**

| Prompt format | Claude Opus 4.6 (n ≈ 10k, full test) | GPT o1 (n ≈ 10.5k, full test) |
|---|:---:|:---:|
| structured | **0.653 (0.634–0.672)** ← Opus BEST (tied) | 0.595 (0.574–0.615) |
| natural | 0.617 (0.598–0.636) | **0.655 (0.635–0.675)** ← o1 BEST |
| abbreviated | **0.651 (0.631–0.670)** ← Opus BEST (tied) | 0.633 (0.614–0.653) |

**Apples-to-apples comparison.** All entries below are on the **same n = 300** sub-sample, for fair head-to-head with o1.

| Model | AUC | 95 % CI |
|---|---:|---|
| **CatBoost (supervised, raw ICD-9)** | **0.760** | 0.668–0.852 |
| XGBoost (supervised) | 0.738 | 0.641–0.835 |
| LogReg (supervised) | 0.720 | 0.609–0.830 |
| Opus 4.6 — abbreviated (sub-sample) | 0.734 | 0.633–0.836 |
| o1 — abbreviated | 0.695 | 0.589–0.800 |
| Opus 4.6 — natural (sub-sample) | 0.708 | 0.621–0.796 |
| o1 — natural | 0.653 | 0.520–0.787 |
| Opus 4.6 — structured (sub-sample) | 0.675 | 0.581–0.769 |
| o1 — structured | 0.602 | 0.465–0.740 |

**Comparison to trained models.** Reading the sub-sample numbers and the full-test numbers together produces a sharper picture than either alone:

- **The n = 300 sub-sample was optimistic for both LLMs.** At sub-sample (CI ±0.10), the best Opus prompt format (`abbreviated`, AUC 0.734) sits within the CI band of supervised XGBoost (0.738) and LogReg (0.720), suggesting essentially tied performance. **At full-test scale (CI ±0.01)**, Opus's best prompt format drops to **0.653** — clearly behind the supervised models (CatBoost full-test AUC 0.704; gap ~0.05). The "essentially tied" impression from the sub-sample was sampling noise.
- **Opus 4.6 and o1 are essentially tied at full scale**, each landing at ~0.65 on its respective best prompt format (Opus = `structured` 0.653 ≈ `abbreviated` 0.651; o1 = `natural` 0.655). The n = 300 finding that "Opus dominates o1 across formats" was sample-size noise — once we have CI ±0.01 instead of ±0.10, the two frontier LLMs are statistically indistinguishable.
- **The two LLMs prefer different prompt formats.** Opus is essentially flat across `structured` (0.653) and `abbreviated` (0.651), with `natural` meaningfully worse (0.617). o1 inverts this — `natural` is best (0.655), `abbreviated` is mid (0.633), `structured` worst (0.595). The takeaway is more nuanced than "abbreviated is the magic format" suggested by the n = 300 numbers: **format matters, but the preferred format is model-specific**.
- **The two parse-coverage failure modes are different.** Opus's losses (5–7 % per format after retries) are **Bedrock-side API throttling** — when the call succeeds, the model returns parseable JSON on **100 %** of patients (zero refusals across 26 k successful calls). o1's losses (0.3 %) are **model-side refusals** — terse `"I'm sorry, but I can't comply with that."` or hedged paragraphs. For deployment, Opus's failure mode is fixable with exponential backoff + retry; o1's failure mode requires a fallback rule.
- **Both LLMs trail supervised by ~0.05 AUC** — best LLM (either model, best format) ≈ 0.655 vs CatBoost 0.704. The gap is roughly the same size as the +0.013 lift CatBoost gets from raw-ICD-9 encoding vs the bucketed feature set, which suggests *feature representation* is a real bottleneck for the LLMs too: they see only the 24 fields we hand them, no patient history embedding, no cross-encounter context.
- **The supervised models remain preferable in deployment** because they are orders of magnitude cheaper to score (a CatBoost forward pass is microseconds vs ~3–9 sec per LLM call), fully reproducible (deterministic given the same seed), inspectable (coefficient and feature-importance views — see Section 1.4), and produce a valid probability on 100 % of patients.

![Figure 5. Claude Opus 4.6 — ROC for Opus's best prompt format (`abbreviated`), full section-1.2 test set (n = 9,869 valid parses out of 10,499 after retry), 95 % bootstrap CI band (amber).](../20260513_auc_ceiling/figs/roc_4panel_opus46.png)

![Figure 6. GPT o1 (2024-12-17) — ROC for o1's best prompt format (`natural`), full section-1.2 test set (n = 10,476 valid parses out of 10,499), 95 % bootstrap CI band (blueviolet).](../20260513_auc_ceiling/figs/roc_4panel_o1.png)

# 1.4 — Distribution Shift (Age <50 vs ≥50)

**Setup.** We re-fit a logistic-regression model (the best linear family from 1.2) on patients in the **<50** age bins only (`[0–10)…[40–50)`), select `C` on a held-out val split from within <50, then evaluate on:

- a 20 % held-out test slice of the **<50** cohort (in-distribution test), and
- the *entire* **≥50** cohort as a shifted target population.

**Results.**

| Evaluation cohort | AUC (95 % DeLong CI) |
|---|---|
| <50 held-out test | 0.683 (0.632–0.735) |
| ≥50 (shifted population) | 0.628 (0.619–0.637) |
| ΔAUC (<50 − ≥50) | **+0.055** |

**Top-5 features pushing 30-day risk *up* in the <50 model:** `number_inpatient` (β=+0.438), `number_diagnoses` (β=+0.234), `diag_2_bucket_Diabetes` (β=+0.181), `discharge_disposition_id_22` (β=+0.153), `discharge_disposition_id_2` (β=+0.138).

**Largest standardized mean differences (≥50 minus <50):** `age_70-80` (SMD +0.69), `age_60-70` (+0.64), `payer_code_MC` (+0.58), `number_diagnoses` (+0.56), `age_50-60` (+0.55) — the ≥50 population is older (tautologically), has more diagnoses on the problem list, and is overwhelmingly on Medicare; the <50 cohort over-indexes on Pediatrics/Ob-Gyn specialties and on injury / pregnancy-related primary diagnoses.

**Why does performance degrade by 0.055?**

- **Outcome prevalence shifts.** The <50 group has a lower 30-day readmission base rate (5.6 %) than ≥50 (7.7 %). A model calibrated on the lower-prevalence cohort will mis-calibrate on the higher-prevalence one — though AUC, being rank-based, is somewhat insulated.
- **Feature mix shifts.** Older patients have more prior inpatient visits, more medications, and a different primary-diagnosis mix (circulatory disease dominates ≥50; the <50 cohort has more genitourinary / injury / pregnancy-adjacent admissions). The relative weight the <50-trained model puts on, say, `number_inpatient` becomes a poor estimate of its true coefficient in the ≥50 population.
- **Implication.** The ΔAUC of +0.055 quantifies how much we should discount the <50 model's apparent skill before deploying it on the broader population. In a regulatory frame (Section 2 below) this is exactly the kind of subgroup-performance drop the FDA's January 2025 draft guidance asks sponsors to surface.

![Figure 7. AUC under <50 → ≥50 shift, and corresponding outcome prevalence by stratum.](figs/04_shift_auc_and_rates.png)

![Figure 8. Top features in the <50 LogReg L2 model — signed coefficients sorted by magnitude. Turquoise = β < 0 (lowers 30-day risk); deeppink = β > 0 (raises 30-day risk). Twelve features shown per direction.](figs/04_shift_feature_importances.png)

**Sidebar — does the best overall model handle the same shift better?** As a follow-up to the PS's "best linear" question, we also re-fit the §1.2 champion (CatBoost with raw ICD-9) on the <50 cohort and evaluated it on the same in-distribution and shifted populations:

| Cohort | LogReg L2 (PS answer) | CatBoost raw-ICD9 (best overall) |
|---|---:|---:|
| <50 held-out test | 0.683 (0.632–0.735) | **0.688** (0.635–0.741) |
| ≥50 (shifted population) | 0.628 (0.619–0.637) | **0.639** (0.631–0.648) |
| ΔAUC | +0.055 | **+0.048** |

CatBoost is *marginally* more robust to the shift (ΔAUC +0.048 vs +0.055) and edges out LogReg in both cohorts, but the difference is small relative to either model's CI width. The take-away is that the shift is a **dataset-level distributional gap**, not a model-capacity gap — every model we tested loses ~0.05 AUC when moved from <50 to ≥50, regardless of family.

# 2 — FDA Request for Comments

## 2.1 Notable parts of the FDA draft guidance (Jan 7, 2025)

The draft guidance — *Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations* (FDA-2024-D-4488) — is the FDA's first single, consolidated framework that ties pre-market submission contents to a **total product lifecycle (TPLC)** view of AI-enabled devices. Two parts stand out:

1. **Predetermined Change Control Plans (PCCPs) become the load-bearing primitive.** The guidance operationalizes PCCPs as the mechanism through which a manufacturer can pre-specify the modifications (training data refresh, retraining cadence, drift-triggered updates, etc.) that they will be permitted to make to an authorized AI device without filing a new 510(k) / PMA. This essentially trades up-front specificity for post-market flexibility, which is a meaningful philosophical shift: the burden of validation moves from "freeze the algorithm at clearance" to "describe and bound how the algorithm will be allowed to evolve." For risk-stratification models like the one in this problem set, a clinically deployed version would need its PCCP to spell out things such as the populations the model may be retrained on, the metrics that gate a refresh, and the rollback policy if a refresh degrades performance on a held-out monitoring cohort.

2. **Explicit bias / representativeness expectations.** Section V calls for documentation that *patient subgroup performance* (race, ethnicity, sex, age, payor mix, geography, comorbidity) be reported alongside aggregate metrics, with attention to whether training and validation data cover the device's intended-use population. This is a direct response to the now-widely-documented pattern of AI tools underperforming on under-represented subgroups, and dovetails with the distribution-shift analysis in Section 1.4 of this problem set: the regulatory expectation is that a sponsor cannot ship an AI device that has only been validated on, say, the <50 population, and then deploy it on a ≥50 cohort without quantifying and disclosing the shift.

## 2.2 One public comment — National Health Council (NHC)

**Author / sector.** The [National Health Council](https://nationalhealthcouncil.org/letters-comments/nhc-submits-comments-on-fda-draft-guidance-for-ai-ml-enabled-medical-devices/) is a ~100-year-old multi-stakeholder coalition of >170 organizations spanning patient-advocacy groups, professional associations, providers, researchers, payers, and biopharmaceutical / device manufacturers. They sit unusually at the intersection of **patient advocacy** and **industry** — patient groups dominate the membership rolls, but the council's policy positions are negotiated with industry seats at the table. Their public comments therefore tend to read as patient-protective but operationally aware of manufacturer constraints.

**What they advocate for.** Three recommendations dominate their letter:

- **"Algorithmovigilance" — modeled on pharmacovigilance.** NHC wants the guidance to require structured post-market surveillance of model behavior, including standardized reporting of malfunctions, drift, and clinically significant performance changes. They argue this is the AI analogue of the adverse-event reporting infrastructure that exists for drugs.
- **Subgroup performance reporting.** They press for sponsors to disaggregate validation results by patient characteristics ("break down key patient characteristics … and illustrate how those characteristics reflect the scope of the device's intended use"), explicitly framing this as both an equity and a generalizability concern.
- **Cybersecurity and audit trails.** They call for explicit cybersecurity risk assessments across the full lifecycle and machine-readable audit logs of system events — partially a patient-safety position, partially an industry-friendly nudge toward concrete technical artifacts that map cleanly into engineering workflows.

**Alignment with their position.** The NHC's stance is *supportive but demanding* — they "commend the FDA" while submitting a fairly long technical wishlist. The asks line up with their multi-stakeholder posture: the algorithmovigilance and subgroup-reporting recommendations reflect patient-advocacy concerns about safety and equity; the cybersecurity / audit-trail recommendations reflect both patient-safety and industry interest in clear technical compliance pathways. None of their recommendations would meaningfully *expand* the scope of what FDA regulates — they want the existing framework strengthened rather than re-imagined. This is consistent with NHC's history of collaborative-but-substantive engagement with FDA dockets.

# 3 — Feedback

The problem set took approximately **16 hours** of focused work, with the bulk going to (a) preprocessing-pipeline design (handling missingness, ICD-9 collapse, leak-safe splits), (b) the LLM prompt-format design (the rest of 1.3 is mostly an evaluation harness with caching), and (c) the AUC-ceiling autoresearch sweep — 22 candidates across 5 model families with parallel SLURM submission, alignment-bug debugging, and a forest-plot consolidation pass. LLM use was substantial but focused: Claude was used to scaffold the autoresearch SLURM templates, debug a TabPFN-v3 alignment ghost (saved preds gave AUC 0.49 vs in-memory 0.70 — root-caused as feature-builder seed-stratification interaction; runtime guard now saves `y_te_aligned.npy` alongside `preds_test.npy` and refuses to declare success unless the disk-roundtrip AUC matches), and accelerate boilerplate for the multi-model recipes. Code was always read and edited by hand before committing.
