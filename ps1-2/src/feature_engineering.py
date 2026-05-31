"""Slim feature representation for the diabetes readmission task.

Hypothesis: model AUCs are clustering at 0.68-0.70 not because models are
capacity-limited but because the feature representation has noise / multicollinearity.
This builds a slimmer alternative:

    * Keep only 8 drugs with >1% non-"No" usage; drop 13 near-constant ones.
    * Replace 21 drug OHEs with: per-drug OHE for the 8 active drugs, plus
      `num_drug_changes` (count of drugs with Up/Down/Steady).
    * Bucket `medical_specialty` to 10 high-level groups.
    * Bucket `discharge_disposition_id` to 6 semantic groups.
    * Bucket `admission_source_id` to 5 groups.
    * Log1p-transform skewed utilization counts.
    * Drop `payer_code` (high-cardinality, mostly missing, weak signal).

Usage:
    from feature_engineering import build_features_slim
    X, y, names, pids = build_features_slim(df_cleaned)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 8 drugs with >1% non-"No" usage in the cohort
ACTIVE_DRUGS = [
    "metformin", "insulin", "glipizide", "glyburide",
    "pioglitazone", "rosiglitazone", "glimepiride", "repaglinide",
]

# all 21 drug columns (for the change count)
ALL_DRUGS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

AGE_MIDPOINTS = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}


# ----- bucketers ------------------------------------------------------------

# Discharge dispositions (from IDs_mapping.csv) — semantic 6-bucket grouping
DISCHARGE_BUCKETS = {
    # Home / self-care
    1: "Home", 6: "Home", 8: "Home",
    # Transferred to inpatient / SNF / rehab
    2: "TransferInpatient", 3: "TransferInpatient", 4: "TransferInpatient",
    5: "TransferInpatient", 9: "TransferInpatient", 15: "TransferInpatient",
    22: "TransferInpatient", 23: "TransferInpatient", 24: "TransferInpatient",
    27: "TransferInpatient", 28: "TransferInpatient", 29: "TransferInpatient",
    30: "TransferInpatient",
    # Outpatient referral
    16: "OutpatientReferral", 17: "OutpatientReferral",
    # Left AMA / unknown
    7: "LeftAMA", 25: "Unknown", 26: "Unknown",
    # Still patient
    12: "StillInpatient",
    # NULL
    18: "Unknown",
}
# (Expired/hospice = 11,13,14,19,20,21 already removed at clean_and_filter.)

# Admission sources -> 5 buckets
ADMISSION_SOURCE_BUCKETS = {
    1: "Referral", 2: "Referral", 3: "Referral",
    4: "TransferFromHospital", 5: "TransferFromHospital", 6: "TransferFromHospital",
    10: "TransferFromHospital", 18: "TransferFromHospital",
    22: "TransferFromHospital", 25: "TransferFromHospital", 26: "TransferFromHospital",
    7: "EmergencyRoom",
    8: "EmergencyRoom",  # Court/Law - treat as ER-adjacent
    11: "Newborn", 12: "Newborn", 13: "Newborn", 14: "Newborn",
    23: "Newborn", 24: "Newborn",
    9: "Unknown", 15: "Unknown", 17: "Unknown", 20: "Unknown", 21: "Unknown",
    19: "Unknown",
}

# Medical specialties -> 10 buckets (handful of common ones, everything else -> Other)
SPECIALTY_BUCKETS = {
    "InternalMedicine": "Medicine",
    "Family/GeneralPractice": "FamilyPractice",
    "Cardiology": "Cardiology",
    "Surgery-General": "Surgery",
    "Surgery-Cardiovascular/Thoracic": "Surgery",
    "Surgery-Cardiovascular": "Surgery",
    "Surgery-Vascular": "Surgery",
    "Surgery-Neuro": "Surgery",
    "Surgery-Plastic": "Surgery",
    "Surgery-Thoracic": "Surgery",
    "Orthopedics": "Surgery",
    "Orthopedics-Reconstructive": "Surgery",
    "Emergency/Trauma": "Emergency",
    "Nephrology": "Nephrology",
    "Pulmonology": "Pulmonology",
    "Psychiatry": "Psychiatry",
    "Psychiatry-Addictive": "Psychiatry",
    "Psychiatry-Child/Adolescent": "Psychiatry",
    "ObstetricsandGynecology": "OBGYN",
    "Obsterics&Gynecology-GynecologicOnco": "OBGYN",
    "Gynecology": "OBGYN",
    "Obstetrics": "OBGYN",
    "Pediatrics": "Pediatrics",
    "Pediatrics-Endocrinology": "Pediatrics",
    "?": "Missing",
}


def _icd9_to_bucket(code):
    if pd.isna(code) or code == "?":
        return "Other"
    s = str(code).strip()
    if s.startswith("250"):
        return "Diabetes"
    if s.startswith(("V", "E")):
        return "Other"
    try:
        num = float(s)
    except ValueError:
        return "Other"
    if 390 <= num < 460 or int(num) == 785: return "Circulatory"
    if 460 <= num < 520 or int(num) == 786: return "Respiratory"
    if 520 <= num < 580 or int(num) == 787: return "Digestive"
    if 800 <= num < 1000: return "Injury"
    if 710 <= num < 740: return "Musculoskeletal"
    if 580 <= num < 630 or int(num) == 788: return "Genitourinary"
    if 140 <= num < 240: return "Neoplasms"
    return "Other"


def build_features_slim(
    df: pd.DataFrame, drop_first: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series]:
    """Slim alternative to preprocessing.build_features.

    Returns (X, y, feature_names, patient_id) — same contract.
    """
    df = df.copy()
    y = df["y_30d"].astype(int)
    patient_id = df["patient_nbr"].copy()

    # age midpoint only (no OHE — saves 9 multicollinear features for linear)
    df["age_midpoint"] = df["age"].map(AGE_MIDPOINTS).astype(float)

    # ICD-9 buckets
    for c in ["diag_1", "diag_2", "diag_3"]:
        df[f"{c}_bucket"] = df[c].apply(_icd9_to_bucket)

    # consolidated discharge / admission source / specialty
    df["discharge_bucket"] = df["discharge_disposition_id"].map(DISCHARGE_BUCKETS).fillna("Other")
    df["adm_source_bucket"] = df["admission_source_id"].map(ADMISSION_SOURCE_BUCKETS).fillna("Other")
    df["specialty_bucket"] = df["medical_specialty"].map(SPECIALTY_BUCKETS).fillna("Other")

    # drug-change count: how many drugs were Up/Down/Steady (i.e., not 'No')
    drug_active = df[ALL_DRUGS].apply(lambda col: (col != "No").astype(int))
    df["num_drugs_active"] = drug_active.sum(axis=1)
    # changes specifically (Up or Down)
    drug_changed = df[ALL_DRUGS].apply(lambda col: col.isin(["Up", "Down"]).astype(int))
    df["num_drugs_changed"] = drug_changed.sum(axis=1)

    # log1p the heavy-tail utilization counts
    for c in ["number_outpatient", "number_emergency", "number_inpatient",
              "num_lab_procedures", "num_medications"]:
        df[f"{c}_log1p"] = np.log1p(df[c].astype(float))

    numeric_cols = [
        "age_midpoint", "time_in_hospital",
        "num_procedures", "number_diagnoses",
        "num_drugs_active", "num_drugs_changed",
        "num_lab_procedures_log1p", "num_medications_log1p",
        "number_outpatient_log1p", "number_emergency_log1p", "number_inpatient_log1p",
    ]

    cat_cols = [
        "race", "gender",
        "admission_type_id",
        "discharge_bucket", "adm_source_bucket", "specialty_bucket",
        "max_glu_serum", "A1Cresult",
        "change", "diabetesMed",
        "diag_1_bucket", "diag_2_bucket", "diag_3_bucket",
    ] + ACTIVE_DRUGS  # only the 8 active drug columns

    for c in cat_cols:
        df[c] = df[c].astype(str)

    X_num = df[numeric_cols].astype(float)
    X_cat = pd.get_dummies(df[cat_cols], drop_first=drop_first, dtype=float)
    X = pd.concat([X_num, X_cat], axis=1)
    X.columns = [c.replace(" ", "_").replace("[", "").replace(")", "").replace("/", "_") for c in X.columns]
    return X, y, list(X.columns), patient_id


__all__ = ["build_features_slim", "ACTIVE_DRUGS"]
