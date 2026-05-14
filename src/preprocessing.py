"""Leak-safe preprocessing for the UCI Diabetes 130-US-Hospitals dataset.

Decisions baked in (see KB note "UCI Diabetes 130-US Hospitals — schema and known caveats"):
    * Target: y = (readmitted == "<30") for the 30-day readmission task.
    * Drop rows with discharge_disposition_id in {11,13,14,19,20,21} (expired/hospice).
    * Drop weight (~97% missing).
    * Drop single-valued meds (examide, citoglipton).
    * Keep '?' as own category for race / payer_code / medical_specialty (signal in missingness).
    * Patient-grouped split: keep first encounter per patient (~71k rows) to prevent leakage.
    * ICD-9 collapse to SDG+14 buckets for diag_1/2/3.
    * Age: keep both midpoint (continuous) and the original 10-yr bin (categorical).

Public API:
    load_diabetes(csv_path) -> raw DataFrame (light: dtypes only)
    clean_and_filter(df, dedupe_patients=True) -> filtered DataFrame
    build_features(df, drop_first=True) -> (X, y, feature_names, patient_id)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ----- constants ------------------------------------------------------------

DEFAULT_CSV = Path("/data/rauschecker2/jkw/cph/CPH200C/data/dataset_diabetes/diabetic_data.csv")

EXPIRED_OR_HOSPICE_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

DROP_COLUMNS = [
    "encounter_id",
    "weight",
    "examide",
    "citoglipton",
]

DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Age 10-yr bin midpoints
AGE_MIDPOINTS = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}


def _icd9_to_bucket(code: str) -> str:
    """Collapse ICD-9 primary diagnosis code to SDG+14 buckets.

    SDG+14 bucketing (Table 2 of Strack et al. 2014):
        Circulatory   390-459, 785
        Respiratory   460-519, 786
        Digestive     520-579, 787
        Diabetes      250.xx
        Injury        800-999
        Musculoskel   710-739
        Genitourinary 580-629, 788
        Neoplasms     140-239
        Other         everything else (or missing)
    """
    if pd.isna(code) or code == "?":
        return "Other"
    s = str(code).strip()
    # diabetes-specific (250.xx)
    if s.startswith("250"):
        return "Diabetes"
    # V/E codes -> Other (external causes, supplementary)
    if s.startswith(("V", "E")):
        return "Other"
    try:
        num = float(s)
    except ValueError:
        return "Other"
    if 390 <= num < 460 or int(num) == 785:
        return "Circulatory"
    if 460 <= num < 520 or int(num) == 786:
        return "Respiratory"
    if 520 <= num < 580 or int(num) == 787:
        return "Digestive"
    if 800 <= num < 1000:
        return "Injury"
    if 710 <= num < 740:
        return "Musculoskeletal"
    if 580 <= num < 630 or int(num) == 788:
        return "Genitourinary"
    if 140 <= num < 240:
        return "Neoplasms"
    return "Other"


# ----- load + clean ---------------------------------------------------------

def load_diabetes(csv_path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """Load raw CSV with strings for ICD codes (which can be e.g. '250.83')."""
    return pd.read_csv(csv_path, dtype={"diag_1": str, "diag_2": str, "diag_3": str})


def clean_and_filter(df: pd.DataFrame, dedupe_patients: bool = True) -> pd.DataFrame:
    """Apply SDG+14-canonical filters and clean up obvious junk."""
    df = df.copy()
    # 1. binary target
    df["y_30d"] = (df["readmitted"] == "<30").astype(int)
    # 2. drop expired / hospice (can't be readmitted)
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE_DISPOSITIONS)].copy()
    # 3. drop columns that are useless or > 90% missing
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    # 4. keep first encounter per patient (leak-safety)
    if dedupe_patients:
        df = df.sort_values("patient_nbr").drop_duplicates("patient_nbr", keep="first").copy()
    return df.reset_index(drop=True)


# ----- feature engineering --------------------------------------------------

def build_features(
    df: pd.DataFrame,
    drop_first: bool = True,
    include_diag_buckets: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series]:
    """Convert cleaned dataframe to numeric feature matrix + binary target.

    Returns:
        X: numeric DataFrame ready for sklearn-style models
        y: 0/1 target Series (30-day readmission)
        feature_names: column names of X
        patient_id: patient_nbr Series aligned with X (for group-aware splits)
    """
    df = df.copy()
    y = df["y_30d"].astype(int)
    patient_id = df["patient_nbr"].copy()

    # age midpoint (continuous)
    df["age_midpoint"] = df["age"].map(AGE_MIDPOINTS).astype(float)

    # ICD-9 buckets
    if include_diag_buckets:
        for col in ["diag_1", "diag_2", "diag_3"]:
            df[f"{col}_bucket"] = df[col].apply(_icd9_to_bucket)

    # numeric columns to keep as-is
    numeric_cols = [
        "age_midpoint", "time_in_hospital",
        "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient",
        "number_diagnoses",
    ]

    # categorical columns -> one-hot
    cat_cols = [
        "race", "gender", "age",
        "admission_type_id", "discharge_disposition_id", "admission_source_id",
        "payer_code", "medical_specialty",
        "max_glu_serum", "A1Cresult",
        "change", "diabetesMed",
    ] + DRUG_COLUMNS
    if include_diag_buckets:
        cat_cols += ["diag_1_bucket", "diag_2_bucket", "diag_3_bucket"]
    # filter to actually-present columns
    cat_cols = [c for c in cat_cols if c in df.columns]
    # cast ID-as-categorical (admission_type_id etc.) to str so OHE works clean
    for c in cat_cols:
        df[c] = df[c].astype(str)

    X_num = df[numeric_cols].astype(float)
    X_cat = pd.get_dummies(df[cat_cols], drop_first=drop_first, dtype=float)
    X = pd.concat([X_num, X_cat], axis=1)
    # tidy column names (sklearn doesn't like brackets / spaces in some places)
    X.columns = [c.replace(" ", "_").replace("[", "").replace(")", "").replace("/", "_") for c in X.columns]

    return X, y, list(X.columns), patient_id


__all__ = [
    "DEFAULT_CSV",
    "AGE_MIDPOINTS",
    "DRUG_COLUMNS",
    "load_diabetes",
    "clean_and_filter",
    "build_features",
]
