"""Versa Opus 4.6 evaluation harness with three prompt formats.

Builds prompts at row granularity, calls Versa, parses to a probability or binary
label, caches responses to disk so reruns resume.

Three prompt formats per the PS2 spec:
    1. `structured`   -- JSON dict of {feature: value} pairs
    2. `natural`      -- narrative sentence describing the patient
    3. `abbreviated`  -- clinical shorthand (Dx, A1c, etc.)
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

# Import Versa client from the ARIA project (canonical install on this machine)
_VERSA_UTILS = "/data/rauschecker2/jkw/aria/dev/src/utils"
if _VERSA_UTILS not in sys.path:
    sys.path.append(_VERSA_UTILS)

# Lazy-import: only attempt when actually calling the API
def _get_versa(model: str, log_dir: str | Path):
    from versa_api import VersaAI
    return VersaAI(deployment=model, usage_log_dir=str(log_dir), run_dir=str(log_dir))


OPUS_46 = "us.anthropic.claude-opus-4-6-v1"


# ----- prompt builders ------------------------------------------------------

# Columns we hand to the LLM (mirror what a chart-reviewer would see)
_PROMPT_FEATURES = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital", "medical_specialty",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses", "diag_1", "diag_2", "diag_3",
    "max_glu_serum", "A1Cresult", "insulin", "metformin", "change", "diabetesMed",
]

# admission_type_id / discharge_disposition_id / admission_source_id decoders
# (use SDG+14-style coarse labels; '?' kept verbatim)
ADM_TYPE = {1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
            5: "Not Available", 6: "NULL", 7: "Trauma Center", 8: "Not Mapped"}
ADM_SRC_COARSE = {
    1: "Physician Referral", 2: "Clinic Referral", 3: "HMO Referral",
    4: "Transfer from a hospital", 5: "Transfer from SNF", 6: "Transfer from another HCF",
    7: "Emergency Room", 8: "Court/Law Enforcement", 9: "Not Available",
    10: "Transfer from critical access hospital", 11: "Normal Delivery",
    17: "NULL", 20: "Not Mapped", 22: "Transfer hospital inpt", 25: "Transfer ASC",
}


def _coerce(v):
    if v is None:
        return "unknown"
    if isinstance(v, float) and pd.isna(v):
        return "unknown"
    return str(v).strip()


_INSTRUCTION = (
    "You are a clinical informatics assistant. Estimate the probability that the "
    "described diabetic patient will be re-admitted to the hospital within 30 days of discharge. "
    'Respond ONLY with a JSON object of the form {"p": <float 0-1>}. Do not include any other text.'
)


def build_prompt_structured(row: dict) -> str:
    """Format 1: structured JSON dict of feature-value pairs."""
    body = {k: _coerce(row.get(k)) for k in _PROMPT_FEATURES}
    return f"{_INSTRUCTION}\n\nPATIENT_RECORD:\n{json.dumps(body, indent=2)}"


def build_prompt_natural(row: dict) -> str:
    """Format 2: narrative natural-language description."""
    g = _coerce(row.get("gender")).lower()
    age = _coerce(row.get("age"))
    race = _coerce(row.get("race"))
    los = _coerce(row.get("time_in_hospital"))
    spec = _coerce(row.get("medical_specialty"))
    n_lab = _coerce(row.get("num_lab_procedures"))
    n_med = _coerce(row.get("num_medications"))
    n_dx = _coerce(row.get("number_diagnoses"))
    n_out = _coerce(row.get("number_outpatient"))
    n_er = _coerce(row.get("number_emergency"))
    n_in = _coerce(row.get("number_inpatient"))
    a1c = _coerce(row.get("A1Cresult"))
    glu = _coerce(row.get("max_glu_serum"))
    ins = _coerce(row.get("insulin"))
    met = _coerce(row.get("metformin"))
    change = _coerce(row.get("change"))
    diab_med = _coerce(row.get("diabetesMed"))
    dx1 = _coerce(row.get("diag_1"))
    dx2 = _coerce(row.get("diag_2"))
    dx3 = _coerce(row.get("diag_3"))
    adm_type_raw = _coerce(row.get("admission_type_id"))
    try:
        adm_label = ADM_TYPE.get(int(adm_type_raw), adm_type_raw)
    except ValueError:
        adm_label = adm_type_raw

    narrative = (
        f"A {race} {g} aged {age} years was admitted under the {adm_label} pathway "
        f"to the {spec} service for {los} day(s). "
        f"Diabetes lab work: A1c result was {a1c}; max serum glucose was {glu}. "
        f"Medications include insulin ({ins}) and metformin ({met}); the regimen was {change} "
        f"during the stay (patient is {diab_med} on diabetes medications overall). "
        f"Primary diagnosis ICD-9 code was {dx1}, with secondary {dx2} and tertiary {dx3}. "
        f"During the encounter the patient had {n_lab} lab procedures, {n_med} medications, and "
        f"{n_dx} total diagnoses on the problem list. "
        f"In the year prior they had {n_out} outpatient visit(s), {n_er} ER visit(s), and "
        f"{n_in} inpatient stay(s)."
    )
    return f"{_INSTRUCTION}\n\nCLINICAL VIGNETTE:\n{narrative}"


def build_prompt_abbreviated(row: dict) -> str:
    """Format 3: clinical shorthand (the 'modification' the PS asks for)."""
    g = _coerce(row.get("gender"))[:1].upper()  # M / F
    age = _coerce(row.get("age")).replace("[", "").replace(")", "")
    los = _coerce(row.get("time_in_hospital"))
    a1c = _coerce(row.get("A1Cresult"))
    glu = _coerce(row.get("max_glu_serum"))
    n_med = _coerce(row.get("num_medications"))
    n_dx = _coerce(row.get("number_diagnoses"))
    n_in = _coerce(row.get("number_inpatient"))
    n_er = _coerce(row.get("number_emergency"))
    n_out = _coerce(row.get("number_outpatient"))
    dx1 = _coerce(row.get("diag_1"))
    dx2 = _coerce(row.get("diag_2"))
    dx3 = _coerce(row.get("diag_3"))
    ins = _coerce(row.get("insulin"))
    met = _coerce(row.get("metformin"))
    change = _coerce(row.get("change"))
    body = (
        f"Pt: {g}, age {age}\n"
        f"LOS: {los}d\n"
        f"A1c: {a1c} | MaxGlu: {glu}\n"
        f"Dx1: {dx1} | Dx2: {dx2} | Dx3: {dx3}\n"
        f"Meds: ins {ins}, met {met}; Rx change: {change}\n"
        f"Util: {n_med} meds, {n_dx} dx, IP {n_in} | ER {n_er} | OP {n_out} (prior yr)\n"
    )
    return f"{_INSTRUCTION}\n\nCHART_SNAPSHOT:\n{body}"


PROMPT_BUILDERS: dict[str, Callable[[dict], str]] = {
    "structured": build_prompt_structured,
    "natural": build_prompt_natural,
    "abbreviated": build_prompt_abbreviated,
}


# ----- parsing --------------------------------------------------------------

_FLOAT_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def parse_prob(text: str) -> float | None:
    """Pull a [0,1] probability out of the LLM response."""
    if not isinstance(text, str):
        return None
    # try JSON first
    txt = text.strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and "p" in obj:
            p = float(obj["p"])
            return min(max(p, 0.0), 1.0)
    except Exception:
        pass
    # fallback: regex-grep any float in [0,1]
    for m in _FLOAT_RE.findall(txt):
        try:
            p = float(m)
            if 0.0 <= p <= 1.0:
                return p
        except ValueError:
            continue
    return None


# ----- main eval loop -------------------------------------------------------

@dataclass
class CacheEntry:
    prompt_format: str
    row_id: int
    response: str
    parsed_p: float | None


def run_versa_eval(
    df: pd.DataFrame,
    row_ids: list[int],
    prompt_format: str,
    cache_path: Path,
    model: str = OPUS_46,
    log_dir: str | Path = "/tmp/cph_versa_log",
    sleep_s: float = 0.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run Versa on the given rows of df under one prompt format, caching to disk.

    Returns a dataframe with columns: row_id, prompt_format, response, parsed_p.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
    else:
        cached = pd.DataFrame(columns=["row_id", "prompt_format", "response", "parsed_p"])

    done = set(zip(cached["row_id"], cached["prompt_format"])) if len(cached) else set()
    builder = PROMPT_BUILDERS[prompt_format]
    versa = _get_versa(model, log_dir)

    new_rows = []
    for i, rid in enumerate(row_ids):
        if (rid, prompt_format) in done:
            continue
        row = df.loc[rid].to_dict()
        prompt = builder(row)
        try:
            response = versa.predict(prompt, verbose=verbose)
        except Exception as e:
            response = f"Error: {e}"
        parsed = parse_prob(response)
        new_rows.append({"row_id": rid, "prompt_format": prompt_format,
                         "response": response, "parsed_p": parsed})
        # flush every 25 calls so a crash doesn't lose work
        if len(new_rows) % 25 == 0:
            pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True).to_csv(cache_path, index=False)
        if sleep_s:
            time.sleep(sleep_s)

    if new_rows:
        cached = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
        cached.to_csv(cache_path, index=False)
    # return only the relevant subset
    mask = (cached["prompt_format"] == prompt_format) & (cached["row_id"].isin(row_ids))
    return cached[mask].reset_index(drop=True)
