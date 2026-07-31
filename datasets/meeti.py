"""
MEETI classification dataset for MedTsLLM (decoder-only ECG classification).

MEETI (Zhang et al., Sci Data 2026; Zenodo 10.5281/zenodo.15893351) is a
multimodal MIMIC-IV-ECG dataset. Each *study* is a folder that contains:

    <study>.mat  -> keys typically include 'id', 'report',
                    'LLM_Interpretation', per-beat FeatureDB parameters and,
                    depending on the release, the raw 12-lead waveform.
    <study>.png  -> plotted 12-lead image (NOT used by this time-series model).

A `record_list.csv` at the dataset root maps subject_id / file_name -> path.
Record paths follow  files/pNNNN/pXXXXXXXX/sZZZZZZZZ/ZZZZZZZZ.

------------------------------------------------------------------------------
WHY THIS LOADER LOOKS THE WAY IT DOES
------------------------------------------------------------------------------
The MedTsLLM classification pipeline needs, per record:
    (a) a numeric time series  x_enc : [T, F]
    (b) an integer class label
MEETI ships neither in the exact shape PTB-XL did, so:

  * SIGNAL: if a raw-waveform array is present in the .mat we use it; otherwise
    we fall back to the stacked per-beat FeatureDB parameters as the "series".
  * LABEL:  we derive one of the five PTB-XL diagnostic superclasses
    (NORM / MI / STTC / CD / HYP) from the free-text `report` using keyword
    rules. This keeps the existing BiomedCoOp head, prompts, class codes and
    the ClassificationTask completely unchanged.
  * DESCRIPTION: the `report` (or LLM interpretation) is passed through as the
    per-record description, which the `clip = true` prompting option consumes.

>>> RUN  `python inspect_meeti.py data/meeti`  FIRST  <<<
and, if the printed key names differ from the defaults below, edit
SIGNAL_KEYS / REPORT_KEYS / ID_KEYS accordingly.

Data layout expected (unzip MEETI.zip under data/meeti/):
    data/meeti/
        record_list.csv          (optional; if missing we walk the tree)
        files/pNNNN/pXXXXXXXX/sZZZZZZZZ/ZZZZZZZZ.mat
        ...

Requires: scipy (loadmat), pandas, numpy.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .ptbxl import (
    ClassificationDataset,
    SUPERCLASS_ORDER,
    SUPERCLASS_TO_IDX,
)

# ---- adapt these to your .mat field names if inspect_meeti.py shows others ----
SIGNAL_KEYS = ["signal", "signals", "ecg", "ecg_signal", "waveform", "val", "data"]
REPORT_KEYS = ["report", "Report", "reports"]
INTERP_KEYS = ["LLM_Interpretation", "llm_interpretation", "interpretation"]
ID_KEYS = ["id", "ID", "study_id", "file_name"]
# ------------------------------------------------------------------------------


# Keyword rules mapping a free-text report -> one PTB-XL superclass.
# Checked in priority order; first hit wins. NORM is the fallback.
_LABEL_RULES = [
    ("MI",   r"infarct|myocardial infarction|stemi|nstemi|\bmi\b|q wave"),
    ("CD",   r"block|bundle branch|\blbbb\b|\brbbb\b|conduction|fascicular|"
             r"\bav\b.*block|wpw|pre-?excitation|paced|pacing"),
    ("HYP",  r"hypertroph|\blvh\b|\brvh\b|enlargement|dilat"),
    ("STTC", r"st[ -]?segment|st depression|st elevation|t[ -]?wave|"
             r"ischemi|ischaemi|repolariz|repolaris|st-t|st/t"),
    ("NORM", r"normal ecg|normal sinus|within normal limits|unremarkable|\bnormal\b"),
]
_LABEL_RULES = [(c, re.compile(p, re.I)) for c, p in _LABEL_RULES]


def _mat_str(v):
    """Turn a scipy.io.loadmat string cell into a plain python str."""
    if v is None:
        return ""
    arr = np.asarray(v).flatten()
    return " ".join(str(x) for x in arr).strip()


def _first_key(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def report_to_label(report_text):
    """Map a free-text ECG report to a superclass index (int)."""
    t = (report_text or "").lower()
    for code, pat in _LABEL_RULES:
        if pat.search(t):
            return SUPERCLASS_TO_IDX[code]
    return SUPERCLASS_TO_IDX["NORM"]


class MEETIClassificationDataset(ClassificationDataset):

    description = (
        "MEETI is a large multimodal 12-lead ECG dataset derived from "
        "MIMIC-IV-ECG, pairing 10-second recordings with clinical reports, "
        "beat-level parameters and LLM-generated interpretations. Recordings "
        "are grouped into five diagnostic superclasses: Normal ECG (NORM), "
        "Myocardial Infarction (MI), ST/T Changes (STTC), Conduction "
        "Disturbance (CD), and Hypertrophy (HYP)."
    )
    task_description = (
        "Classify the following 12-lead ECG recording into one of five "
        "diagnostic categories: Normal, Myocardial Infarction, ST/T Change, "
        "Conduction Disturbance, or Hypertrophy."
    )

    # MIMIC-IV-ECG raw waveforms are 500 Hz; the feature fallback ignores this.
    sampling_rate = 500

    @property
    def n_classes(self):
        return len(SUPERCLASS_ORDER)

    # ---- deterministic, patient-disjoint 80/10/10 split by subject_id ----
    def _split_of(self, subject_id):
        h = abs(hash(("meeti-split", str(subject_id)))) % 10
        if h < 8:
            return "train"
        elif h == 8:
            return "val"
        return "test"

    def _basepath(self):
        return Path(__file__).parent / "../data/meeti/"

    def _list_records(self):
        """Return a list of (subject_id, mat_path) for every study."""
        base = self._basepath().resolve()
        rl = base / "record_list.csv"
        records = []
        if rl.exists():
            df = pd.read_csv(rl)
            # be liberal about column names
            path_col = next((c for c in df.columns
                             if c.lower() in ("path", "file_path", "record_path")), None)
            subj_col = next((c for c in df.columns
                             if c.lower() in ("subject_id", "subject", "patient_id")), None)
            fn_col = next((c for c in df.columns
                           if c.lower() in ("file_name", "filename", "record")), None)
            for _, row in df.iterrows():
                subj = row[subj_col] if subj_col else None
                if path_col and pd.notna(row[path_col]):
                    p = base / str(row[path_col])
                    if p.suffix != ".mat":
                        p = p.with_suffix(".mat")
                elif fn_col:
                    # reconstruct files/pNNNN/pXXXXXXXX/sZZ/ZZ.mat
                    fn = str(row[fn_col])
                    p = next(base.rglob(f"{fn}.mat"), None)
                else:
                    p = None
                if p and p.exists():
                    records.append((subj, p))
        if not records:
            # fall back to walking the tree
            for p in sorted(base.rglob("*.mat")):
                # subject_id is the pXXXXXXXX directory two levels up
                subj = None
                for part in p.parts:
                    if re.fullmatch(r"p\d{6,}", part):
                        subj = part
                records.append((subj, p))
        return records

    def get_data(self, split=None):
        from scipy.io import loadmat  # local import so repo works without scipy

        split = split or self.split
        crop = int(self.history_len)

        signals, labels, descriptions = [], [], []
        for subj, p in self._list_records():
            # subject-level split (skip records not in this split)
            key_for_split = subj if subj is not None else p.stem
            if self._split_of(key_for_split) != split:
                continue

            try:
                d = loadmat(str(p))
            except Exception:
                continue

            report = _mat_str(_first_key(d, REPORT_KEYS))
            interp = _mat_str(_first_key(d, INTERP_KEYS))

            sig = self._extract_signal(d, crop)
            if sig is None:
                continue  # nothing usable in this file

            signals.append(sig)
            labels.append(report_to_label(report or interp))

            text = report or interp or "No report available."
            descriptions.append(f"Clinical report: {text[:600]}")

        if not signals:
            raise RuntimeError(
                f"No MEETI records found for split={split!r} under "
                f"{self._basepath().resolve()}. Did you unzip MEETI.zip there, "
                f"and do the .mat field names match datasets/meeti.py?"
            )

        data = np.stack(signals, axis=0).astype(np.float32)  # [N, crop, F]
        labels = np.asarray(labels, dtype=np.int64)
        return {"data": data, "labels": labels, "descriptions": descriptions}

    # ------------------------------------------------------------------
    def _extract_signal(self, d, crop):
        """Return a [crop, F] float array from a loaded .mat, or None."""
        raw = _first_key(d, SIGNAL_KEYS)
        arr = None
        if raw is not None:
            arr = np.asarray(raw, dtype=np.float32)
            # orient to [T, F]: MIMIC-IV-ECG waveforms are often [12, 5000]
            if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                arr = arr.T
        else:
            # ---- feature fallback: stack every 2-D numeric field ----
            feats = []
            for k, v in d.items():
                if k.startswith("__"):
                    continue
                v = np.asarray(v)
                if v.ndim == 2 and np.issubdtype(v.dtype, np.number) and min(v.shape) >= 1:
                    m = v.astype(np.float32)
                    if m.shape[0] < m.shape[1]:
                        m = m.T
                    feats.append(m)
            if feats:
                T = max(f.shape[0] for f in feats)
                feats = [np.pad(f, ((0, T - f.shape[0]), (0, 0))) for f in feats]
                arr = np.concatenate(feats, axis=1)

        if arr is None or arr.ndim != 2:
            return None
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # center-crop / pad to `crop` steps
        t = arr.shape[0]
        if t >= crop:
            start = (t - crop) // 2
            arr = arr[start:start + crop]
        else:
            arr = np.concatenate(
                [arr, np.zeros((crop - t, arr.shape[1]), dtype=np.float32)], axis=0
            )
        return arr


meeti_datasets = {
    "classification": MEETIClassificationDataset,
}
