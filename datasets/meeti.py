"""
MEETI classification dataset for MedTsLLM (decoder-only ECG classification).

MEETI (Zhang et al., Sci Data 2026; Zenodo 10.5281/zenodo.15893351) is a
multimodal MIMIC-IV-ECG dataset. In the Zenodo release each study folder holds:

    <study>.mat  -> keys are exactly:  'id', 'report', 'LLM_Interpretation'
                    (ALL TEXT -- there is NO waveform and NO numeric feature
                     array inside the .mat).
    <study>.png  -> plotted 12-lead image (not used by this time-series model).

The folder layout  pNNNN / pXXXXXXXX / sZZZZZZZZ / ZZZZZZZZ  is identical to
MIMIC-IV-ECG, and <study> == the MIMIC-IV-ECG study id. So the raw 12-lead
waveform is obtained from MIMIC-IV-ECG (PhysioNet, credentialed) and MEETI's
.mat is used only for the label + the text description.

------------------------------------------------------------------------------
WHAT THIS LOADER PRODUCES, per record, for the classification pipeline:
    x_enc : [T, F]   time series
    label : int      one of the 5 PTB-XL superclasses (NORM/MI/STTC/CD/HYP),
                     derived from MEETI's free-text `report` by keyword rules
    desc  : str      MEETI's report / interpretation text (used by clip=true)

TWO SIGNAL SOURCES (set in the config, see below):

  signal_source = "mimic"          (DEFAULT, real ECG)
      Loads the waveform with wfdb.rdsamp from `mimic_ecg_root`, matching the
      same relative path as the MEETI .mat. Requires MIMIC-IV-ECG downloaded
      from PhysioNet and the `wfdb` package.

  signal_source = "text_features"  (SMOKE TEST ONLY -- NOT a real ECG)
      Builds a numeric vector by regex-extracting measurements (HR, PR, QRS,
      QT/QTc) from the LLM interpretation text and repeating it across T steps.
      This lets you verify the full train/eval loop executes end-to-end before
      you have PhysioNet access. It does NOT represent the waveform; do not
      report accuracy from this mode as an ECG result.

------------------------------------------------------------------------------
CONFIG  ([datasets.MEETI] block):
    root           = "data/MEETI"              # where you unzipped MEETI.zip
    mimic_ecg_root = "data/mimic-iv-ecg"       # PhysioNet MIMIC-IV-ECG root
    signal_source  = "mimic"                   # or "text_features"
"""

import re
from pathlib import Path

import numpy as np

from .ptbxl import (
    ClassificationDataset,
    SUPERCLASS_ORDER,
    SUPERCLASS_TO_IDX,
)

# Confirmed .mat field names (from the Zenodo release):
REPORT_KEYS = ["report", "Report"]
INTERP_KEYS = ["LLM_Interpretation", "llm_interpretation", "interpretation"]
ID_KEYS = ["id", "ID", "study_id"]


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


# --- text_features fallback: pull numbers out of the interpretation text ---
_NUM = r"(\d+(?:\.\d+)?)"
_FEATURE_PATTERNS = {
    "hr":  re.compile(r"heart rate[^0-9]{0,15}" + _NUM, re.I),
    "pr":  re.compile(r"pr interval[^0-9]{0,15}" + _NUM, re.I),
    "qrs": re.compile(r"qrs[^0-9]{0,15}" + _NUM, re.I),
    "qt":  re.compile(r"\bqt\b[^c][^0-9]{0,15}" + _NUM, re.I),
    "qtc": re.compile(r"qtc[^0-9]{0,15}" + _NUM, re.I),
}


def _text_feature_vector(text):
    vec = []
    for _, pat in _FEATURE_PATTERNS.items():
        m = pat.search(text or "")
        vec.append(float(m.group(1)) if m else 0.0)
    return np.asarray(vec, dtype=np.float32)   # shape [5]


class MEETIClassificationDataset(ClassificationDataset):

    description = (
        "MEETI is a large multimodal 12-lead ECG dataset derived from "
        "MIMIC-IV-ECG, pairing 10-second recordings with clinical reports and "
        "LLM-generated interpretations. Recordings are grouped into five "
        "diagnostic superclasses: Normal ECG (NORM), Myocardial Infarction "
        "(MI), ST/T Changes (STTC), Conduction Disturbance (CD), and "
        "Hypertrophy (HYP)."
    )
    task_description = (
        "Classify the following 12-lead ECG recording into one of five "
        "diagnostic categories: Normal, Myocardial Infarction, ST/T Change, "
        "Conduction Disturbance, or Hypertrophy."
    )

    sampling_rate = 500  # MIMIC-IV-ECG waveforms are 500 Hz

    @property
    def n_classes(self):
        return len(SUPERCLASS_ORDER)

    # ---- config accessors with sensible defaults ----
    def _cfg(self, key, default):
        return self.dataset_config.get(key, default)

    def _meeti_root(self):
        return (Path(__file__).parent / ".." / self._cfg("root", "data/MEETI")).resolve()

    def _mimic_root(self):
        return (Path(__file__).parent / ".." / self._cfg("mimic_ecg_root", "data/mimic-iv-ecg")).resolve()

    def _signal_source(self):
        return self._cfg("signal_source", "mimic")

    # ---- deterministic, patient-disjoint 80/10/10 split by subject_id ----
    def _split_of(self, subject_id):
        h = abs(hash(("meeti-split", str(subject_id)))) % 10
        if h < 8:
            return "train"
        elif h == 8:
            return "val"
        return "test"

    def _list_records(self):
        """Return [(subject_id, study_id, mat_path, rel_path_no_suffix), ...]."""
        base = self._meeti_root()
        records = []
        for p in sorted(base.rglob("*.mat")):
            subj = None
            for part in p.parts:
                if re.fullmatch(r"p\d{6,}", part):
                    subj = part
            rel = p.relative_to(base).with_suffix("")   # e.g. p1000/p10000032/s49036311/49036311
            records.append((subj or p.stem, p.stem, p, rel))
        return records

    def _load_waveform(self, rel, crop):
        """Load [crop, 12] waveform from MIMIC-IV-ECG matching this rel path."""
        import wfdb
        mroot = self._mimic_root()
        # MEETI drops the leading 'files/' that MIMIC-IV-ECG uses; try both.
        candidates = [mroot / rel, mroot / "files" / rel]
        rec = next((c for c in candidates if c.with_suffix(".hea").exists()), None)
        if rec is None:
            return None
        sig, _ = wfdb.rdsamp(str(rec))          # [T, 12]
        sig = np.nan_to_num(np.asarray(sig, dtype=np.float32))
        t = sig.shape[0]
        if t >= crop:
            s = (t - crop) // 2
            sig = sig[s:s + crop]
        else:
            sig = np.concatenate([sig, np.zeros((crop - t, sig.shape[1]), np.float32)], 0)
        return sig

    def get_data(self, split=None):
        from scipy.io import loadmat

        split = split or self.split
        crop = int(self.history_len)
        source = self._signal_source()

        signals, labels, descriptions = [], [], []
        n_seen = n_missing_wave = 0

        for subj, study, mat_path, rel in self._list_records():
            if self._split_of(subj) != split:
                continue
            n_seen += 1

            try:
                d = loadmat(str(mat_path))
            except Exception:
                continue

            report = _mat_str(_first_key(d, REPORT_KEYS))
            interp = _mat_str(_first_key(d, INTERP_KEYS))

            if source == "mimic":
                sig = self._load_waveform(rel, crop)
                if sig is None:
                    n_missing_wave += 1
                    continue
            elif source == "text_features":
                vec = _text_feature_vector(interp or report)   # [5]
                sig = np.tile(vec[None, :], (crop, 1))          # [crop, 5]
            else:
                raise ValueError(f"Unknown signal_source={source!r}")

            signals.append(sig)
            labels.append(report_to_label(report or interp))
            text = report or interp or "No report available."
            descriptions.append(f"Clinical report: {text[:600]}")

        if not signals:
            hint = (
                f"\n  signal_source = {source!r}"
                f"\n  MEETI root    = {self._meeti_root()}  (records seen: {n_seen})"
            )
            if source == "mimic":
                hint += (
                    f"\n  MIMIC-IV-ECG  = {self._mimic_root()}  "
                    f"(waveforms not found for {n_missing_wave} records)"
                    f"\n  -> Download MIMIC-IV-ECG from PhysioNet into that folder, "
                    f"or set signal_source=\"text_features\" to smoke-test the pipeline."
                )
            raise RuntimeError("No MEETI records produced for split="
                               f"{split!r}." + hint)

        data = np.stack(signals, axis=0).astype(np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        return {"data": data, "labels": labels, "descriptions": descriptions}


meeti_datasets = {
    "classification": MEETIClassificationDataset,
}
