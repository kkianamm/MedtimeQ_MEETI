"""
Inspect a downloaded MEETI dataset so you can adapt datasets/meeti.py to the
exact field names in YOUR copy of the .mat files.

Usage:
    python inspect_meeti.py data/meeti

It will:
  * locate record_list.csv (or walk the tree for *.mat),
  * open the first few .mat files,
  * print every key, its shape/dtype, and a short preview,
  * print the `report` text so you can sanity-check the label keywords.

Look at the output and, if needed, edit the SIGNAL_KEYS / REPORT_KEYS /
FEATURE_KEYS lists at the top of datasets/meeti.py to match.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def preview(v):
    if isinstance(v, np.ndarray):
        flat = v.flatten()
        head = flat[:6]
        return f"ndarray shape={v.shape} dtype={v.dtype} head={head}"
    return f"{type(v).__name__}: {str(v)[:120]}"


def main(root):
    root = Path(root)
    assert root.exists(), f"{root} does not exist"

    rl = root / "record_list.csv"
    if rl.exists():
        print(f"Found record_list.csv at {rl}")
        import pandas as pd
        df = pd.read_csv(rl)
        print("record_list columns:", list(df.columns))
        print(df.head(3).to_string())
        print("-" * 70)

    mats = sorted(root.rglob("*.mat"))
    print(f"Found {len(mats)} .mat files under {root}")
    if not mats:
        return

    for p in mats[:3]:
        print("=" * 70)
        print("FILE:", p)
        d = loadmat(str(p))
        for k, v in d.items():
            if k.startswith("__"):
                continue
            print(f"  key={k!r:24} {preview(v)}")
        # try to print report / interpretation text
        for tk in ("report", "Report", "LLM_Interpretation", "interpretation"):
            if tk in d:
                s = d[tk]
                s = "".join(map(str, np.asarray(s).flatten()))
                print(f"  --- {tk} text ---\n  {s[:400]}")
    print("=" * 70)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/meeti")
