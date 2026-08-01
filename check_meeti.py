"""
Verify the MEETI dataset loads correctly BEFORE launching a full training run.

This builds the dataset through the repo's own config path (datasets.get_dataset)
but does NOT load the LLM, so it runs in seconds on CPU.

Usage:
    python check_meeti.py configs/datasets/meeti_decoder.toml
    python check_meeti.py configs/datasets/meeti_decoder.toml --all-splits

What a healthy result looks like:
    * each split has > 0 records,
    * x_enc shape is [history_len, F]  (F == 12 for real MIMIC waveforms),
    * labels span multiple classes (not everything collapsed to one),
    * for signal_source="mimic": signals are NOT all-zero / constant.
"""

import sys
from collections import Counter

import numpy as np
import toml

from utils import dict_to_object
from datasets import get_dataset
from datasets.meeti import SUPERCLASS_ORDER


def check_split(config, split):
    ds = get_dataset(config, split)
    n = len(ds)
    print(f"\n[{split}] records = {n}")
    if n == 0:
        print("  !! EMPTY split -- check your split hashing / data root.")
        return

    sample = ds[0]
    x = sample["x_enc"].numpy() if hasattr(sample["x_enc"], "numpy") else np.asarray(sample["x_enc"])
    print(f"  x_enc shape = {tuple(x.shape)}  (expected [{config.history_len}, F])")
    print(f"  n_features  = {ds.n_features}   n_classes = {ds.n_classes}")

    # label distribution
    labs = ds.labels.numpy() if hasattr(ds.labels, "numpy") else np.asarray(ds.labels)
    dist = Counter(int(v) for v in labs)
    pretty = {SUPERCLASS_ORDER[k]: v for k, v in sorted(dist.items())}
    print(f"  label distribution = {pretty}")
    if len(dist) < 2:
        print("  !! Only one class present -- model can't learn to discriminate.")

    # signal sanity
    finite = np.isfinite(x).all()
    constant = np.allclose(x, x.flat[0])
    print(f"  finite = {finite}   constant = {constant}")
    if constant and config.datasets.MEETI.get("signal_source") == "mimic":
        print("  !! Constant signal in MIMIC mode -- waveform likely not loaded.")

    # description sanity
    if "descriptions" in sample:
        print(f"  sample description: {str(sample['descriptions'])[:120]}...")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cfg_path = args[0] if args else "configs/datasets/meeti_decoder.toml"
    all_splits = "--all-splits" in sys.argv

    config = dict_to_object(toml.load(cfg_path))
    print(f"config: {cfg_path}")
    print(f"dataset={config.data.dataset}  task={config.task}  "
          f"signal_source={config.datasets.MEETI.get('signal_source')}")

    splits = ["train", "val", "test"] if all_splits else ["train"]
    for s in splits:
        try:
            check_split(config, s)
        except Exception as e:
            print(f"\n[{s}] FAILED: {type(e).__name__}: {e}")

    print("\nDone. If shapes, class spread and finiteness look right, "
          "you're ready to run train.py.")


if __name__ == "__main__":
    main()
