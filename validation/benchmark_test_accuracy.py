#!/usr/bin/env python
"""Head-to-head accuracy benchmark: OpenSpliceAI (PyTorch) vs original SpliceAI (Keras).

Streams a human MANE ``dataset_test.h5`` through both 5-model ensembles and computes the canonical
SpliceAI metrics on the **full** test set (no sub-sampling): top-kL accuracy (k = 0.5, 1, 2, 4) and
AUPRC for donor and acceptor, plus per-class precision/recall/F1. The two backends share the *same*
windowed inputs and labels, so any difference localises to the model.

The two GPU passes run as **separate processes** (one PyTorch, one TensorFlow) so the frameworks do
not fight over GPU memory; a final ``--report`` step diffs the two JSON metric dumps:

    ENV=/home/kchao10/miniconda3/envs/pytorch_cuda/bin/python
    $ENV benchmark_test_accuracy.py --backend osai     --test-h5 <test.h5> \
         --osai-dir     models/openspliceai-mane/10000nt          --out benchmark_out/osai.json
    $ENV benchmark_test_accuracy.py --backend spliceai --test-h5 <test.h5> \
         --spliceai-dir models/spliceai/SpliceAI_models_release   --out benchmark_out/spliceai.json
    $ENV benchmark_test_accuracy.py --report --osai-json benchmark_out/osai.json \
         --spliceai-json benchmark_out/spliceai.json              --out benchmark_out/table.md

The top-kL / AUPRC computation mirrors ``openspliceai/train_base/utils.py::print_topl_statistics``;
ensembling averages the (softmax) probabilities, matching the original SpliceAI tool and OpenSpliceAI
``predict``. Channel order is (non-splice, acceptor, donor) for both backends.
"""
import argparse
import gc
import glob
import json
import os
import time

import h5py
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

CL_max = 10000  # openspliceai.constants.CL_max — test windows carry 10000 nt of total context
SL = 5000       # openspliceai.constants.SL     — central prediction window length


def topk_auprc(y_true, y_pred):
    """Top-kL accuracy (k = 0.5, 1, 2, 4) and AUPRC for one splice-site class.

    Mirrors ``print_topl_statistics``: rank all positions by predicted score, take the top
    ``k * L`` (L = number of true sites), and report the fraction that are real, plus AUPRC.
    """
    idx_true = np.nonzero(y_true == 1)[0]
    argsorted = np.argsort(y_pred)
    res = {}
    for tl in (0.5, 1, 2, 4):
        n = int(tl * len(idx_true))
        idx_pred = argsorted[-n:] if n > 0 else np.empty(0, dtype=np.intp)
        res[f"top{tl}L"] = float(
            np.size(np.intersect1d(idx_true, idx_pred)) / (min(len(idx_pred), len(idx_true)) + 1e-10)
        )
    del argsorted          # free the ~2 GB index temp before AUPRC argsorts again (8 GiB cgroup cap)
    gc.collect()
    res["auprc"] = float(average_precision_score(y_true, y_pred))
    res["n_true"] = int(len(idx_true))
    return res


class Accumulator:
    """Collect per-class true/pred arrays across all shards, then compute final metrics."""

    def __init__(self):
        self.acc_t, self.acc_p = [], []   # acceptor true (0/1) / pred prob
        self.don_t, self.don_p = [], []   # donor    true (0/1) / pred prob
        self.true_c, self.pred_c = [], []  # argmax class (0/1/2) true / pred

    def add(self, ens, Y):
        """``ens`` and ``Y`` are both (N, SL, 3) — ensembled probs and one-hot labels."""
        self.acc_p.append(ens[..., 1].ravel().astype(np.float32))
        self.don_p.append(ens[..., 2].ravel().astype(np.float32))
        self.pred_c.append(ens.argmax(-1).ravel().astype(np.int8))
        self.acc_t.append(Y[..., 1].ravel().astype(np.int8))
        self.don_t.append(Y[..., 2].ravel().astype(np.int8))
        self.true_c.append(Y.argmax(-1).ravel().astype(np.int8))

    def _drain(self, attr):
        """Concatenate one buffer list and release the per-shard chunks immediately."""
        arr = np.concatenate(getattr(self, attr))
        setattr(self, attr, None)
        gc.collect()
        return arr

    def finalize(self):
        # Concatenate then free each source list immediately, and handle one class at a time,
        # so peak memory stays well under the 8 GiB SLURM cgroup cap (~249M positions/class).
        true_c = self._drain("true_c")
        pred_c = self._drain("pred_c")
        n_positions = int(true_c.size)
        prec, rec, f1, _ = precision_recall_fscore_support(
            true_c, pred_c, labels=[0, 1, 2], average=None, zero_division=0
        )
        del true_c, pred_c
        gc.collect()

        acc_t = self._drain("acc_t")
        acc_p = self._drain("acc_p")
        acceptor = topk_auprc(acc_t, acc_p)
        del acc_t, acc_p
        gc.collect()
        acceptor.update(precision=float(prec[1]), recall=float(rec[1]), f1=float(f1[1]))

        don_t = self._drain("don_t")
        don_p = self._drain("don_p")
        donor = topk_auprc(don_t, don_p)
        del don_t, don_p
        gc.collect()
        donor.update(precision=float(prec[2]), recall=float(rec[2]), f1=float(f1[2]))

        return {"acceptor": acceptor, "donor": donor, "n_positions": n_positions}


def clip_context(X, flanking_size):
    """Trim a (N, length, 4) window from CL_max context down to ``flanking_size`` (no-op at 10000)."""
    clip = (CL_max - flanking_size) // 2
    return X[:, clip:-clip, :] if clip > 0 else X


def _shard_count(f, max_shards):
    n = len(f.keys()) // 2
    return n if not max_shards else min(n, max_shards)


def run_osai(args):
    import torch
    from openspliceai.predict.predict import load_pytorch_models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, _ = load_pytorch_models(args.osai_dir, device, SL, args.flanking_size)
    print(f"[osai] loaded {len(models)} PyTorch models on {device}", flush=True)
    acc = Accumulator()
    with h5py.File(args.test_h5, "r") as f:
        n = _shard_count(f, args.max_shards)
        for i in range(n):
            X = clip_context(f[f"X{i}"][:].astype(np.float32), args.flanking_size)  # (N, Lc, 4)
            Y = f[f"Y{i}"][0]                                                        # (N, SL, 3)
            xt = torch.from_numpy(np.ascontiguousarray(X.transpose(0, 2, 1)))        # (N, 4, Lc)
            ens = np.empty((X.shape[0], SL, 3), dtype=np.float32)
            with torch.no_grad():
                for b in range(0, xt.shape[0], args.batch):
                    xb = xt[b:b + args.batch].to(device)
                    out = None
                    for m in models:
                        o = m(xb)                       # (B, 3, SL), softmax already applied
                        out = o if out is None else out + o
                    ens[b:b + args.batch] = (out / len(models)).permute(0, 2, 1).cpu().numpy()
            acc.add(ens, Y)
            print(f"[osai] shard {i + 1}/{n}  N={X.shape[0]}", flush=True)
    return acc.finalize()


def run_spliceai(args):
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    from tensorflow.keras.models import load_model

    paths = sorted(glob.glob(os.path.join(args.spliceai_dir, "*.h5")))
    models = [load_model(p, compile=False) for p in paths]
    print(f"[spliceai] loaded {len(models)} Keras models: {[os.path.basename(p) for p in paths]}", flush=True)
    acc = Accumulator()
    with h5py.File(args.test_h5, "r") as f:
        n = _shard_count(f, args.max_shards)
        for i in range(n):
            X = clip_context(f[f"X{i}"][:].astype(np.float32), args.flanking_size)  # (N, Lc, 4)
            Y = f[f"Y{i}"][0]                                                        # (N, SL, 3)
            ens = None
            for m in models:
                p = m.predict(X, batch_size=args.batch, verbose=0)                   # (N, SL, 3)
                ens = p if ens is None else ens + p
            acc.add((ens / len(models)).astype(np.float32), Y)
            print(f"[spliceai] shard {i + 1}/{n}  N={X.shape[0]}", flush=True)
    return acc.finalize()


def write_report(args):
    O = json.load(open(args.osai_json))
    S = json.load(open(args.spliceai_json))
    rows = [
        ("top0.5L", "Top-0.5L accuracy"), ("top1L", "Top-1L accuracy"),
        ("top2L", "Top-2L accuracy"), ("top4L", "Top-4L accuracy"),
        ("auprc", "AUPRC"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
    ]
    out = []
    out.append(f"# OpenSpliceAI vs. original SpliceAI — human MANE test set "
               f"({args.flanking_size} nt, 5-model ensembles)\n")
    out.append(f"Test set: `{O.get('test_h5', '')}`  ")
    out.append(f"Positions scored per backend: {O.get('n_positions', '?'):,}  ")
    out.append(f"Wall time: OpenSpliceAI {O.get('seconds', '?')}s, SpliceAI {S.get('seconds', '?')}s\n")
    for ss in ("donor", "acceptor"):
        out.append(f"\n## {ss.capitalize()} (n_true = {O[ss]['n_true']:,})\n")
        out.append("| Metric | OpenSpliceAI | SpliceAI | Δ (OSAI − SAI) |")
        out.append("|---|---|---|---|")
        for key, label in rows:
            o, s = O[ss][key], S[ss][key]
            out.append(f"| {label} | {o:.4f} | {s:.4f} | {o - s:+.4f} |")
    md = "\n".join(out) + "\n"
    with open(args.out, "w") as fh:
        fh.write(md)
    print(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["osai", "spliceai"])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--test-h5")
    ap.add_argument("--osai-dir")
    ap.add_argument("--spliceai-dir")
    ap.add_argument("--flanking-size", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-shards", type=int, default=0, help="0 = all shards; >0 for a smoke test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--osai-json")
    ap.add_argument("--spliceai-json")
    args = ap.parse_args()

    if args.report:
        write_report(args)
        return

    t = time.time()
    res = run_osai(args) if args.backend == "osai" else run_spliceai(args)
    res.update(backend=args.backend, flanking_size=args.flanking_size,
               test_h5=args.test_h5, seconds=round(time.time() - t, 1))
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
