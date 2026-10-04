"""Is the shift only in P(empty), or in the whole count distribution?

Estimates the test count prior from the model's mean posterior, corrected by the
per-class calibration ratio measured out-of-fold, then re-tunes the decode under
importance weights matching that prior.
"""
import json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune2
from tune2 import coord_ascent, GRID, CTX
from decode import DEF, decode_all
from common import score_row, score_parts
from fuse import peak_counts, crossfit_peak_post

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
W = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK"
tr = pd.read_csv(D + r"\train.csv")
truth = [json.loads(s) for s in tr.answer_json]
cnt = np.array([t["region_count"] for t in truth])
O = {k: v for k, v in np.load(W + r"\oof_v1.npz").items()}
fold = O["fold"]; idx = np.where(fold >= 0)[0]
TP = np.load(W + r"\test_pred_v1.npz")["cnt"]

prior_tr = np.bincount(np.clip(cnt, 0, 12), minlength=13).astype(float); prior_tr /= prior_tr.sum()
mean_oof = O["cnt"][idx].mean(0)
mean_te = TP.mean(0)
ratio = np.where(mean_oof > 1e-6, prior_tr / np.maximum(mean_oof, 1e-9), 0.0)
pi_te = mean_te * ratio
pi_te = np.clip(pi_te, 0, None); pi_te /= pi_te.sum()

np.set_printoptions(precision=4, suppress=True)
print("train prior      :", prior_tr)
print("mean OOF post    :", mean_oof, " -> calibration looks",
      "good" if abs(mean_oof[0] - prior_tr[0]) < 0.02 else "off")
print("mean TEST post   :", mean_te)
print("calibrated pi_te :", pi_te)
print(f"\nP(empty): train {prior_tr[0]:.4f}  ->  test {pi_te[0]:.4f}")
print(f"mean count: train {(prior_tr*np.arange(13)).sum():.3f}  ->  test {(pi_te*np.arange(13)).sum():.3f}")
nz_tr = (prior_tr[1:] * np.arange(1, 13)).sum() / prior_tr[1:].sum()
nz_te = (pi_te[1:] * np.arange(1, 13)).sum() / pi_te[1:].sum()
print(f"mean count | non-empty: train {nz_tr:.3f}  ->  test {nz_te:.3f}")

# importance weights matching the full calibrated prior
w_full = np.where(prior_tr[np.clip(cnt, 0, 12)] > 0,
                  pi_te[np.clip(cnt, 0, 12)] / np.maximum(prior_tr[np.clip(cnt, 0, 12)], 1e-9), 0.0)
w_full = w_full[idx]; w_full = w_full / w_full.mean()
print(f"\nweight stats: min {w_full.min():.3f} max {w_full.max():.3f} "
      f"ESS {w_full.sum()**2/ (w_full**2).sum():.0f}/{len(w_full)}")

CTX.update(O=O, truth=truth, p_cnn=O["cnt"], p_knn=np.load(W + r"\knn_oof.npy"))
CTX["p_peak"] = crossfit_peak_post(peak_counts(O["peak"], 0.25), cnt, fold)
GRID["w_peak"] = [0.0]; GRID["w_knn"] = [0.0, 0.15, 0.3]

base = dict(DEF); base.update(empty_prior=0.145, knn_alpha=0.0, w_knn=0.0, w_peak=0.0)
prm_emptyonly = json.load(open(W + r"\prm_final.json"))

print("\n=== re-tune under FULL-distribution weights ===")
prm2, best2 = coord_ascent(idx, w_full, base, verbose=True)
print("  params:", json.dumps(prm2))

def sc(prm, w):
    c, zv, av = tune2.counts_and_dicts(prm)
    pr = decode_all(O, idx, prm, None, c, zv, av)
    s = np.array([score_row(p, truth[i]) for p, i in zip(pr, idx)])
    return float((s * w).sum() / w.sum())

print("\n=== do the two parameter sets actually differ in score? ===")
print(f"  empty-only params, full-dist weights : {sc(prm_emptyonly, w_full):.5f}")
print(f"  full-dist  params, full-dist weights : {sc(prm2, w_full):.5f}")
print(f"  -> cost of keeping the simpler params: {sc(prm2, w_full)-sc(prm_emptyonly, w_full):+.5f}")
diff = {k: (prm_emptyonly[k], prm2[k]) for k in prm2 if prm_emptyonly.get(k) != prm2[k]}
print("  differing params:", diff if diff else "NONE")
