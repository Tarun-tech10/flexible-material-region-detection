"""Does decoupling the zone/area tally's box size from the card boxes help?

The cards want boxes left alone (IoU is squared-error-optimal as predicted); the
area-bin tally wants the L1 shrinkage undone. One shared knob cannot serve both.
Everything except the two new parameters is pinned, so this is a clean LOFO test.
"""
import json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune2
from tune2 import CTX
from decode import decode_all
from common import score_row, score_parts
from post import row_weights
from fuse import peak_counts, crossfit_peak_post

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
W = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK"
tr = pd.read_csv(D + r"\train.csv")
truth = [json.loads(s) for s in tr.answer_json]
cnt = np.array([t["region_count"] for t in truth])
O = {k: v for k, v in np.load(W + r"\oof_v1.npz").items()}
fold = O["fold"]; idx = np.where(fold >= 0)[0]
CTX.update(O=O, truth=truth, p_cnn=O["cnt"], p_knn=np.load(W + r"\knn_oof.npy"))
CTX["p_peak"] = crossfit_peak_post(peak_counts(O["peak"], 0.25), cnt, fold)
BASE = json.load(open(W + r"\prm_final3.json"))

SCALES = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25]
GAMMAS = [1.0, 1.1, 1.2, 1.3]


def sc(prm, ii, w):
    c, zv, av = tune2.counts_and_dicts(prm)
    pr = decode_all(O, ii, prm, None, c, zv, av)
    s = np.array([score_row(p, truth[i]) for p, i in zip(pr, ii)])
    return float((s * w).sum() / w.sum())


def best_on(ii, w):
    best = (-1, 1.0, 1.0)
    for s_ in SCALES:
        for g in GAMMAS:
            p = dict(BASE); p["dict_box_scale"] = s_; p["dict_size_gamma"] = g
            v = sc(p, ii, w)
            if v > best[0]: best = (v, s_, g)
    return best


print("=== leave-one-fold-out ===")
base_tot, new_tot = [], []
for f in (0, 1, 2):
    ia = np.where((fold >= 0) & (fold != f))[0]; ib = np.where(fold == f)[0]
    wa, wb = row_weights(cnt[ia], 0.145), row_weights(cnt[ib], 0.145)
    _, s_, g = best_on(ia, wa)
    p = dict(BASE); p["dict_box_scale"] = s_; p["dict_size_gamma"] = g
    b, n = sc(BASE, ib, wb), sc(p, ib, wb)
    base_tot.append(b); new_tot.append(n)
    print(f"  fold{f}: base {b:.5f} -> new {n:.5f} ({n-b:+.5f})   chosen scale={s_} gamma={g}")
print(f"  MEAN : base {np.mean(base_tot):.5f} -> new {np.mean(new_tot):.5f} "
      f"({np.mean(new_tot)-np.mean(base_tot):+.5f})")

wt = row_weights(cnt[idx], 0.145)
v, s_, g = best_on(idx, wt)
print(f"\ntuned on all: {v:.5f}  scale={s_} gamma={g}   (base {sc(BASE, idx, wt):.5f})")
p = dict(BASE); p["dict_box_scale"] = s_; p["dict_size_gamma"] = g
c, zv, av = tune2.counts_and_dicts(p)
print("parts:", {k: round(x, 4) for k, x in
                 score_parts(decode_all(O, idx, p, None, c, zv, av), [truth[i] for i in idx]).items()})
json.dump(p, open(W + r"\prm_final4.json", "w"), indent=1)
