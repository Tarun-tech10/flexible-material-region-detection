"""Fuse three partly-independent estimates of the region count.

  1. the network's count head              (global classification)
  2. the detection head's peak count       (how many centres survive NMS)
  3. the retrieval posterior               (labels of visually similar train frames)

Count drives cards / cells / zone / area as well as its own two terms, so a better
count posterior lifts most of the row score.  The peak-count table is CROSS-FITTED
over the grouped folds so it never sees the row it scores.
"""
import numpy as np
from decode import nms

NMAX = 12


def peak_counts(peaks, thr, nms_thr=0.45, cap=16):
    """peaks: (n, topk, 5) -> (n,) number of surviving peaks above thr."""
    out = np.zeros(len(peaks), int)
    for i, pk in enumerate(peaks):
        sc, bx = pk[:, 0], pk[:, 1:5]
        m = sc > thr
        if m.sum() == 0:
            out[i] = 0; continue
        out[i] = min(cap, len(nms(bx[m], sc[m], nms_thr)))
    return out


def build_table(k, y, cap=16, smooth=1.0):
    """Empirical P(true count | peak count), Laplace-smoothed."""
    T = np.full((cap + 1, NMAX + 1), smooth, np.float64)
    for ki, yi in zip(k, y):
        T[min(int(ki), cap), min(int(yi), NMAX)] += 1.0
    return T / T.sum(1, keepdims=True)


def crossfit_peak_post(k, y, fold, cap=16, smooth=1.0):
    """Posterior from the peak count, using a table fitted on the OTHER folds."""
    out = np.zeros((len(k), NMAX + 1), np.float32)
    for f in sorted(set(fold.tolist())):
        if f < 0: continue
        te = fold == f
        trn = (~te) & (fold >= 0)
        T = build_table(k[trn], y[trn], cap, smooth)
        out[te] = T[np.clip(k[te], 0, cap)]
    return out


def geo_blend(parts, weights):
    """Weighted geometric mean of count posteriors."""
    acc = np.zeros_like(parts[0], dtype=np.float64)
    tot = 0.0
    for p, w in zip(parts, weights):
        if w <= 0: continue
        acc += w * np.log(np.clip(p, 1e-9, None))
        tot += w
    if tot == 0:
        return parts[0]
    q = np.exp(acc / tot)
    return (q / q.sum(-1, keepdims=True)).astype(np.float32)
