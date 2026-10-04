"""Similarity retrieval over the labelled training images.

Two regimes:
  A) a near-duplicate match (sim >= ~0.99) -> the neighbour's answer transfers almost exactly
     (measured inside train: copy-NN scores 0.9996 in that band)
  B) otherwise -> a soft count posterior from the k nearest labelled images, blended with the CNN.
Validated with GROUP folds (own scene cluster removed), which is the honest regime.
"""
import numpy as np

NMAX = 12


def build_desc(imgs, sz=24):
    n = len(imgs)
    X = np.zeros((n, sz * sz * 3), np.float32)
    for i in range(n):
        a = np.asarray(imgs[i], np.float32)
        h, w, _ = a.shape
        a = a[: (h // sz) * sz, : (w // sz) * sz].reshape(sz, h // sz, sz, w // sz, 3).mean((1, 3))
        v = a.reshape(-1)
        v -= v.mean(); v /= v.std() + 1e-6
        X[i] = v
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X


def knn_count_post(Sq, cnt_ref, k=12, tau=0.02, prior=None, floor=1e-3):
    """Sq: (nq, nref) similarities. -> (nq, 13) count posterior from neighbours."""
    nq = Sq.shape[0]
    k = min(k, Sq.shape[1])
    part = np.argpartition(-Sq, k - 1, axis=1)[:, :k]
    out = np.zeros((nq, NMAX + 1), np.float32)
    for i in range(nq):
        idx = part[i]
        s = Sq[i, idx]
        o = np.argsort(-s); idx, s = idx[o], s[o]
        wgt = np.exp((s - s[0]) / tau)
        for j, c in enumerate(cnt_ref[idx]):
            out[i, min(int(c), NMAX)] += wgt[j]
    out += floor
    out /= out.sum(1, keepdims=True)
    if prior is not None:
        out = out * prior
        out /= out.sum(1, keepdims=True)
    return out


def blend(p_cnn, p_knn, alpha):
    """Geometric blend of two count posteriors."""
    if alpha <= 0: return p_cnn
    q = np.power(np.clip(p_cnn, 1e-9, None), 1 - alpha) * np.power(np.clip(p_knn, 1e-9, None), alpha)
    return q / q.sum(-1, keepdims=True)
