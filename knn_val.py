"""Honest (group-fold) validation of the kNN count posterior."""
import json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from knn import knn_count_post, blend
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
W = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK"
tr = pd.read_csv(D + r"\train.csv")
cnt = np.array([json.loads(s)["region_count"] for s in tr.answer_json])
ntr = len(tr)
X = np.load(W + r"\desc24.npy")
grp = np.load(W + r"\grp.npy")
S = X[:ntr] @ X[:ntr].T
np.fill_diagonal(S, -1)

print("=== kNN count posterior, GROUP 5-fold (own scene cluster never visible) ===")
oof = np.zeros((ntr, 13), np.float32)
for tri, vai in GroupKFold(5).split(np.arange(ntr), groups=grp):
    oof[vai] = knn_count_post(S[np.ix_(vai, tri)], cnt[tri], k=12, tau=0.02)
np.save(W + r"\knn_oof.npy", oof)
pe = oof[:, 0]
print(f"  AUC(P_knn(empty)) : {roc_auc_score(cnt == 0, pe):.4f}")
print(f"  MAE(argmax count) : {np.abs(oof.argmax(1) - cnt).mean():.4f}")
med = np.array([np.searchsorted(np.cumsum(oof[i]), 0.5) for i in range(ntr)])
print(f"  MAE(median count) : {np.abs(med - cnt).mean():.4f}")
print(f"  const-median MAE  : {np.abs(np.median(cnt) - cnt).mean():.4f}")

print("\n  calibration of P_knn(empty):")
for lo, hi in [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
    m = (pe >= lo) & (pe < hi)
    if m.sum(): print(f"    [{lo:.1f},{hi:.1f}) n={m.sum():4d}  actual empty-rate {(cnt[m]==0).mean():.3f}")

print("\n=== k / tau sweep (group folds) ===")
best = None
for k in (5, 8, 12, 20, 30):
    for tau in (0.005, 0.01, 0.02, 0.04, 0.08):
        o = np.zeros((ntr, 13), np.float32)
        for tri, vai in GroupKFold(5).split(np.arange(ntr), groups=grp):
            o[vai] = knn_count_post(S[np.ix_(vai, tri)], cnt[tri], k=k, tau=tau)
        a = roc_auc_score(cnt == 0, o[:, 0])
        nll = -np.log(np.clip(o[np.arange(ntr), np.clip(cnt, 0, 12)], 1e-9, None)).mean()
        if best is None or a > best[0]: best = (a, k, tau, nll)
        print(f"  k={k:3d} tau={tau:.3f}: AUC(empty) {a:.4f}  NLL {nll:.4f}")
print("  best by AUC:", best)
