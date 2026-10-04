import json, sys, collections
import pandas as pd, numpy as np
sys.path.insert(0, r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK")
from common import *

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
W = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK"
tr = pd.read_csv(D + r"\train.csv"); te = pd.read_csv(D + r"\test.csv")
A = [json.loads(s) for s in tr.answer_json]
ntr, nte = len(tr), len(te)
cnt = np.array([a["region_count"] for a in A])

im = np.load(W + r"\imgs_320.npy", mmap_mode="r")
# --- descriptor: 16x16 grayscale, contrast-normalised  (scene fingerprint)
def fp(sz=16):
    X = np.zeros((ntr + nte, sz * sz), np.float32)
    for i in range(ntr + nte):
        a = im[i].astype(np.float32).mean(2)
        h, w = a.shape
        a = a[: (h // sz) * sz, : (w // sz) * sz].reshape(sz, h // sz, sz, w // sz).mean((1, 3))
        v = a.ravel(); v -= v.mean(); v /= v.std() + 1e-6
        X[i] = v
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X
X = fp()
S = X @ X.T
np.fill_diagonal(S, -1)

print("=== scene clusters (union-find at sim>0.93, TRAIN only) ===")
par = list(range(ntr))
def find(a):
    while par[a] != a: par[a] = par[par[a]]; a = par[a]
    return a
ii, jj = np.where(S[:ntr, :ntr] > 0.93)
for a, b in zip(ii, jj):
    ra, rb = find(a), find(b)
    if ra != rb: par[ra] = rb
grp = np.array([find(i) for i in range(ntr)])
u, c = np.unique(grp, return_counts=True)
print(f"  {len(u)} clusters over {ntr} train rows; sizes: max {c.max()}, mean {c.mean():.1f}, singletons {(c==1).sum()}")

# within-cluster label agreement
same, diff = [], []
for g in u[c > 1]:
    idx = np.where(grp == g)[0]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            same.append(abs(cnt[idx[a]] - cnt[idx[b]]))
rng = np.random.default_rng(0)
for _ in range(20000):
    a, b = rng.integers(0, ntr, 2)
    diff.append(abs(cnt[a] - cnt[b]))
print(f"  |Δcount| within cluster : {np.mean(same):.3f}  (n={len(same)})")
print(f"  |Δcount| random pair    : {np.mean(diff):.3f}")

print("\n=== 1-NN count transfer, RANDOM vs GROUP holdout ===")
from sklearn.model_selection import KFold, GroupKFold
for name, splitter, g in [("random 5-fold", KFold(5, shuffle=True, random_state=0), None),
                          ("group  5-fold", GroupKFold(5), grp)]:
    errs, errs_med = [], []
    for tri, vai in splitter.split(np.arange(ntr), groups=g):
        sub = S[np.ix_(vai, tri)]
        nn = sub.argmax(1)
        pred = cnt[tri][nn]
        errs.append(np.abs(pred - cnt[vai]))
        errs_med.append(np.abs(np.median(cnt[tri]) - cnt[vai]))
    print(f"  {name}: 1-NN MAE {np.concatenate(errs).mean():.3f}   const-median MAE {np.concatenate(errs_med).mean():.3f}")

print("\n=== test-vs-train similarity profile (does CV match test?) ===")
tt = S[ntr:, :ntr].max(1)
trtr = S[:ntr, :ntr].max(1)
print(f"  test  best-train sim: mean {tt.mean():.4f}  >0.93: {(tt>0.93).mean():.3f}")
print(f"  train best-train sim: mean {trtr.mean():.4f}  >0.93: {(trtr>0.93).mean():.3f}")
# group-holdout analogue
gh = []
for tri, vai in GroupKFold(5).split(np.arange(ntr), groups=grp):
    gh.append(S[np.ix_(vai, tri)].max(1))
gh = np.concatenate(gh)
print(f"  group-holdout best sim: mean {gh.mean():.4f}  >0.93: {(gh>0.93).mean():.3f}")

print("\n=== count distribution: train vs (unknown) test ===")
print("  train:", np.bincount(cnt, minlength=11).tolist())
print("  train mean/median:", cnt.mean().round(3), np.median(cnt))

print("\n=== simple global-brightness / empty separability sanity ===")
feat = np.stack([im[i].astype(np.float32).mean() for i in range(ntr)])
from sklearn.metrics import roc_auc_score
print("  AUC(mean brightness -> nonempty):", round(roc_auc_score(cnt > 0, feat), 4))
np.save(W + r"\grp.npy", grp); np.save(W + r"\fp.npy", X)
print("saved grp.npy, fp.npy")
