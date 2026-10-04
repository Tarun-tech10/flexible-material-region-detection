"""Do EMPTY containers look alike ACROSS different containers?
If yes -> the test empty-rate is inferable from the similarity bands.
If no  -> the 87 high-sim test rows are merely leaked duplicate frames."""
import json, sys, os
import numpy as np, pandas as pd

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
W = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK"
tr = pd.read_csv(D + r"\train.csv")
A = [json.loads(s) for s in tr.answer_json]
cnt = np.array([a["region_count"] for a in A])
ntr = len(tr)
X = np.load(W + r"\desc24.npy")
S = X @ X.T
np.fill_diagonal(S, -1)
Str = S[:ntr, :ntr]

# scene clusters on train at a STRICT threshold (same physical scene/time)
par = list(range(ntr))
def find(a):
    while par[a] != a: par[a] = par[par[a]]; a = par[a]
    return a
ii, jj = np.where(Str > 0.97)
for a, b in zip(ii, jj):
    ra, rb = find(a), find(b)
    if ra != rb: par[ra] = rb
cl = np.array([find(i) for i in range(ntr)])
u, c = np.unique(cl, return_counts=True)
print(f"clusters@0.97: {len(u)}  max {c.max()}  empty-rate of biggest: "
      f"{(cnt[cl==u[np.argmax(c)]]==0).mean():.3f}")

# best match EXCLUDING own cluster
Sx = Str.copy()
same = cl[:, None] == cl[None, :]
Sx[same] = -1
best_x = Sx.max(1)
print("\nP(empty) and cross-cluster best-sim:")
for lab, m in [("empty rows   ", cnt == 0), ("non-empty rows", cnt > 0)]:
    print(f"  {lab}: n={m.sum():4d}  mean cross-cluster best-sim {best_x[m].mean():.4f}  "
          f"median {np.median(best_x[m]):.4f}  >0.90: {(best_x[m]>0.90).mean():.3f}  "
          f">0.95: {(best_x[m]>0.95).mean():.3f}")

print("\nband table using CROSS-CLUSTER similarity (train):")
BANDS = [(0.95, 1.01), (0.90, 0.95), (0.87, 0.90), (0.84, 0.87), (0.80, 0.84), (0.0, 0.80)]
for lo, hi in BANDS:
    m = (best_x >= lo) & (best_x < hi)
    if m.sum() == 0: continue
    print(f"  [{lo:.2f},{hi:.2f}) n={m.sum():4d}  P(empty)={(cnt[m]==0).mean():.3f}  mean_count={cnt[m].mean():.3f}")

# ---- the real question: is a test row's high-sim match a *duplicate frame* or a *look-alike*?
print("\n=== TEST rows: sim to best train match, and to best train match OUTSIDE that match's cluster ===")
Ste = S[ntr:, :ntr]
nn = Ste.argmax(1)
sim1 = Ste.max(1)
sim2 = np.array([Ste[i][cl != cl[nn[i]]].max() for i in range(Ste.shape[0])])
hi = sim1 >= 0.99
print(f"  rows with sim1>=0.99: n={hi.sum()}")
print(f"    their sim to a DIFFERENT cluster: mean {sim2[hi].mean():.4f} median {np.median(sim2[hi]):.4f}")
print(f"    P(empty) of that other-cluster match: {(cnt[[np.where(cl!=cl[nn[i]])[0][Ste[i][cl!=cl[nn[i]]].argmax()] for i in np.where(hi)[0]]]==0).mean():.3f}")
lo_ = sim1 < 0.90
print(f"  rows with sim1<0.90: n={lo_.sum()}  their best-match train count dist:",
      np.bincount(cnt[nn[lo_]], minlength=8).tolist())

# ---- strongest evidence: leave-one-CLUSTER-out empty detection by retrieval alone
print("\n=== can retrieval alone detect 'empty' across clusters? (train, own cluster removed) ===")
from sklearn.metrics import roc_auc_score
k = 5
part = np.argpartition(-Sx, k, axis=1)[:, :k]
knn_empty = np.array([(cnt[part[i]] == 0).mean() for i in range(ntr)])
print(f"  AUC(kNN-empty-rate, own cluster excluded) -> is_empty : {roc_auc_score(cnt==0, knn_empty):.4f}")
print(f"  AUC(cross-cluster best-sim)               -> is_empty : {roc_auc_score(cnt==0, best_x):.4f}")
