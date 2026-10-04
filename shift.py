"""Does the TEST set have a different empty-rate / count distribution than TRAIN?
Estimate it via the (comparable) best-match-similarity structure."""
import json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
W = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK"
tr = pd.read_csv(D + r"\train.csv"); te = pd.read_csv(D + r"\test.csv")
A = [json.loads(s) for s in tr.answer_json]
cnt = np.array([a["region_count"] for a in A])
ntr, nte = len(tr), len(te)
X = np.load(W + r"\desc24.npy")
S = X @ X.T
np.fill_diagonal(S, -1)

sim_tr = S[:ntr, :ntr].max(1)          # train row -> best OTHER train row
sim_te = S[ntr:, :ntr].max(1)          # test  row -> best train row
nn_te = S[ntr:, :ntr].argmax(1)

BANDS = [(0.99, 1.01), (0.95, 0.99), (0.90, 0.95), (0.87, 0.90), (0.84, 0.87),
         (0.80, 0.84), (0.75, 0.80), (0.0, 0.75)]
print("band            n_train  P(empty|train)  mean_count   n_test")
rows = []
for lo, hi in BANDS:
    mt = (sim_tr >= lo) & (sim_tr < hi)
    ms = (sim_te >= lo) & (sim_te < hi)
    pe = (cnt[mt] == 0).mean() if mt.sum() else np.nan
    mc = cnt[mt].mean() if mt.sum() else np.nan
    rows.append((lo, hi, mt.sum(), pe, mc, ms.sum()))
    print(f"[{lo:.2f},{hi:.2f})   {mt.sum():6d}   {pe:12.3f}   {mc:9.3f}   {ms.sum():6d}")

n_t = np.array([r[5] for r in rows], float)
pe = np.array([r[3] for r in rows]); mc = np.array([r[4] for r in rows])
ok = ~np.isnan(pe)
w = n_t[ok] / n_t[ok].sum()
print(f"\nTRAIN  actual: P(empty)={ (cnt==0).mean():.3f}  mean_count={cnt.mean():.3f}")
print(f"TEST  implied: P(empty)={(w*pe[ok]).sum():.3f}  mean_count={(w*mc[ok]).sum():.3f}")

print("\n=== labels of the near-duplicate matches for TEST rows (sim>=0.99) ===")
m = sim_te >= 0.99
print(f"  n={m.sum()}; their matched-train counts:",
      np.bincount(cnt[nn_te[m]], minlength=6).tolist(), " P(empty)=", (cnt[nn_te[m]] == 0).mean().round(3))

print("\n=== pixel-level check of the 'exact duplicate' claim ===")
im = np.load(W + r"\imgs_512.npy", mmap_mode="r")
idx = np.where(sim_te >= 0.999)[0][:6]
for i in idx:
    a = np.asarray(im[ntr + i], np.float32); b = np.asarray(im[nn_te[i]], np.float32)
    print(f"  test[{i}] vs train[{nn_te[i]}] sim={sim_te[i]:.5f}  MAE={np.abs(a-b).mean():6.2f}  "
          f"train_count={cnt[nn_te[i]]}  same_bytes={np.array_equal(a,b)}")
idx2 = np.where((sim_te >= 0.85) & (sim_te < 0.9))[0][:4]
for i in idx2:
    a = np.asarray(im[ntr + i], np.float32); b = np.asarray(im[nn_te[i]], np.float32)
    print(f"  test[{i}] vs train[{nn_te[i]}] sim={sim_te[i]:.5f}  MAE={np.abs(a-b).mean():6.2f}  "
          f"train_count={cnt[nn_te[i]]}")
