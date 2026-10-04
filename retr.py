"""Is near-duplicate retrieval from train worth using, and above what similarity?"""
import json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import score_row, build_answer

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
W = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK"
tr = pd.read_csv(D + r"\train.csv"); te = pd.read_csv(D + r"\test.csv")
truth = [json.loads(s) for s in tr.answer_json]
ntr, nte = len(tr), len(te)
im = np.load(W + r"\imgs_288.npy", mmap_mode="r")


def desc(sz=24):
    X = np.zeros((ntr + nte, sz * sz * 3), np.float32)
    for i in range(ntr + nte):
        a = im[i].astype(np.float32)
        h, w, _ = a.shape
        a = a[: (h // sz) * sz, : (w // sz) * sz]
        a = a.reshape(sz, h // sz, sz, w // sz, 3).mean((1, 3))
        v = a.reshape(-1)
        v -= v.mean(); v /= v.std() + 1e-6
        X[i] = v
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X

X = desc()
np.save(W + r"\desc24.npy", X)
S = X @ X.T
np.fill_diagonal(S, -1)

print("=== copy-nearest-train-answer, score vs similarity (measured inside TRAIN) ===")
nn = S[:ntr, :ntr].argmax(1)
sim = S[:ntr, :ntr].max(1)
sc = np.array([score_row(truth[nn[i]], truth[i]) for i in range(ntr)])
empty = np.array([build_answer([])] * 1)[0]
sc_empty = np.array([score_row(empty, truth[i]) for i in range(ntr)])
for lo, hi in [(0.999, 1.01), (0.99, 0.999), (0.98, 0.99), (0.97, 0.98), (0.95, 0.97),
               (0.93, 0.95), (0.90, 0.93), (0.85, 0.90), (0.0, 0.85)]:
    m = (sim >= lo) & (sim < hi)
    if m.sum() == 0: continue
    print(f"  sim [{lo:.3f},{hi:.3f}): n={m.sum():4d}  copy-NN {sc[m].mean():.4f}   "
          f"always-empty {sc_empty[m].mean():.4f}")

print("\n=== how many TEST rows sit in each band (best train match) ===")
ts = S[ntr:, :ntr].max(1)
for lo, hi in [(0.999, 1.01), (0.99, 0.999), (0.98, 0.99), (0.97, 0.98), (0.95, 0.97),
               (0.93, 0.95), (0.90, 0.93), (0.85, 0.90), (0.0, 0.85)]:
    m = (ts >= lo) & (ts < hi)
    print(f"  sim [{lo:.3f},{hi:.3f}): {m.sum():4d} / {nte}")

print("\n=== k-NN answer agreement (are top-2 train neighbours consistent?) ===")
ord2 = np.argsort(-S[:ntr, :ntr], 1)[:, :2]
ag = np.array([score_row(truth[ord2[i, 0]], truth[ord2[i, 1]]) for i in range(ntr)])
for lo in [0.99, 0.98, 0.97, 0.95, 0.93]:
    m = sim >= lo
    print(f"  sim>={lo}: n={m.sum():4d}  agreement(top1 vs top2 label) {ag[m].mean():.4f}")
