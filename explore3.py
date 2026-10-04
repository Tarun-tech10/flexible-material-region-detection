import json, sys, collections
import pandas as pd, numpy as np
sys.path.insert(0, r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK")
from common import *

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
tr = pd.read_csv(D + r"\train.csv")
ss = pd.read_csv(D + r"\sample_submission.csv")
A = [json.loads(s) for s in tr.answer_json]

# ---------- 1. ORACLE: rebuild answers from the true bboxes only
print("=== ORACLE (rebuild every field from true card bboxes) ===")
sub = [a for a in A if a["region_count"] <= 8]      # cards = ALL regions here
rec = [build_answer([c["bbox"] for c in a["region_cards"]]) for a in sub]
print("  rows:", len(sub), " score:", round(score_all(rec, sub), 5))
p = score_parts(rec, sub)
print("  parts:", {k: round(v, 4) for k, v in p.items()})
nid = sum(json.dumps(r, sort_keys=True) == json.dumps(a, sort_keys=True) for r, a in zip(rec, sub))
print(f"  byte-identical reconstructions: {nid}/{len(sub)}")

# ---------- 2. baselines
print("\n=== BASELINES (on all 1800 train rows) ===")
empty = build_answer([])
print("  always-empty          :", round(score_all([empty] * len(A), A), 5))
smp = json.loads(ss.answer_json.iloc[0])
print("  sample_submission-style:", round(score_all([smp] * len(A), A), 5))
print("  parts(always-empty)   :", {k: round(v, 4) for k, v in score_parts([empty] * len(A), A).items()})

# best single constant answer built from a box template: search a few
print("\n  --- constant non-empty templates ---")
best = (0, None)
for n in range(0, 5):
    # place n boxes at the modal locations of rank-i boxes
    boxes = []
    for i in range(n):
        bb = np.array([c["bbox"] for a in A for c in a["region_cards"] if c["region_rank"] == i + 1])
        boxes.append(np.median(bb, 0).tolist())
    ans = build_answer(boxes)
    s = score_all([ans] * len(A), A)
    print(f"    n={n}: {s:.5f}")
    if s > best[0]: best = (s, n)
print("  best constant:", best)

# ---------- 3. ORACLE COUNT ONLY (know n, use median boxes for that n)
print("\n=== ORACLE-COUNT, median-boxes ===")
med = {}
for n in range(0, 13):
    rows = [a for a in A if a["region_count"] == n and a["region_count"] <= 8]
    if not rows or n == 0:
        med[n] = build_answer([]); continue
    boxes = []
    for i in range(n):
        bb = np.array([c["bbox"] for a in rows for c in a["region_cards"] if c["region_rank"] == i + 1])
        boxes.append(np.median(bb, 0).tolist())
    med[n] = build_answer(boxes)
pred = [med[min(a["region_count"], 8)] for a in A]
print("  score:", round(score_all(pred, A), 5))
print("  parts:", {k: round(v, 4) for k, v in score_parts(pred, A).items()})

# ---------- 4. how much is the empty/non-empty gate worth?
print("\n=== gate value ===")
ne = [a for a in A if a["region_count"] == 0]
print(f"  empty rows: {len(ne)}/{len(A)} = {len(ne)/len(A):.3f} -> each scores 1.000 if predicted empty")
nz = [a for a in A if a["region_count"] > 0]
print("  cost of predicting empty on a non-empty row:",
      round(np.mean([score_row(empty, a) for a in nz]), 4))

# ---------- 5. near-duplicate images (container/camera groups)?
print("\n=== container-group check (16x16 gray thumbprints) ===")
from PIL import Image
te = pd.read_csv(D + r"\test.csv")
allp = list(tr.image_path) + list(te.image_path)
th = np.zeros((len(allp), 12 * 12), np.float32)
for i, p in enumerate(allp):
    im = Image.open(f"{D}\\" + p.replace("/", "\\")).convert("L").resize((12, 12), Image.BILINEAR)
    v = np.asarray(im, np.float32).ravel()
    th[i] = (v - v.mean()) / (v.std() + 1e-6)
th /= np.linalg.norm(th, axis=1, keepdims=True) + 1e-9
ntr = len(tr)
S = th @ th.T
np.fill_diagonal(S, -1)
print("  mean best-match sim, train->train :", round(float(S[:ntr, :ntr].max(1).mean()), 4))
print("  mean best-match sim, test ->train :", round(float(S[ntr:, :ntr].max(1).mean()), 4))
print("  mean best-match sim, test ->test  :", round(float(S[ntr:, ntr:].max(1).mean()), 4))
for t in [0.90, 0.95, 0.98]:
    print(f"  test rows with a train match > {t}: {int((S[ntr:, :ntr].max(1) > t).sum())}/{len(te)}")
