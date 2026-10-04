import json, collections
import pandas as pd, numpy as np

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
tr = pd.read_csv(D + r"\train.csv")
A = [json.loads(s) for s in tr.answer_json]
R, C = 10, 12

def cells_intersect(bbs, eps=0.0):
    """cells whose [c/C,(c+1)/C]x[r/R,(r+1)/R] overlaps any bbox by > eps"""
    out = set()
    for x0, y0, x1, y1 in bbs:
        c0 = max(0, int(np.floor(x0 * C + eps)));  c1 = min(C - 1, int(np.ceil(x1 * C - eps)) - 1)
        r0 = max(0, int(np.floor(y0 * R + eps)));  r1 = min(R - 1, int(np.ceil(y1 * R - eps)) - 1)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out.add(f"r{r:02d}_c{c:02d}")
    return out

def cells_center(bbs):
    """cells whose CENTER lies inside a bbox"""
    out = set()
    for x0, y0, x1, y1 in bbs:
        for r in range(R):
            cy = (r + 0.5) / R
            if not (y0 <= cy <= y1): continue
            for c in range(C):
                cx = (c + 0.5) / C
                if x0 <= cx <= x1: out.add(f"r{r:02d}_c{c:02d}")
    return out

def cells_frac(bbs, thr):
    """cells whose overlap area with union of bboxes >= thr * cell area"""
    # rasterize union finely
    S = 20  # subsamples per cell dim
    H, W = R * S, C * S
    m = np.zeros((H, W), bool)
    ys = (np.arange(H) + 0.5) / H
    xs = (np.arange(W) + 0.5) / W
    for x0, y0, x1, y1 in bbs:
        m |= ((ys[:, None] >= y0) & (ys[:, None] <= y1) & (xs[None, :] >= x0) & (xs[None, :] <= x1))
    f = m.reshape(R, S, C, S).mean(axis=(1, 3))
    return {f"r{r:02d}_c{c:02d}" for r in range(R) for c in range(C) if f[r, c] >= thr}

# only rows where cards == all regions (count <= 8) so bboxes are complete
rows = [a for a in A if 0 < a["region_count"] <= 8]
print("testable rows (0<count<=8):", len(rows))
print("empty rows have empty cells:", all(not a["occupied_cells"] for a in A if a["region_count"] == 0))

def score(fn, name):
    exact = 0; f1s = []; over = 0; under = 0
    for a in rows:
        bbs = [c["bbox"] for c in a["region_cards"]]
        p = fn(bbs); t = set(a["occupied_cells"])
        if p == t: exact += 1
        if p - t: over += 1
        if t - p: under += 1
        inter = len(p & t)
        f1s.append(0 if not inter else 2 * inter / (len(p) + len(t)))
    print(f"  {name:28s} exact {exact:4d}/{len(rows)} ({exact/len(rows):.3f})  meanF1 {np.mean(f1s):.4f}  over {over} under {under}")

print("\n=== occupied_cells hypotheses ===")
score(lambda b: cells_intersect(b, 0.0), "bbox-intersect(any overlap)")
score(lambda b: cells_intersect(b, 1e-9), "bbox-intersect(eps)")
score(cells_center, "cell-center-inside-bbox")
for t in [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
    score(lambda b, t=t: cells_frac(b, t), f"cell-overlap-frac >= {t}")
