import json, collections, math
import pandas as pd, numpy as np

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
tr = pd.read_csv(D + r"\train.csv")
te = pd.read_csv(D + r"\test.csv")
print("train", tr.shape, "test", te.shape)
print("ids unique:", tr.id.nunique(), te.id.nunique(), "overlap:", len(set(tr.id) & set(te.id)))
print("img overlap:", len(set(tr.image_path) & set(te.image_path)))
print("w/h train:", tr.width.unique(), tr.height.unique(), "test:", te.width.unique(), te.height.unique())

A = [json.loads(s) for s in tr.answer_json]

# ---- field-level distributions
print("\n=== region_count ===")
rc = pd.Series([a["region_count"] for a in A])
print(rc.value_counts().sort_index().to_dict())

print("\n=== count -> bin mapping ===")
m = collections.defaultdict(set)
for a in A:
    m[a["region_count"]].add(a["region_count_bin"])
for k in sorted(m):
    print(f"  count={k:2d} -> {sorted(m[k])}")

print("\n=== n_cards vs region_count ===")
cc = collections.Counter((a["region_count"], len(a["region_cards"])) for a in A)
bad = [(k, v) for k, v in cc.items() if k[1] != min(k[0], 8)]
print("  rows where n_cards != min(count,8):", sum(v for _, v in bad), bad[:10])

print("\n=== consistency: sum(zone_counts) == region_count? ===")
z_ok = sum(sum(a["zone_counts"].values()) == a["region_count"] for a in A)
ab_ok = sum(sum(a["area_bin_counts"].values()) == a["region_count"] for a in A)
print(f"  zone sum ok {z_ok}/{len(A)}   area_bin sum ok {ab_ok}/{len(A)}")

# ---- do the CARDS (top-8) explain zone_counts / area_bin_counts when count<=8 ?
print("\n=== cards reproduce zone/area counts (count<=8 rows) ===")
zbad = abad = n = 0
for a in A:
    if a["region_count"] > 8 or a["region_count"] == 0:
        continue
    n += 1
    zc = collections.Counter(c["zone"] for c in a["region_cards"])
    ac = collections.Counter(c["area_bin"] for c in a["region_cards"])
    if dict(zc) != {k: v for k, v in a["zone_counts"].items() if v}:
        zbad += 1
    if dict(ac) != {k: v for k, v in a["area_bin_counts"].items() if v}:
        abad += 1
print(f"  n={n}  zone mismatches={zbad}  area mismatches={abad}")

# ---- rank ordering / area monotonicity
print("\n=== card ranks & area ordering ===")
rank_ok = area_desc = ncards = 0
for a in A:
    cds = a["region_cards"]
    if not cds: continue
    ncards += 1
    if [c["region_rank"] for c in cds] == list(range(1, len(cds) + 1)):
        rank_ok += 1
    ar = [(c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1]) for c in cds]
    if all(ar[i] >= ar[i + 1] - 1e-9 for i in range(len(ar) - 1)):
        area_desc += 1
print(f"  rows with cards={ncards}  ranks==1..n: {rank_ok}  bbox-area descending: {area_desc}")

# ---- zone rule from bbox center y
print("\n=== zone vs center-y ===")
pts = []
for a in A:
    for c in a["region_cards"]:
        x0, y0, x1, y1 = c["bbox"]
        pts.append((c["zone"], (y0 + y1) / 2, y0, y1))
dfz = pd.DataFrame(pts, columns=["zone", "cy", "y0", "y1"])
print(dfz.groupby("zone")["cy"].agg(["min", "max", "count"]))

# ---- area_bin thresholds vs bbox area
print("\n=== area_bin vs bbox area (normalized) ===")
pts = []
for a in A:
    for c in a["region_cards"]:
        x0, y0, x1, y1 = c["bbox"]
        pts.append((c["area_bin"], (x1 - x0) * (y1 - y0), (x1 - x0), (y1 - y0)))
dfa = pd.DataFrame(pts, columns=["bin", "area", "w", "h"])
print(dfa.groupby("bin")["area"].agg(["min", "max", "count"]).sort_values("min"))

# ---- center_cell rule
print("\n=== center_cell rule check ===")
def cell_of(cx, cy, R=10, C=12):
    r = min(R - 1, max(0, int(cy * R)))
    c = min(C - 1, max(0, int(cx * C)))
    return f"r{r:02d}_c{c:02d}"
ok = tot = 0
mism = []
for a in A:
    for c in a["region_cards"]:
        x0, y0, x1, y1 = c["bbox"]
        tot += 1
        p = cell_of((x0 + x1) / 2, (y0 + y1) / 2)
        if p == c["center_cell"]: ok += 1
        elif len(mism) < 8: mism.append((c["bbox"], c["center_cell"], p))
print(f"  floor-rule matches {ok}/{tot}")
for m_ in mism: print("   ", m_)

# ---- bbox value granularity
print("\n=== bbox granularity ===")
vals = np.array([v for a in A for c in a["region_cards"] for v in c["bbox"]])
print("  n:", len(vals), "min", vals.min(), "max", vals.max())
print("  decimals: unique count of round(v,6)==round(v,2):", np.mean(np.isclose(vals, np.round(vals, 2))))

# ---- occupied cells stats
print("\n=== occupied_cells ===")
nc = pd.Series([len(a["occupied_cells"]) for a in A])
print("  per-row count:", nc.describe().to_dict())
print("  sorted&unique always:", all(a["occupied_cells"] == sorted(set(a["occupied_cells"])) for a in A))
