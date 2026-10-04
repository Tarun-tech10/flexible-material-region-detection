"""Where is the card score lost?  Decompose the greedy-matched pair score."""
import json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import _iou
from decode import decode_all, rebase, precompute_dicts
from fuse import peak_counts, crossfit_peak_post, geo_blend
from post import row_weights

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
W = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK"
tr = pd.read_csv(D + r"\train.csv")
truth = [json.loads(s) for s in tr.answer_json]
cnt = np.array([t["region_count"] for t in truth])
O = {k: v for k, v in np.load(W + r"\oof_v1.npz").items()}
fold = O["fold"]; idx = np.where(fold >= 0)[0]
prm = json.load(open(W + r"\prm_v1.json"))

p = geo_blend([O["cnt"], np.load(W + r"\knn_oof.npy"),
               crossfit_peak_post(peak_counts(O["peak"], 0.25), cnt, fold)],
              [1.0, prm["w_knn"], prm["w_peak"]])
p = rebase(p, prm["empty_prior"])
zv, av = precompute_dicts(p, O["zon"], O["abn"])
pred = decode_all(O, idx, prm, None, p, zv, av)

comp = {k: [] for k in ["iou2", "cell", "zone", "abin", "rank"]}
ious, npred, ntrue = [], [], []
for pr, i in zip(pred, idx):
    tc = truth[i]["region_cards"]; pc = pr["region_cards"]
    npred.append(len(pc)); ntrue.append(len(tc))
    used = set()
    for c in pc:
        best, bi = -1, -1
        for j, t in enumerate(tc):
            if j in used: continue
            s = (0.45 * _iou(c["bbox"], t["bbox"]) ** 2 + 0.20 * (c["center_cell"] == t["center_cell"])
                 + 0.15 * (c["zone"] == t["zone"]) + 0.15 * (c["area_bin"] == t["area_bin"])
                 + 0.05 * (c["region_rank"] == t["region_rank"]))
            if s > best: best, bi = s, j
        if bi < 0: continue
        used.add(bi); t = tc[bi]
        io = _iou(c["bbox"], t["bbox"])
        ious.append(io)
        comp["iou2"].append(io ** 2)
        comp["cell"].append(c["center_cell"] == t["center_cell"])
        comp["zone"].append(c["zone"] == t["zone"])
        comp["abin"].append(c["area_bin"] == t["area_bin"])
        comp["rank"].append(c["region_rank"] == t["region_rank"])

print(f"matched pairs: {len(ious)}   mean IoU {np.mean(ious):.4f}   mean IoU^2 {np.mean(comp['iou2']):.4f}")
print("\nper-pair component means (and their contribution to the 1.00 pair score):")
for k, wgt in [("iou2", .45), ("cell", .20), ("zone", .15), ("abin", .15), ("rank", .05)]:
    m = float(np.mean(comp[k]))
    print(f"  {k:5s} mean {m:.4f}  x{wgt:.2f} = {m*wgt:.4f}   (max {wgt:.2f}, losing {wgt*(1-m):.4f})")
tot = sum(np.mean(comp[k]) * w for k, w in [("iou2", .45), ("cell", .20), ("zone", .15), ("abin", .15), ("rank", .05)])
print(f"  -> mean matched pair score {tot:.4f}")
print(f"\ncards predicted {np.sum(npred)}  true {np.sum(ntrue)}  ratio {np.sum(npred)/max(np.sum(ntrue),1):.3f}")

print("\n=== predicted vs true bbox size distribution (matched pairs) ===")
pw, ph, tw, th = [], [], [], []
for pr, i in zip(pred, idx):
    for c, t in zip(pr["region_cards"], truth[i]["region_cards"]):
        pw.append(c["bbox"][2] - c["bbox"][0]); ph.append(c["bbox"][3] - c["bbox"][1])
        tw.append(t["bbox"][2] - t["bbox"][0]); th.append(t["bbox"][3] - t["bbox"][1])
print(f"  width  pred mean {np.mean(pw):.4f} sd {np.std(pw):.4f} | true mean {np.mean(tw):.4f} sd {np.std(tw):.4f}")
print(f"  height pred mean {np.mean(ph):.4f} sd {np.std(ph):.4f} | true mean {np.mean(th):.4f} sd {np.std(th):.4f}")
pa = np.array(pw) * np.array(ph); ta = np.array(tw) * np.array(th)
print(f"  area   pred mean {pa.mean():.5f} sd {pa.std():.5f} | true mean {ta.mean():.5f} sd {ta.std():.5f}")
print(f"  area ratio pred/true: {pa.mean()/ta.mean():.4f}   sd ratio {pa.std()/ta.std():.4f}")

from decode import abin_of, ABINS
print("\n  area-bin histograms (matched pairs):")
print("   pred:", np.bincount([abin_of(a) for a in pa], minlength=4).tolist(), ABINS)
print("   true:", np.bincount([abin_of(a) for a in ta], minlength=4).tolist())
