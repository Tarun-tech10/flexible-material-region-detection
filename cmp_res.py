"""Paired comparison of two model configs on the SAME held-out fold.

Reports the tuned row score plus the finer-grained detection metrics that the
resolution change is meant to move (IoU, centre-cell hit rate, card score), which
are far less noisy on 600 rows than the headline number.
"""
import json, sys, os, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune2
from tune2 import coord_ascent, GRID, CTX
from decode import DEF, decode_all
from common import score_row, score_parts, _iou
from post import row_weights
from fuse import peak_counts, crossfit_peak_post

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
W = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK"
tr = pd.read_csv(D + r"\train.csv")
truth = [json.loads(s) for s in tr.answer_json]
cnt = np.array([t["region_count"] for t in truth])
PRM = json.load(open(W + r"\prm_final3.json"))


def card_metrics(pred, idx):
    ious, cell, zone, abin, rank, npair = [], [], [], [], [], 0
    npred = ntrue = 0
    for pr, i in zip(pred, idx):
        tc = truth[i]["region_cards"]; pc = pr["region_cards"]
        npred += len(pc); ntrue += len(tc)
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
            used.add(bi); t = tc[bi]; npair += 1
            ious.append(_iou(c["bbox"], t["bbox"]))
            cell.append(c["center_cell"] == t["center_cell"])
            zone.append(c["zone"] == t["zone"]); abin.append(c["area_bin"] == t["area_bin"])
            rank.append(c["region_rank"] == t["region_rank"])
    return dict(pairs=npair, IoU=float(np.mean(ious)), cell=float(np.mean(cell)),
                zone=float(np.mean(zone)), abin=float(np.mean(abin)), rank=float(np.mean(rank)),
                card_ratio=npred / max(ntrue, 1))


def evaluate_tag(tag, fold_id):
    O = {k: v for k, v in np.load(f"{W}\\oof_{tag}.npz").items()}
    fold = O["fold"]
    idx = np.where(fold == fold_id)[0]
    if len(idx) == 0:
        return None, None, None
    CTX.clear()
    CTX.update(O=O, truth=truth, p_cnn=O["cnt"], p_knn=np.load(W + r"\knn_oof.npy"))
    CTX["p_peak"] = crossfit_peak_post(peak_counts(O["peak"], 0.25), cnt, fold)
    tune2._CNTC.clear()
    w = row_weights(cnt[idx], 0.145)
    c, zv, av = tune2.counts_and_dicts(PRM)
    pred = decode_all(O, idx, PRM, None, c, zv, av)
    s = np.array([score_row(p, truth[i]) for p, i in zip(pred, idx)])
    return float((s * w).sum() / w.sum()), score_parts(pred, [truth[i] for i in idx]), card_metrics(pred, idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="v1")
    ap.add_argument("--b", default="hi384")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    print(f"paired comparison on fold {args.fold}, identical decode params\n")
    rows = {}
    for tag in (args.a, args.b):
        sc, parts, cm = evaluate_tag(tag, args.fold)
        if sc is None:
            print(f"{tag}: fold {args.fold} not present"); continue
        rows[tag] = (sc, parts, cm)
        print(f"{tag:8s} test-matched {sc:.5f}")
        print(f"         parts  {{{', '.join(f'{k}:{v:.4f}' for k,v in parts.items())}}}")
        print(f"         cards  IoU {cm['IoU']:.4f}  cell {cm['cell']:.4f}  zone {cm['zone']:.4f}  "
              f"abin {cm['abin']:.4f}  rank {cm['rank']:.4f}  pairs {cm['pairs']}  "
              f"pred/true {cm['card_ratio']:.3f}\n")
    if len(rows) == 2:
        a, b = rows[args.a], rows[args.b]
        print(f"DELTA ({args.b} - {args.a}):")
        print(f"  score  {b[0]-a[0]:+.5f}")
        for k in a[1]:
            print(f"  {k:8s} {b[1][k]-a[1][k]:+.4f}")
        for k in ("IoU", "cell", "zone", "abin", "rank"):
            print(f"  card.{k:5s} {b[2][k]-a[2][k]:+.4f}")


if __name__ == "__main__":
    main()
