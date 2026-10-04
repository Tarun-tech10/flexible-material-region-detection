import json, sys, os, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import score_row, score_all, score_parts
from decode import decode_all, DEF
from post import row_weights, P_EMPTY_TEST

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
W = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK"

GRID = {
    "cell_empty": [0.25, 0.35, 0.45, 0.55, 0.62, 0.70, 0.80, 0.90, 0.97],
    "card_empty": [0.25, 0.35, 0.45, 0.55, 0.62, 0.70, 0.80, 0.90, 0.97],
    "cell_thr":   [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    "pk_thr":     [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30],
    "nms_thr":    [0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
    "card_extra": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    "box_scale":  [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10],
    "use_grid_cells": [0, 1],
    "knn_alpha":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
}


def wsc(preds, trues, w):
    s = np.array([score_row(p, t) for p, t in zip(preds, trues)])
    return float((s * w).sum() / w.sum())


KNN = None   # set in main()

def evaluate(O, idx, truth, prm, w):
    return wsc(decode_all(O, idx, prm, KNN), [truth[i] for i in idx], w)


def coord_ascent(O, idx, truth, w, prm0, rounds=4, verbose=True):
    prm = dict(prm0)
    best = evaluate(O, idx, truth, prm, w)
    if verbose: print(f"  start {best:.5f}")
    for r in range(rounds):
        improved = False
        for k, vals in GRID.items():
            cur = prm[k]; bv, bs = cur, best
            for v in vals:
                if v == cur: continue
                p2 = dict(prm); p2[k] = v
                s = evaluate(O, idx, truth, p2, w)
                if s > bs + 1e-6: bs, bv = s, v
            if bv != cur:
                prm[k] = bv; best = bs; improved = True
                if verbose: print(f"   r{r} {k:16s} {cur} -> {bv}   {best:.5f}")
        if not improved: break
    return prm, best


def main():
    global KNN
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--pe", type=float, default=P_EMPTY_TEST)
    a = ap.parse_args()
    if os.path.exists(f"{W}\\knn_oof.npy"):
        KNN = np.load(f"{W}\\knn_oof.npy")
        print("loaded knn_oof.npy")
    tr = pd.read_csv(D + r"\train.csv")
    truth = [json.loads(s) for s in tr.answer_json]
    cnt = np.array([t["region_count"] for t in truth])
    O = np.load(f"{W}\\oof_{a.tag}.npz")
    fold = O["fold"]
    idx = np.where(fold >= 0)[0]
    print(f"OOF rows: {len(idx)}   folds present: {sorted(set(fold[idx].tolist()))}")

    wt = row_weights(cnt[idx], a.pe)                 # test-matched weights
    w1 = np.ones(len(idx))                           # raw train distribution
    base = dict(DEF); base["empty_prior"] = a.pe

    print("\n=== default decode ===")
    pr = decode_all(O, idx, base)
    T = [truth[i] for i in idx]
    print(f"  raw-train-dist  : {score_all(pr, T):.5f}")
    print(f"  TEST-matched    : {wsc(pr, T, wt):.5f}")

    print("\n=== coordinate ascent under the TEST-matched distribution ===")
    prm, best = coord_ascent(O, idx, truth, wt, base)
    pr = decode_all(O, idx, prm)
    print(f"  TUNED test-matched: {best:.5f}")
    print("  params:", prm)
    print("  parts (test-weighted):")
    sp = {}
    for nm in ["cell_f1", "count", "bin", "zone", "area", "card", "TOTAL"]:
        sp[nm] = None
    pp = score_parts(pr, T)
    print("   unweighted parts:", {k: round(v, 4) for k, v in pp.items()})

    print("\n=== what tuning on the RAW train distribution would have chosen (the trap) ===")
    base_raw = dict(DEF); base_raw["empty_prior"] = 0.39556
    prm_raw, best_raw = coord_ascent(O, idx, truth, w1, base_raw, verbose=False)
    print(f"  raw-tuned params : {prm_raw}")
    print(f"  its score on raw-train dist  : {best_raw:.5f}")
    print(f"  its score on TEST-matched    : {evaluate(O, idx, truth, prm_raw, wt):.5f}")
    print(f"  test-tuned score on TEST     : {best:.5f}   -> gain {best - evaluate(O, idx, truth, prm_raw, wt):+.5f}")

    print("\n=== decode-overfit check (tune on folds A, eval on folds B) ===")
    fu = sorted(set(fold[idx].tolist()))
    if len(fu) >= 2:
        h = max(1, len(fu) // 2)
        ia = np.where(np.isin(fold, fu[:h]))[0]
        ib = np.where(np.isin(fold, fu[h:]))[0]
        wa, wb = row_weights(cnt[ia], a.pe), row_weights(cnt[ib], a.pe)
        pa, _ = coord_ascent(O, ia, truth, wa, base, verbose=False)
        print(f"  eval-half: default {evaluate(O, ib, truth, base, wb):.5f} | "
              f"tuned-on-A {evaluate(O, ib, truth, pa, wb):.5f} | "
              f"tuned-on-ALL {evaluate(O, ib, truth, prm, wb):.5f}")

    print("\n=== sensitivity of the tuned score to the assumed test empty-rate ===")
    for pe in [0.10, 0.145, 0.20, 0.30, 0.396]:
        p2 = dict(prm); p2["empty_prior"] = pe
        w2 = row_weights(cnt[idx], pe)
        print(f"  P(empty)={pe:.3f}: {evaluate(O, idx, truth, p2, w2):.5f}")

    json.dump(prm, open(f"{W}\\prm_{a.tag}.json", "w"), indent=1)
    print("\nsaved", f"prm_{a.tag}.json")


if __name__ == "__main__":
    main()
