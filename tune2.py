import json, sys, os, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import score_row, score_all, score_parts, build_answer
from decode import decode_all, DEF
from post import row_weights, P_EMPTY_TEST
from fuse import peak_counts, crossfit_peak_post, geo_blend

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
W = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK"

GRID = {
    "cell_empty": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.97],
    "card_empty": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.97],
    "cell_thr":   [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    "pk_thr":     [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30],
    "nms_thr":    [0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
    "card_extra": [-1.0, -0.5, 0.0, 0.5, 1.0],
    "box_scale":  [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15],
    "use_grid_cells": [0, 1],
    "w_knn":      [0.0, 0.15, 0.3, 0.5, 0.75, 1.0],
    "w_peak":     [0.0, 0.15, 0.3, 0.5, 0.75, 1.0],
}
CTX = {}


from decode import rebase, precompute_dicts

_CNTC = {}
def counts_and_dicts(prm):
    """Fused + rebased count posterior and the expected-score zone/area vectors.
    Cached on the only three parameters they depend on."""
    key = (round(prm["w_knn"], 6), round(prm["w_peak"], 6), round(prm["empty_prior"], 6))
    if key not in _CNTC:
        c = geo_blend([CTX["p_cnn"], CTX["p_knn"], CTX["p_peak"]],
                      [1.0, prm["w_knn"], prm["w_peak"]])
        c = rebase(c, prm["empty_prior"])
        zv, av = precompute_dicts(c, CTX["O"]["zon"], CTX["O"]["abn"])
        _CNTC[key] = (c, zv, av)
    return _CNTC[key]


def evaluate(idx, prm, w):
    c, zv, av = counts_and_dicts(prm)
    pr = decode_all(CTX["O"], idx, prm, None, c, zv, av)
    s = np.array([score_row(p, CTX["truth"][i]) for p, i in zip(pr, idx)])
    return float((s * w).sum() / w.sum())


def coord_ascent(idx, w, prm0, rounds=4, verbose=True):
    prm = dict(prm0)
    best = evaluate(idx, prm, w)
    if verbose: print(f"  start {best:.5f}")
    for r in range(rounds):
        improved = False
        for k, vals in GRID.items():
            cur = prm[k]; bv, bs = cur, best
            for v in vals:
                if v == cur: continue
                p2 = dict(prm); p2[k] = v
                s = evaluate(idx, p2, w)
                if s > bs + 1e-6: bs, bv = s, v
            if bv != cur:
                prm[k] = bv; best = bs; improved = True
                if verbose: print(f"   r{r} {k:16s} {cur} -> {bv}   {best:.5f}")
        if not improved: break
    return prm, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--pe", type=float, default=P_EMPTY_TEST)
    a = ap.parse_args()
    tr = pd.read_csv(D + r"\train.csv")
    truth = [json.loads(s) for s in tr.answer_json]
    cnt = np.array([t["region_count"] for t in truth])
    # materialise: indexing an NpzFile re-decompresses the whole array every time
    O = {k: v for k, v in np.load(f"{W}\\oof_{a.tag}.npz").items()}
    fold = O["fold"]
    idx = np.where(fold >= 0)[0]
    print(f"OOF rows: {len(idx)}  folds: {sorted(set(fold[idx].tolist()))}")

    CTX["O"] = O; CTX["truth"] = truth
    CTX["p_cnn"] = O["cnt"]
    knn = np.load(f"{W}\\knn_oof.npy")
    CTX["p_knn"] = knn
    pk = peak_counts(O["peak"], 0.25)
    CTX["p_peak"] = crossfit_peak_post(pk, cnt, fold)
    print("peak-count vs true count: MAE", np.abs(pk[idx] - cnt[idx]).mean().round(3))

    # ---- how good is each count source alone (L1-optimal median decode)?
    print("\n=== count sources (median decode, OOF rows) ===")
    for nm, p in [("cnn", CTX["p_cnn"]), ("knn", CTX["p_knn"]), ("peak", CTX["p_peak"])]:
        med = np.array([np.searchsorted(np.cumsum(p[i]), 0.5) for i in idx])
        print(f"  {nm:5s} MAE {np.abs(med - cnt[idx]).mean():.4f}   "
              f"empty-acc {(np.equal(med == 0, cnt[idx] == 0)).mean():.4f}")

    wt = row_weights(cnt[idx], a.pe)
    base = dict(DEF); base["empty_prior"] = a.pe; base["knn_alpha"] = 0.0
    base["w_knn"] = 0.0; base["w_peak"] = 0.0

    print("\n=== reference points (test-matched weighting) ===")
    empty = build_answer([])
    se = np.array([score_row(empty, truth[i]) for i in idx])
    print(f"  always-empty     : {float((se*wt).sum()/wt.sum()):.5f}")
    print(f"  default decode   : {evaluate(idx, base, wt):.5f}")

    print("\n=== coordinate ascent (test-matched) ===")
    prm, best = coord_ascent(idx, wt, base)
    print(f"  TUNED: {best:.5f}")
    print("  params:", json.dumps(prm))
    c, zv, av = counts_and_dicts(prm)
    pr = decode_all(O, idx, prm, None, c, zv, av)
    print("  unweighted parts:", {k: round(v, 4) for k, v in score_parts(pr, [truth[i] for i in idx]).items()})

    print("\n=== the trap: tuning on the RAW train distribution instead ===")
    braw = dict(base); braw["empty_prior"] = 0.39556
    praw, sraw = coord_ascent(idx, np.ones(len(idx)), braw, verbose=False)
    print(f"  raw-tuned score on raw dist : {sraw:.5f}")
    print(f"  raw-tuned score on TEST dist: {evaluate(idx, praw, wt):.5f}")
    print(f"  test-tuned score on TEST    : {best:.5f}  -> gain {best-evaluate(idx, praw, wt):+.5f}")

    print("\n=== decode-overfit check (tune on half the folds, eval on the other) ===")
    fu = sorted(set(fold[idx].tolist()))
    if len(fu) >= 2:
        h = max(1, len(fu) // 2)
        ia = np.where(np.isin(fold, fu[:h]))[0]
        ib = np.where(np.isin(fold, fu[h:]))[0]
        wa, wb = row_weights(cnt[ia], a.pe), row_weights(cnt[ib], a.pe)
        pa, _ = coord_ascent(ia, wa, base, verbose=False)
        print(f"  held-out half: default {evaluate(ib, base, wb):.5f} | "
              f"tuned-on-A {evaluate(ib, pa, wb):.5f} | tuned-on-ALL {evaluate(ib, prm, wb):.5f}")
        print(f"  optimism of tuning on all folds: {evaluate(ib, prm, wb)-evaluate(ib, pa, wb):+.5f}")

    print("\n=== sensitivity to the assumed test empty-rate ===")
    for pe in [0.10, 0.145, 0.20, 0.30, 0.396]:
        p2 = dict(prm); p2["empty_prior"] = pe
        print(f"  assume {pe:.3f} -> scored under {pe:.3f}: {evaluate(idx, p2, row_weights(cnt[idx], pe)):.5f}"
              f" | our params under {pe:.3f}: {evaluate(idx, prm, row_weights(cnt[idx], pe)):.5f}")

    json.dump(prm, open(f"{W}\\prm_{a.tag}.json", "w"), indent=1)
    print("\nsaved prm_" + a.tag + ".json")


if __name__ == "__main__":
    main()
