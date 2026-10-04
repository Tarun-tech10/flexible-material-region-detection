"""Expected-score decoding.

The row score is a weighted sum of six INDEPENDENT field scores; the released
sample_submission is itself internally inconsistent (region_count=12 with one
card and four cells) yet is documented as valid, so each field may be decoded to
maximise its own expectation rather than forced to agree with the others.
"""
import itertools, math
import numpy as np

R, C = 10, 12
ZONES = ["upper", "middle", "lower"]
ABINS = ["tiny", "small", "medium", "large"]
AREA_THR = (0.008, 0.026, 0.07)
NMAX = 12


# ------------------------------------------------------------------ helpers
def nms(boxes, scores, thr=0.45):
    idx = np.argsort(-scores)
    keep = []
    while len(idx):
        i = idx[0]; keep.append(i)
        if len(idx) == 1: break
        b, r = boxes[i], boxes[idx[1:]]
        ix0 = np.maximum(b[0], r[:, 0]); iy0 = np.maximum(b[1], r[:, 1])
        ix1 = np.minimum(b[2], r[:, 2]); iy1 = np.minimum(b[3], r[:, 3])
        iw = np.clip(ix1 - ix0, 0, None); ih = np.clip(iy1 - iy0, 0, None)
        inter = iw * ih
        ar_b = (b[2] - b[0]) * (b[3] - b[1])
        ar_r = (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
        iou = inter / np.maximum(ar_b + ar_r - inter, 1e-9)
        idx = idx[1:][iou < thr]
    return np.array(keep, int)


def zone_of(cy):
    return 0 if cy < 1 / 3 else (1 if cy < 2 / 3 else 2)

def abin_of(a):
    return 0 if a < AREA_THR[0] else 1 if a < AREA_THR[1] else 2 if a < AREA_THR[2] else 3

def cells_of_boxes(boxes):
    out = set()
    for x0, y0, x1, y1 in boxes:
        c0 = min(C - 1, max(0, int(math.floor(x0 * C + 1e-9))))
        c1 = min(C - 1, max(0, int(math.ceil(x1 * C - 1e-9)) - 1))
        r0 = min(R - 1, max(0, int(math.floor(y0 * R + 1e-9))))
        r1 = min(R - 1, max(0, int(math.ceil(y1 * R - 1e-9)) - 1))
        for r in range(r0, max(r0, r1) + 1):
            for c in range(c0, max(c0, c1) + 1):
                out.add((r, c))
    return out

def cell_name(r, c):
    return "r%02d_c%02d" % (r, c)

def count_bin_of(n):
    return "none" if n == 0 else "few" if n <= 2 else "several" if n <= 5 else "many"


# ------------------------------------- exact expected-score for dict fields
def _vecs(nbin, smax):
    out = []
    for s in range(smax + 1):
        for c in itertools.combinations_with_replacement(range(nbin), s):
            v = np.zeros(nbin, int)
            for k in c: v[k] += 1
            out.append(v)
    return np.array(out)

_CACHE = {}
def dict_tables(nbin, smax_pred=8, smax_true=NMAX):
    key = (nbin, smax_pred, smax_true)
    if key in _CACHE: return _CACHE[key]
    P = _vecs(nbin, smax_pred)            # candidate predictions
    T = _vecs(nbin, smax_true)            # possible truths
    sp, st = P.sum(1)[:, None], T.sum(1)[None, :]
    err = np.abs(P[:, None, :] - T[None, :, :]).sum(2)
    tot = sp + st
    S = np.where(tot == 0, 1.0, np.maximum(0.0, 1.0 - err / np.maximum(tot, 1)))
    # multinomial coefficient log C(n; T_j) -- depends only on T, so cache it
    lg = np.array([math.lgamma(k + 1) for k in range(NMAX + 2)])
    n = T.sum(1)
    logcoef = lg[n] - lg[T].sum(1)
    _CACHE[key] = (P, T, S.astype(np.float32), n, logcoef)
    return _CACHE[key]


def best_dict(cnt_p, rates, smax_pred=8):
    P, T, S, n, logcoef = dict_tables(len(rates), smax_pred)
    q = rates / max(rates.sum(), 1e-9)
    lp = logcoef + T @ np.log(np.clip(q, 1e-9, None))
    pr = np.exp(lp) * cnt_p[np.clip(n, 0, len(cnt_p) - 1)]
    s = pr.sum()
    pt = pr / s if s > 0 else np.full(len(T), 1.0 / len(T))
    return P[int(np.argmax(S @ pt))]


# ------------------------------------------------------------------ decoder
DEF = dict(
    nms_thr=0.45, pk_thr=0.10,        # peak filtering
    cell_thr=0.35, cell_empty=0.62,   # occupied-cells threshold / empty gate
    card_empty=0.55,                  # cards empty gate
    card_extra=0.0,                   # add this to the card count
    box_scale=1.0,                    # multiplicative box size correction
    use_grid_cells=1,                 # 1: grid head, 0: derive cells from boxes
    empty_prior=0.145,                # assumed P(empty) on the evaluated set
    knn_alpha=0.0,                    # geometric blend weight of the retrieval count posterior
    dict_src=0,                       # 0: zone/area from the global heads, 1: tally the boxes
    cell_hi=1.01,                     # also take cells whose grid prob exceeds this
    size_gamma=1.0,                   # >1 re-inflates box-size spread (L1 regression shrinks it)
    dict_box_scale=1.0,               # size correction applied ONLY to the zone/area tally
    dict_size_gamma=1.0,              # ...so it need not trade against card IoU
)
LOGW0, LOGH0 = -2.160, -1.655         # mean log predicted box width/height, measured on OOF


TRAIN_EMPTY = 0.39556


def rebase(cnt_arr, empty_prior):
    """Re-base a count posterior (or array of them) onto a different empty prior."""
    if empty_prior is None:
        return cnt_arr
    r = np.ones(cnt_arr.shape[-1])
    r[0] = empty_prior / TRAIN_EMPTY
    r[1:] = (1 - empty_prior) / (1 - TRAIN_EMPTY)
    q = cnt_arr * r
    return q / np.maximum(q.sum(-1, keepdims=True), 1e-12)


def precompute_dicts(cnt_arr, zon, abn):
    """Expected-score zone/area vectors. Depends only on the count posterior, so it
    is hoisted out of the per-parameter tuning loop."""
    n = len(cnt_arr)
    zv = np.zeros((n, 3), int); av = np.zeros((n, 4), int)
    for i in range(n):
        zv[i] = best_dict(cnt_arr[i], np.maximum(zon[i], 1e-6))
        av[i] = best_dict(cnt_arr[i], np.maximum(abn[i], 1e-6))
    return zv, av


def decode_row(pk, grid, cnt_p, zon, abn, prm, knn_p=None, zv=None, av=None):
    p_empty = float(cnt_p[0])
    # ---- boxes: NMS on peaks
    sc, bx = pk[:, 0], pk[:, 1:5]
    ok = sc > prm["pk_thr"]
    if ok.sum() == 0: ok = np.zeros(len(sc), bool); ok[np.argmax(sc)] = True
    bx, sc = bx[ok], sc[ok]
    k = nms(bx, sc, prm["nms_thr"])
    bx, sc = bx[k], sc[k]
    g = prm.get("size_gamma", 1.0)
    if prm["box_scale"] != 1.0 or g != 1.0:
        cx, cy = (bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2
        w = np.maximum(bx[:, 2] - bx[:, 0], 1e-6); h = np.maximum(bx[:, 3] - bx[:, 1], 1e-6)
        if g != 1.0:   # re-inflate the spread of log-size about its global mean
            w = np.exp(LOGW0 + g * (np.log(w) - LOGW0))
            h = np.exp(LOGH0 + g * (np.log(h) - LOGH0))
        w = w * prm["box_scale"]; h = h * prm["box_scale"]
        bx = np.clip(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), 0, 1)

    # ---- region_count : L1-optimal = weighted median of the count posterior
    cdf = np.cumsum(cnt_p)
    n_med = int(np.searchsorted(cdf, 0.5))
    # ---- region_count_bin : 0/1 loss -> argmax over binned posterior
    bp = [cnt_p[0], cnt_p[1:3].sum(), cnt_p[3:6].sum(), cnt_p[6:].sum()]
    cbin = ["none", "few", "several", "many"][int(np.argmax(bp))]

    # ---- cards
    ncard = int(np.clip(round(n_med + prm["card_extra"]), 0, 8))
    if p_empty > prm["card_empty"]: ncard = 0
    ncard = min(ncard, len(bx))
    if ncard > 0:
        sel = bx[np.argsort(-sc)[:ncard]]
        ar = (sel[:, 2] - sel[:, 0]) * (sel[:, 3] - sel[:, 1])
        sel = sel[np.argsort(-ar)]
        cards = []
        for i, (x0, y0, x1, y1) in enumerate(sel):
            x0, y0, x1, y1 = [round(float(v), 4) for v in (x0, y0, x1, y1)]
            if x1 <= x0: x1 = min(1.0, x0 + 1e-3)
            if y1 <= y0: y1 = min(1.0, y0 + 1e-3)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            cards.append({
                "area_bin": ABINS[abin_of((x1 - x0) * (y1 - y0))],
                "bbox": [x0, y0, x1, y1],
                "center_cell": cell_name(min(R - 1, int(cy * R)), min(C - 1, int(cx * C))),
                "region_rank": i + 1,
                "zone": ZONES[zone_of(cy)]})
    else:
        cards = []

    # ---- occupied cells
    if p_empty > prm["cell_empty"]:
        cells = []
    elif prm["use_grid_cells"]:
        m = grid > prm["cell_thr"]
        if not m.any(): m = grid >= grid.max()
        cells = sorted(cell_name(r, c) for r, c in zip(*np.where(m)))
    else:
        src = [c["bbox"] for c in cards] if cards else bx[np.argsort(-sc)[:max(1, n_med)]]
        cs = cells_of_boxes(src)
        if prm.get("cell_hi", 1.01) <= 1.0:
            cs |= {(int(r), int(c)) for r, c in zip(*np.where(grid > prm["cell_hi"]))}
        cells = sorted(cell_name(r, c) for r, c in cs)

    # ---- zone / area-bin dictionaries : exact expected-score choice
    if prm.get("dict_src", 0) == 1:
        # tally the top-n_med predicted boxes instead of using the global heads
        top = bx[np.argsort(-sc)[:max(0, n_med)]]
        ds, dg = prm.get("dict_box_scale", 1.0), prm.get("dict_size_gamma", 1.0)
        zr = np.zeros(3); ar_ = np.zeros(4)
        for x0, y0, x1, y1 in top:
            bw_, bh_ = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
            if dg != 1.0:
                bw_ = np.exp(LOGW0 + dg * (np.log(bw_) - LOGW0))
                bh_ = np.exp(LOGH0 + dg * (np.log(bh_) - LOGH0))
            zr[zone_of((y0 + y1) / 2)] += 1
            ar_[abin_of(bw_ * ds * bh_ * ds)] += 1
        zv = best_dict(cnt_p, np.maximum(zr, 1e-6))
        av = best_dict(cnt_p, np.maximum(ar_, 1e-6))
    else:
        if zv is None: zv = best_dict(cnt_p, np.maximum(zon, 1e-6))
        if av is None: av = best_dict(cnt_p, np.maximum(abn, 1e-6))

    return {"area_bin_counts": {b: int(av[i]) for i, b in enumerate(ABINS)},
            "occupied_cells": cells,
            "region_cards": cards,
            "region_count": int(n_med),
            "region_count_bin": cbin,
            "zone_counts": {z: int(zv[i]) for i, z in enumerate(ZONES)}}


def decode_all(O, idx, prm, knn=None, cnt_override=None, zv=None, av=None):
    """cnt_override must ALREADY be rebased; zv/av may be precomputed for speed."""
    cs = rebase(O["cnt"], prm.get("empty_prior")) if cnt_override is None else cnt_override
    return [decode_row(O["peak"][i], O["grid"][i], cs[i], O["zon"][i], O["abn"][i], prm,
                       None if knn is None else knn[i],
                       None if zv is None else zv[i], None if av is None else av[i])
            for i in idx]
