"""Shared: exact scorer + exact label-derivation contract (reverse-engineered)."""
import json, math
import numpy as np

R, C = 10, 12
ZONES = ["upper", "middle", "lower"]
ABINS = ["tiny", "small", "medium", "large"]
AREA_THR = (0.008, 0.026, 0.07)   # tiny|small|medium|large  (verified: zero overlap in train)


# ----------------------------------------------------------------- derivation
def cell_name(r, c):
    return "r%02d_c%02d" % (r, c)

def zone_of(cy):
    return "upper" if cy < 1.0 / 3 else ("middle" if cy < 2.0 / 3 else "lower")

def area_bin_of(a):
    return ABINS[0] if a < AREA_THR[0] else ABINS[1] if a < AREA_THR[1] else ABINS[2] if a < AREA_THR[2] else ABINS[3]

def center_cell_of(x0, y0, x1, y1):
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    r = min(R - 1, max(0, int(math.floor(cy * R + 1e-9))))
    c = min(C - 1, max(0, int(math.floor(cx * C + 1e-9))))
    return cell_name(r, c)

def cells_of_boxes(boxes):
    out = set()
    for x0, y0, x1, y1 in boxes:
        c0 = min(C - 1, max(0, int(math.floor(x0 * C + 1e-9))))
        c1 = min(C - 1, max(0, int(math.ceil(x1 * C - 1e-9)) - 1))
        r0 = min(R - 1, max(0, int(math.floor(y0 * R + 1e-9))))
        r1 = min(R - 1, max(0, int(math.ceil(y1 * R - 1e-9)) - 1))
        for r in range(r0, max(r0, r1) + 1):
            for c in range(c0, max(c0, c1) + 1):
                out.add(cell_name(r, c))
    return out

def count_bin_of(n):
    return "none" if n == 0 else "few" if n <= 2 else "several" if n <= 5 else "many"

def clip01(v):
    return float(min(1.0, max(0.0, v)))

def build_answer(boxes):
    """boxes: iterable of (x0,y0,x1,y1) normalized. Returns the full answer dict."""
    bs = []
    for b in boxes:
        x0, y0, x1, y1 = [clip01(v) for v in b]
        if x1 <= x0: x1 = min(1.0, x0 + 1e-3)
        if y1 <= y0: y1 = min(1.0, y0 + 1e-3)
        bs.append((round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)))
    # rank by area, descending
    bs.sort(key=lambda b: -((b[2] - b[0]) * (b[3] - b[1])))
    bs = bs[:12]
    n = len(bs)
    zc = {z: 0 for z in ZONES}
    ac = {a: 0 for a in ABINS}
    cards = []
    for i, (x0, y0, x1, y1) in enumerate(bs):
        cy = (y0 + y1) / 2.0
        ar = (x1 - x0) * (y1 - y0)
        z, ab = zone_of(cy), area_bin_of(ar)
        zc[z] += 1
        ac[ab] += 1
        if i < 8:
            cards.append({"area_bin": ab, "bbox": [x0, y0, x1, y1],
                          "center_cell": center_cell_of(x0, y0, x1, y1),
                          "region_rank": i + 1, "zone": z})
    return {"area_bin_counts": ac,
            "occupied_cells": sorted(cells_of_boxes(bs)),
            "region_cards": cards,
            "region_count": n,
            "region_count_bin": count_bin_of(n),
            "zone_counts": zc}


# --------------------------------------------------------------------- scorer
def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0: return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def _set_f1(p, t):
    if not p and not t: return 1.0
    if not p or not t: return 0.0
    inter = len(p & t)
    if inter == 0: return 0.0
    prec, rec = inter / len(p), inter / len(t)
    return 2 * prec * rec / (prec + rec)

def _dict_score(p, t, keys):
    tot = sum(p.get(k, 0) for k in keys) + sum(t.get(k, 0) for k in keys)
    if tot == 0: return 1.0
    err = sum(abs(p.get(k, 0) - t.get(k, 0)) for k in keys)
    return max(0.0, 1 - err / tot)

def _card_score(pc, tc):
    if not pc and not tc: return 1.0
    if not pc or not tc: return 0.0
    used = set()
    tot = 0.0
    for p in pc:
        best, bi = 0.0, -1
        for j, t in enumerate(tc):
            if j in used: continue
            s = (0.45 * _iou(p["bbox"], t["bbox"]) ** 2
                 + 0.20 * (p["center_cell"] == t["center_cell"])
                 + 0.15 * (p["zone"] == t["zone"])
                 + 0.15 * (p["area_bin"] == t["area_bin"])
                 + 0.05 * (p["region_rank"] == t["region_rank"]))
            if s > best: best, bi = s, j
        if bi >= 0: used.add(bi)
        tot += best
    mp, mr = tot / len(pc), tot / len(tc)
    return 0.0 if mp + mr == 0 else 2 * mp * mr / (mp + mr)

def score_row(pred, true):
    cf = _set_f1(set(pred["occupied_cells"]), set(true["occupied_cells"]))
    cs = max(0.0, 1 - abs(pred["region_count"] - true["region_count"]) / 12.0)
    bm = 1.0 if pred["region_count_bin"] == true["region_count_bin"] else 0.0
    zs = _dict_score(pred["zone_counts"], true["zone_counts"], ZONES)
    as_ = _dict_score(pred["area_bin_counts"], true["area_bin_counts"], ABINS)
    ks = _card_score(pred["region_cards"], true["region_cards"])
    return min(1.0, max(0.0, 0.24 * cf ** 2 + 0.12 * cs + 0.06 * bm + 0.14 * zs + 0.12 * as_ + 0.32 * ks))

def score_all(preds, trues):
    return float(np.mean([score_row(p, t) for p, t in zip(preds, trues)]))

def score_parts(preds, trues):
    out = {}
    for nm, fn in [("cell_f1", lambda p, t: _set_f1(set(p["occupied_cells"]), set(t["occupied_cells"]))),
                   ("count", lambda p, t: max(0.0, 1 - abs(p["region_count"] - t["region_count"]) / 12.0)),
                   ("bin", lambda p, t: float(p["region_count_bin"] == t["region_count_bin"])),
                   ("zone", lambda p, t: _dict_score(p["zone_counts"], t["zone_counts"], ZONES)),
                   ("area", lambda p, t: _dict_score(p["area_bin_counts"], t["area_bin_counts"], ABINS)),
                   ("card", lambda p, t: _card_score(p["region_cards"], t["region_cards"]))]:
        out[nm] = float(np.mean([fn(p, t) for p, t in zip(preds, trues)]))
    out["TOTAL"] = score_all(preds, trues)
    return out
