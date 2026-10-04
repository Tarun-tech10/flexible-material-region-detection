import json, math
import numpy as np

IH, IW = 192, 288          # network input
ST = 8
FH, FW = IH // ST, IW // ST   # 24 x 36


def set_res(ih, iw, st=8):
    """Change the working resolution. Mutates module globals, so consumers must
    read data.FH / data.FW through the module rather than star-importing them."""
    global IH, IW, ST, FH, FW
    IH, IW, ST = ih, iw, st
    FH, FW = ih // st, iw // st
    return FH, FW
R, C = 10, 12
ZONES = ["upper", "middle", "lower"]
ABINS = ["tiny", "small", "medium", "large"]
AREA_THR = (0.008, 0.026, 0.07)
MEAN = np.float32([0.3734, 0.3801, 0.3635]) * 255   # measured on the released images
STD = np.float32([0.2848, 0.2879, 0.2915]) * 255


def gaussian_radius(h, w, min_overlap=0.7):
    a1, b1, c1 = 1, h + w, w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0))) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 - math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0))) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0))) / 2
    return max(0.0, min(r1, r2, r3))


def draw_gauss(hm, cx, cy, rad):
    d = max(1, int(rad))
    sigma = (2 * d + 1) / 6.0
    x = np.arange(-d, d + 1)
    g = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / (2 * sigma ** 2))
    ix, iy = int(cx), int(cy)
    l, r = min(ix, d), min(hm.shape[1] - ix, d + 1)
    t, b = min(iy, d), min(hm.shape[0] - iy, d + 1)
    if r <= -l or b <= -t: return
    np.maximum(hm[iy - t: iy + b, ix - l: ix + r], g[d - t: d + b, d - l: d + r],
               out=hm[iy - t: iy + b, ix - l: ix + r])


def zone_of(cy):
    return 0 if cy < 1 / 3 else (1 if cy < 2 / 3 else 2)

def abin_of(a):
    return 0 if a < AREA_THR[0] else 1 if a < AREA_THR[1] else 2 if a < AREA_THR[2] else 3

def cells_of(boxes):
    out = np.zeros((R, C), np.float32)
    for x0, y0, x1, y1 in boxes:
        c0 = min(C - 1, max(0, int(math.floor(x0 * C + 1e-9))))
        c1 = min(C - 1, max(0, int(math.ceil(x1 * C - 1e-9)) - 1))
        r0 = min(R - 1, max(0, int(math.floor(y0 * R + 1e-9))))
        r1 = min(R - 1, max(0, int(math.ceil(y1 * R - 1e-9)) - 1))
        out[r0:max(r0, r1) + 1, c0:max(c0, c1) + 1] = 1
    return out


# ------------------------------------------------------------------ augment
def augment(img, boxes, rng):
    """img uint8 HxWx3 (any size), boxes (N,4) normalized. Returns float32 img, boxes."""
    im = img.astype(np.float32)
    b = boxes.copy()
    # horizontal flip
    if rng.random() < 0.5:
        im = im[:, ::-1]
        if len(b):
            b = np.stack([1 - b[:, 2], b[:, 1], 1 - b[:, 0], b[:, 3]], 1)
    # translate (replicate padding), +-4%
    if rng.random() < 0.8:
        h, w = im.shape[:2]
        dx = int(round(rng.uniform(-0.04, 0.04) * w))
        dy = int(round(rng.uniform(-0.04, 0.04) * h))
        if dx or dy:
            im = np.roll(im, (dy, dx), (0, 1))
            if dy > 0: im[:dy] = im[dy:dy + 1]
            elif dy < 0: im[dy:] = im[dy - 1:dy]
            if dx > 0: im[:, :dx] = im[:, dx:dx + 1]
            elif dx < 0: im[:, dx:] = im[:, dx - 1:dx]
            if len(b):
                b[:, [0, 2]] = np.clip(b[:, [0, 2]] + dx / w, 0, 1)
                b[:, [1, 3]] = np.clip(b[:, [1, 3]] + dy / h, 0, 1)
    # photometric: brightness*contrast*channel-gain fuse into one scale+shift pass
    if rng.random() < 0.9:
        g1 = rng.uniform(0.75, 1.30); g2 = rng.uniform(0.75, 1.30)
        ch = rng.uniform(0.92, 1.08, 3).astype(np.float32)
        scale = (g1 * g2 * ch).astype(np.float32)
        shift = (g1 * float(im.mean()) * (1 - g2) * ch).astype(np.float32)
        im = im * scale + shift
    if rng.random() < 0.25:
        im = 255.0 * np.clip(im * (1.0 / 255.0), 0, 1) ** rng.uniform(0.75, 1.35)
    if rng.random() < 0.25:
        im = im + rng.standard_normal(im.shape, dtype=np.float32) * rng.uniform(2, 9)
    np.clip(im, 0, 255, out=im)
    # random erasing (occlusion robustness) - avoid wiping a whole box
    if rng.random() < 0.25:
        h, w = im.shape[:2]
        eh, ew = int(h * rng.uniform(0.05, 0.16)), int(w * rng.uniform(0.05, 0.16))
        y0, x0 = rng.integers(0, h - eh), rng.integers(0, w - ew)
        im[y0:y0 + eh, x0:x0 + ew] = rng.uniform(0, 255)
    return np.ascontiguousarray(im), b


def make_targets(boxes, count, zon_true, abn_true):
    """boxes (N,4) normalized (top-8 known regions). Returns target dict."""
    hm = np.zeros((FH, FW), np.float32)
    wh = np.zeros((2, FH, FW), np.float32)
    off = np.zeros((2, FH, FW), np.float32)
    msk = np.zeros((FH, FW), np.float32)
    for x0, y0, x1, y1 in boxes:
        bw, bh = (x1 - x0) * FW, (y1 - y0) * FH
        if bw <= 0 or bh <= 0: continue
        cx, cy = (x0 + x1) / 2 * FW, (y0 + y1) / 2 * FH
        ix, iy = min(FW - 1, int(cx)), min(FH - 1, int(cy))
        draw_gauss(hm, cx, cy, max(1.0, gaussian_radius(bh, bw)))
        wh[0, iy, ix] = math.log(max(bw, 1e-3))
        wh[1, iy, ix] = math.log(max(bh, 1e-3))
        off[0, iy, ix] = cx - ix
        off[1, iy, ix] = cy - iy
        msk[iy, ix] = 1
    if len(boxes) and count <= 8:
        cyv = (boxes[:, 1] + boxes[:, 3]) / 2
        ar = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        z = np.zeros(3, np.float32); a = np.zeros(4, np.float32)
        for v in cyv: z[zone_of(v)] += 1
        for v in ar: a[abin_of(v)] += 1
    else:
        z, a = np.float32(zon_true), np.float32(abn_true)
    return {"hm": hm, "wh": wh, "off": off, "msk": msk, "grid": cells_of(boxes),
            "cnt": np.int64(min(count, 12)), "zon": z, "abn": a}


def to_chw(im):
    """float32 HxWx3 in 0..255 -> normalized CHW float32."""
    return ((im - MEAN) / STD).transpose(2, 0, 1)


def norm_hwc(im):
    """float32 HxWx3 -> normalized HxWx3 (kept in HWC so the batch can become
    a channels_last tensor with no transpose)."""
    return (im - MEAN) / STD


def batch_from_hwc(xs):
    """list of HWC float32 -> (N,3,H,W) tensor whose memory layout IS channels_last."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(np.stack(xs))).permute(0, 3, 1, 2)
