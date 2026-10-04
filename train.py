import json, sys, time, argparse, os
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Net, focal
import data as DT
from data import (augment, make_targets, norm_hwc, batch_from_hwc, ZONES, ABINS)

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
W = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK"
TOPK = 16


def load():
    tr = pd.read_csv(D + r"\train.csv")
    A = [json.loads(s) for s in tr.answer_json]
    boxes = [np.array([c["bbox"] for c in a["region_cards"]], np.float32).reshape(-1, 4) for a in A]
    cnt = np.array([a["region_count"] for a in A], np.int64)
    zon = np.array([[a["zone_counts"][z] for z in ZONES] for a in A], np.float32)
    abn = np.array([[a["area_bin_counts"][b] for b in ABINS] for a in A], np.float32)
    # load fully into RAM: the cache lives on a synced folder, and re-reading it
    # every epoch through a memory map dominated the step time
    imgs = np.load(W + f"\\imgs_{DT.IW}.npy")
    assert imgs.shape[1:3] == (DT.IH, DT.IW), (imgs.shape, DT.IH, DT.IW)
    return tr, A, boxes, cnt, zon, abn, imgs


def batch_tensor(idx, imgs, boxes, cnt, zon, abn, rng, train=True):
    xs, ts = [], []
    for i in idx:
        im = imgs[i]
        b = boxes[i]
        if train:
            im, b = augment(im, b, rng)
        else:
            im = im.astype(np.float32)
        xs.append(norm_hwc(im))
        ts.append(make_targets(b, cnt[i], zon[i], abn[i]))
    x = batch_from_hwc(xs)
    T = {k: torch.from_numpy(np.stack([t[k] for t in ts])) for k in ts[0]}
    return x, T


def loss_fn(o, T):
    l_hm = focal(o["hm"][:, 0], T["hm"])
    m = T["msk"].unsqueeze(1)
    npos = m.sum().clamp(min=1)
    l_wh = (torch.abs(o["wh"] - T["wh"]) * m).sum() / npos
    l_off = (torch.abs(o["off"] - T["off"]) * m).sum() / npos
    l_grid = F.binary_cross_entropy_with_logits(o["grid"], T["grid"])
    l_cnt = F.cross_entropy(o["cnt"], T["cnt"], label_smoothing=0.03)
    l_zon = F.l1_loss(F.softplus(o["zon"]), T["zon"])
    l_abn = F.l1_loss(F.softplus(o["abn"]), T["abn"])
    tot = l_hm + 1.0 * l_wh + 1.0 * l_off + 2.0 * l_grid + 1.0 * l_cnt + 0.15 * (l_zon + l_abn)
    return tot, dict(hm=l_hm.item(), wh=l_wh.item(), off=l_off.item(), grid=l_grid.item(),
                     cnt=l_cnt.item(), zon=l_zon.item(), abn=l_abn.item())


@torch.no_grad()
def decode(o, topk=TOPK):
    """-> peaks (B,topk,5): score,x0,y0,x1,y1 normalized."""
    hm = torch.sigmoid(o["hm"][:, 0])
    keep = (F.max_pool2d(hm.unsqueeze(1), 3, 1, 1).squeeze(1) - hm).abs() < 1e-9
    hm = hm * keep
    B = hm.shape[0]
    fh, fw = hm.shape[1], hm.shape[2]
    flat = hm.reshape(B, -1)
    sc, ind = flat.topk(topk, dim=1)
    ys = torch.div(ind, fw, rounding_mode="floor").float(); xs = (ind % fw).float()
    wh = o["wh"].reshape(B, 2, -1).gather(2, ind.unsqueeze(1).expand(-1, 2, -1))
    off = o["off"].reshape(B, 2, -1).gather(2, ind.unsqueeze(1).expand(-1, 2, -1))
    cx = (xs + off[:, 0]) / fw
    cy = (ys + off[:, 1]) / fh
    bw = torch.exp(wh[:, 0].clamp(-6, 4)) / fw
    bh = torch.exp(wh[:, 1].clamp(-6, 4)) / fh
    return torch.stack([sc, (cx - bw / 2).clamp(0, 1), (cy - bh / 2).clamp(0, 1),
                        (cx + bw / 2).clamp(0, 1), (cy + bh / 2).clamp(0, 1)], -1)


@torch.no_grad()
def predict(net, imgs, idx, bs=32, tta=True):
    net.eval()
    P = {"peak": [], "grid": [], "cnt": [], "zon": [], "abn": []}
    for s in range(0, len(idx), bs):
        ii = idx[s:s + bs]
        x = batch_from_hwc([norm_hwc(np.asarray(imgs[i]).astype(np.float32)) for i in ii])
        outs = [net(x)]
        if tta:
            # flip input, then map every spatial head back into the ORIGINAL frame
            of = net(torch.flip(x, [3]).contiguous(memory_format=torch.channels_last))
            of["hm"] = torch.flip(of["hm"], [3])
            of["grid"] = torch.flip(of["grid"], [2])
            of["wh"] = torch.flip(of["wh"], [3])
            of["off"] = torch.flip(of["off"], [3])
            of["off"] = torch.stack([1 - of["off"][:, 0], of["off"][:, 1]], 1)
            outs.append(of)
        # average the dense heads in the original frame, then decode once
        avg = {k: torch.stack([o[k] for o in outs]).mean(0) for k in ("wh", "off")}
        avg["hm"] = torch.log(torch.stack([torch.sigmoid(o["hm"]) for o in outs]).mean(0)
                              .clamp(1e-6, 1 - 1e-6) / (1 - torch.stack(
                                  [torch.sigmoid(o["hm"]) for o in outs]).mean(0).clamp(1e-6, 1 - 1e-6)))
        P["peak"].append(decode(avg).numpy())
        P["grid"].append(torch.stack([torch.sigmoid(o["grid"]) for o in outs]).mean(0).numpy())
        P["cnt"].append(torch.stack([F.softmax(o["cnt"], 1) for o in outs]).mean(0).numpy())
        P["zon"].append(torch.stack([F.softplus(o["zon"]) for o in outs]).mean(0).numpy())
        P["abn"].append(torch.stack([F.softplus(o["abn"]) for o in outs]).mean(0).numpy())
    return {k: np.concatenate(v) for k, v in P.items()}


def run_fold(tri, vai, imgs, boxes, cnt, zon, abn, epochs, seed, wm, bs, lr, log=print):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net(wm).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    steps = epochs * max(1, len(tri) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.25)
    t0 = time.time(); it = 0
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tri)
        agg = {}
        for s in range(0, len(perm) - bs + 1, bs):
            x, T = batch_tensor(perm[s:s + bs], imgs, boxes, cnt, zon, abn, rng, True)
            o = net(x)
            l, parts = loss_fn(o, T)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            if it < steps - 1: sch.step()
            it += 1
            for k, v in parts.items(): agg[k] = agg.get(k, 0) + v
        if ep % 5 == 0 or ep == epochs - 1:
            n = max(1, len(perm) // bs)
            log(f"    ep{ep:3d} " + " ".join(f"{k}{v/n:.3f}" for k, v in agg.items())
                + f"  {time.time()-t0:.0f}s")
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only", type=int, default=-1)
    ap.add_argument("--wm", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="v1")
    ap.add_argument("--ih", type=int, default=192)
    ap.add_argument("--iw", type=int, default=288)
    a = ap.parse_args()
    torch.set_num_threads(16)
    DT.set_res(a.ih, a.iw)
    print(f"resolution {a.iw}x{a.ih}  feature {DT.FW}x{DT.FH}", flush=True)

    tr, A, boxes, cnt, zon, abn, imgs = load()
    grp = np.load(W + r"\grp.npy")
    from sklearn.model_selection import GroupKFold
    splits = list(GroupKFold(a.folds).split(np.arange(len(tr)), groups=grp))

    OOF = {k: None for k in ["peak", "grid", "cnt", "zon", "abn"]}
    fold_id = np.full(len(tr), -1)
    for f, (tri, vai) in enumerate(splits):
        if a.only >= 0 and f != a.only: continue
        print(f"[fold {f}] train {len(tri)} val {len(vai)}", flush=True)
        net = run_fold(tri, vai, imgs, boxes, cnt, zon, abn, a.epochs, a.seed + f, a.wm, a.bs, a.lr,
                       log=lambda s: print(s, flush=True))
        P = predict(net, imgs, vai)
        for k in OOF:
            if OOF[k] is None:
                sh = (len(tr),) + P[k].shape[1:]
                OOF[k] = np.zeros(sh, np.float32)
            OOF[k][vai] = P[k]
        fold_id[vai] = f
        torch.save(net.state_dict(), f"{W}\\net_{a.tag}_f{f}.pt")
        np.savez(f"{W}\\oof_{a.tag}.npz", fold=fold_id, **{k: v for k, v in OOF.items() if v is not None})
        print(f"[fold {f}] saved", flush=True)
    print("done")


if __name__ == "__main__":
    main()
