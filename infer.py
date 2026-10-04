import json, sys, os, glob, argparse
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Net
from data import to_chw, FH, FW
from train import decode as decode_peaks
from decode import decode_all
from knn import knn_count_post

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
W = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK"


def _flip_back(o):
    o = dict(o)
    o["hm"] = torch.flip(o["hm"], [3])
    o["grid"] = torch.flip(o["grid"], [2])
    o["wh"] = torch.flip(o["wh"], [3])
    off = torch.flip(o["off"], [3])
    o["off"] = torch.stack([1 - off[:, 0], off[:, 1]], 1)
    return o


@torch.no_grad()
def predict_ensemble(nets, imgs, idx, bs=24, tta=True):
    for n in nets: n.eval()
    out = {"peak": [], "grid": [], "cnt": [], "zon": [], "abn": []}
    out.update({"hmp": [], "wh": [], "off": []})
    for s in range(0, len(idx), bs):
        ii = idx[s:s + bs]
        x = torch.from_numpy(np.stack([to_chw(np.asarray(imgs[i]).astype(np.float32)) for i in ii]))
        x = x.to(memory_format=torch.channels_last)
        xf = torch.flip(x, [3]).contiguous(memory_format=torch.channels_last) if tta else None
        acc = None; k = 0
        for net in nets:
            for o in ([net(x)] + ([_flip_back(net(xf))] if tta else [])):
                cur = {"hmp": torch.sigmoid(o["hm"]), "wh": o["wh"], "off": o["off"],
                       "grid": torch.sigmoid(o["grid"]), "cnt": F.softmax(o["cnt"], 1),
                       "zon": F.softplus(o["zon"]), "abn": F.softplus(o["abn"])}
                acc = cur if acc is None else {kk: acc[kk] + cur[kk] for kk in acc}
                k += 1
        acc = {kk: v / k for kk, v in acc.items()}
        p = acc["hmp"].clamp(1e-6, 1 - 1e-6)
        out["peak"].append(decode_peaks({"hm": torch.log(p / (1 - p)),
                                         "wh": acc["wh"], "off": acc["off"]}).numpy())
        for kk in ("grid", "cnt", "zon", "abn", "hmp", "wh", "off"):
            out[kk].append(acc[kk].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--out", default=r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_SUB\submission.csv")
    a = ap.parse_args()
    torch.set_num_threads(16)
    tr = pd.read_csv(D + r"\train.csv"); te = pd.read_csv(D + r"\test.csv")
    ntr, nte = len(tr), len(te)
    imgs = np.load(W + r"\imgs_288.npy", mmap_mode="r")
    cks = sorted(glob.glob(f"{W}\\net_{a.tag}_f*.pt"))
    print("checkpoints:", [os.path.basename(c) for c in cks])
    nets = []
    for c in cks:
        n = Net(1.0).to(memory_format=torch.channels_last)
        n.load_state_dict(torch.load(c, map_location="cpu"))
        nets.append(n)
    P = predict_ensemble(nets, imgs, np.arange(ntr, ntr + nte))
    np.savez(f"{W}\\test_pred_{a.tag}.npz", **P)

    # retrieval posterior from ALL labelled rows
    X = np.load(W + r"\desc24.npy")
    cnt = np.array([json.loads(s)["region_count"] for s in tr.answer_json])
    knn = knn_count_post(X[ntr:] @ X[:ntr].T, cnt, k=12, tau=0.02)
    np.save(f"{W}\\knn_test.npy", knn)

    prm = json.load(open(f"{W}\\prm_{a.tag}.json"))
    print("params:", prm)

    # same three-way count fusion the decode was tuned with; the peak->count table
    # is fitted on the full OOF (test rows were never part of it)
    from fuse import peak_counts, build_table, geo_blend
    from decode import rebase, precompute_dicts
    O = {k: v for k, v in np.load(f"{W}\\oof_{a.tag}.npz").items()}
    tbl = build_table(peak_counts(O["peak"], 0.25), cnt)
    p_peak = tbl[np.clip(peak_counts(P["peak"], 0.25), 0, tbl.shape[0] - 1)]
    fusedp = geo_blend([P["cnt"], knn, p_peak], [1.0, prm["w_knn"], prm["w_peak"]])
    fusedp = rebase(fusedp, prm["empty_prior"])
    zv, av = precompute_dicts(fusedp, P["zon"], P["abn"])
    np.save(f"{W}\\test_cnt_{a.tag}.npy", fusedp)
    ans = decode_all(P, np.arange(nte), prm, knn, fusedp, zv, av)
    sub = pd.DataFrame({"id": te.id, "answer_json": [json.dumps(x, sort_keys=True, separators=(",", ":")) for x in ans]})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sub.to_csv(a.out, index=False)
    print("wrote", a.out, sub.shape)
    print("count dist:", np.bincount([x["region_count"] for x in ans], minlength=11).tolist())
    print("P(empty) predicted:", np.mean([x["region_count"] == 0 for x in ans]).round(4))


if __name__ == "__main__":
    main()
