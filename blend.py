"""Blend the two full-data models with the three grouped-CV fold models.

Both prediction sets already exist, so this is a 5-model ensemble for the cost of
one inference pass. Averaging happens on the DENSE heads in the original frame,
before a single decode -- averaging decoded boxes would be wrong.
Weight is split evenly between the two ensembles, which gives the full-data
models (trained on 1800 rows rather than 1200) the larger per-model share.
"""
import json, sys, os, glob, argparse
import numpy as np, pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Net
from infer import predict_ensemble
from train import decode as decode_peaks
from decode import decode_all, rebase, precompute_dicts

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
W = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w_full", type=float, default=0.5)
    ap.add_argument("--prm", default="prm_final3.json")
    ap.add_argument("--out", default=r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_SUB\submission_blend5.csv")
    a = ap.parse_args()
    torch.set_num_threads(16)
    tr = pd.read_csv(D + r"\train.csv"); te = pd.read_csv(D + r"\test.csv")
    ntr, nte = len(tr), len(te)

    fold_cache = W + r"\test_heads_folds.npz"
    if os.path.exists(fold_cache):
        F = {k: v for k, v in np.load(fold_cache).items()}
        print("loaded cached fold-model heads")
    else:
        imgs = np.load(W + r"\imgs_288.npy", mmap_mode="r")
        nets = []
        for c in sorted(glob.glob(f"{W}\\net_v1_f*.pt")):
            n = Net(1.0).to(memory_format=torch.channels_last)
            n.load_state_dict(torch.load(c, map_location="cpu", weights_only=True))
            nets.append(n)
        print("fold checkpoints:", len(nets))
        F = predict_ensemble(nets, imgs, np.arange(ntr, ntr + nte))
        np.savez(fold_cache, **{k: v for k, v in F.items() if k != "peak"})
    S = {k: v for k, v in np.load(W + r"\sol_pred.npz").items()}

    wf = a.w_full
    B = {}
    for k in ("hmp", "wh", "off", "grid", "cnt", "zon", "abn"):
        B[k] = wf * S[k] + (1 - wf) * F[k]
    p = np.clip(B["hmp"], 1e-6, 1 - 1e-6)
    B["peak"] = decode_peaks({"hm": torch.from_numpy(np.log(p / (1 - p))),
                              "wh": torch.from_numpy(B["wh"]),
                              "off": torch.from_numpy(B["off"])}).numpy()

    prm = json.load(open(f"{W}\\{a.prm}"))
    cntp = rebase(B["cnt"], prm["empty_prior"])
    zv, av = precompute_dicts(cntp, B["zon"], B["abn"])
    ans = decode_all(B, np.arange(nte), prm, S["knn"], cntp, zv, av)
    pd.DataFrame({"id": te.id,
                  "answer_json": [json.dumps(x, sort_keys=True, separators=(",", ":")) for x in ans]}
                 ).to_csv(a.out, index=False)
    n0 = np.array([x["region_count"] for x in ans])
    print(f"wrote {a.out}")
    print(f"  w_full={wf}  count dist {np.bincount(n0, minlength=9).tolist()}  "
          f"P(empty) {float((n0 == 0).mean()):.4f}  cards/row {np.mean([len(x['region_cards']) for x in ans]):.3f}")


if __name__ == "__main__":
    main()
