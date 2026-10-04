"""Independent estimate of the test count prior via Saerens-Latinne-Decaestecker EM.

Uses only the trained classifier's posteriors on the (unlabelled) test images, so
it is an independent check on the similarity-band estimate of P(empty).
"""
import numpy as np


def em_prior(post, prior_train, iters=2000, tol=1e-10, floor=2e-3):
    """post: (n, K) train-conditioned posteriors. Returns the estimated test prior.

    Classes with a (near-)zero training prior make the pi/prior ratio explode and
    the iteration collapses onto them, so the training prior is floored and both
    distributions are restricted to the support that actually carries mass.
    """
    keep = prior_train > floor
    p0 = prior_train[keep] / prior_train[keep].sum()
    P = post[:, keep]
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
    pi = p0.copy()
    for _ in range(iters):
        q = P * (pi / p0)
        q /= np.maximum(q.sum(1, keepdims=True), 1e-12)
        new = q.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    out = np.zeros_like(prior_train)
    out[keep] = pi
    return out / out.sum()


if __name__ == "__main__":
    import json, sys, os
    import pandas as pd
    D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
    W = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK"
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    tr = pd.read_csv(D + r"\train.csv")
    cnt = np.array([json.loads(s)["region_count"] for s in tr.answer_json])
    prior_tr = np.bincount(np.clip(cnt, 0, 12), minlength=13).astype(float)
    prior_tr /= prior_tr.sum()

    O = np.load(f"{W}\\oof_{tag}.npz")
    fold = O["fold"]; have = fold >= 0
    P = np.load(f"{W}\\test_pred_{tag}.npz")["cnt"]

    print("train prior            :", np.round(prior_tr, 4).tolist())
    print("mean OOF posterior     :", np.round(O["cnt"][have].mean(0), 4).tolist())
    print("mean TEST posterior    :", np.round(P.mean(0), 4).tolist())
    print()
    print(f"OOF  mean P(empty) = {O['cnt'][have][:,0].mean():.4f}   (train truth 0.3956)")
    print(f"TEST mean P(empty) = {P[:,0].mean():.4f}")
    pi = em_prior(P, prior_tr)
    print(f"\nEM-estimated TEST prior: {np.round(pi, 4).tolist()}")
    print(f"EM-estimated P(empty)  : {pi[0]:.4f}")
    print(f"EM-estimated mean count: {(pi * np.arange(13)).sum():.4f}   (train {cnt.mean():.4f})")
    # sanity: EM applied to the OOF posteriors should recover the TRAIN prior
    pio = em_prior(O["cnt"][have], prior_tr)
    print(f"\n[sanity] EM on OOF -> P(empty) {pio[0]:.4f} (should be ~0.396)")
