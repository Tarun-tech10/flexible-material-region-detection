"""Test-distribution correction.

Train is 39.6% empty; the test set is ~14.5% empty (see shift.py / shift2.py).
Within the non-empty rows the count distribution is unchanged (mean 2.74 in train,
2.68-2.78 in every low-similarity band), so the shift is a pure two-class
reweighting of P(empty).
"""
import numpy as np

P_EMPTY_TEST = 0.145
P_EMPTY_TRAIN = 712.0 / 1800.0


def row_weights(cnt, p_test=P_EMPTY_TEST, p_train=P_EMPTY_TRAIN):
    """Importance weights making a train/OOF set look like the test set."""
    w = np.where(np.asarray(cnt) == 0, p_test / p_train, (1 - p_test) / (1 - p_train))
    return w / w.mean()


def prior_ratio(p_test=P_EMPTY_TEST, p_train=P_EMPTY_TRAIN):
    r = np.ones(13, np.float64) * ((1 - p_test) / (1 - p_train))
    r[0] = p_test / p_train
    return r


def correct_counts(cnt_p, r):
    """p'(c) proportional to p(c) * pi_test(c)/pi_train(c)."""
    q = cnt_p * r
    s = q.sum(-1, keepdims=True)
    return np.where(s > 0, q / np.maximum(s, 1e-12), cnt_p)


def wscore(preds, trues, w, score_row):
    s = np.array([score_row(p, t) for p, t in zip(preds, trues)])
    return float((s * w).sum() / w.sum())
