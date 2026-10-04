# Container grid detection

Working code for an overhead industrial-container detection task: a CNN
(heatmap / box / grid-cell / count / zone / area-bin heads) plus kNN retrieval,
post-processing and parameter tuning that turn detections into the task's grid
labels (cells, count bin, zone, area bin, region cards).

Key files: `common.py` (scorer and label derivation), `data.py`, `model.py`,
`train.py`, `infer.py`, `decode.py`, `tune.py`, `blend.py`.

The dataset, cached image arrays (`*.npy`), model weights (`*.pt`) and
out-of-fold predictions (`*.npz`) are not included. Scripts use hardcoded
local paths (`D` = data dir, `W` = work dir) that need editing before running.

Licensed under the MIT License.
