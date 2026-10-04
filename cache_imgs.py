"""Cache all 2400 images to a uint8 npy at native and working resolution."""
import sys, numpy as np, pandas as pd
from PIL import Image

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
W = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK"
tr = pd.read_csv(D + r"\train.csv"); te = pd.read_csv(D + r"\test.csv")
paths = list(tr.image_path) + list(te.image_path)

for (w, h, tag) in [(512, 340, "512"), (320, 224, "320")]:
    arr = np.zeros((len(paths), h, w, 3), np.uint8)
    for i, p in enumerate(paths):
        im = Image.open(f"{D}\\" + p.replace("/", "\\")).convert("RGB")
        if im.size != (w, h):
            im = im.resize((w, h), Image.BILINEAR)
        arr[i] = np.asarray(im)
        if i % 600 == 0: print(tag, i, flush=True)
    np.save(f"{W}\\imgs_{tag}.npy", arr)
    print("saved", tag, arr.shape, arr.nbytes / 1e6, "MB", flush=True)
np.save(W + r"\ids.npy", np.array(list(tr.id) + list(te.id)))
print("done")
