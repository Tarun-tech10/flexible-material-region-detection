import json
import pandas as pd, numpy as np
from PIL import Image, ImageDraw

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
tr = pd.read_csv(D + r"\train.csv")
A = [json.loads(s) for s in tr.answer_json]
tr["n"] = [a["region_count"] for a in A]
tr["ans"] = A

rng = np.random.default_rng(0)
# montage: rows grouped by region_count, boxes drawn
groups = [0, 1, 2, 3, 5]
COLS = 5
tiles = []
for g in groups:
    sub = tr[tr.n == g]
    idx = rng.choice(len(sub), min(COLS, len(sub)), replace=False)
    row = []
    for i in idx:
        r = sub.iloc[i]
        im = Image.open(f"{D}\\{r.image_path}".replace("/", "\\")).convert("RGB")
        d = ImageDraw.Draw(im)
        for c in r.ans["region_cards"]:
            x0, y0, x1, y1 = c["bbox"]
            d.rectangle([x0 * 512, y0 * 340, x1 * 512, y1 * 340], outline=(255, 0, 0), width=3)
            d.text((x0 * 512 + 3, y0 * 340 + 3), f'{c["region_rank"]}{c["area_bin"][0]}', fill=(255, 255, 0))
        d.text((5, 5), f"n={g}", fill=(0, 255, 255))
        row.append(im)
    tiles.append(row)

W, H = 512, 340
out = Image.new("RGB", (W * COLS, H * len(groups)), (20, 20, 20))
for ri, row in enumerate(tiles):
    for ci, im in enumerate(row):
        out.paste(im, (ci * W, ri * H))
out.save(r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_WORK\montage.png")
print("saved montage", out.size)

im0 = Image.open(f"{D}\\{tr.iloc[0].image_path}".replace("/", "\\"))
print("mode/size:", im0.mode, im0.size)
a = np.asarray(im0.convert("RGB")).astype(np.float32)
print("per-channel mean/std:", a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0))
