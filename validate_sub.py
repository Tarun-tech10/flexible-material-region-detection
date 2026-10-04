"""Strict format check of a submission against the published spec."""
import json, sys, re
import pandas as pd

D = r"c:\Users\tarun\OneDrive\Desktop\ERIS\CONTAINER_DATA"
CELL = re.compile(r"^r(0\d)_c(0\d|1[01])$")
ZONES = ["upper", "middle", "lower"]
ABINS = ["tiny", "small", "medium", "large"]
CBINS = ["none", "few", "several", "many"]
FIELDS = {"region_count", "region_count_bin", "occupied_cells", "zone_counts",
          "area_bin_counts", "region_cards"}
CARDF = {"region_rank", "center_cell", "zone", "area_bin", "bbox"}


def cell_ok(s):
    m = CELL.match(s)
    if not m: return False
    return 0 <= int(m.group(1)) <= 9 and 0 <= int(m.group(2)) <= 11


def main(path):
    te = pd.read_csv(D + r"\test.csv")
    sub = pd.read_csv(path)
    errs = []
    if list(sub.columns) != ["id", "answer_json"]:
        errs.append(f"columns are {list(sub.columns)}, expected ['id','answer_json']")
    if len(sub) != len(te):
        errs.append(f"{len(sub)} rows, expected {len(te)}")
    if sub.id.duplicated().any():
        errs.append("duplicate ids")
    if set(sub.id) != set(te.id):
        errs.append(f"id set mismatch: missing {len(set(te.id)-set(sub.id))}, extra {len(set(sub.id)-set(te.id))}")
    if list(sub.id) != list(te.id):
        errs.append("id ORDER differs from test.csv (usually fine, but flagged)")

    ncards = []
    for i, s in enumerate(sub.answer_json):
        try:
            a = json.loads(s)
        except Exception as e:
            errs.append(f"row {i}: bad JSON ({e})"); continue
        if set(a) != FIELDS:
            errs.append(f"row {i}: fields {sorted(set(a) ^ FIELDS)} differ"); continue
        rc = a["region_count"]
        if not isinstance(rc, int) or isinstance(rc, bool) or not (0 <= rc <= 12):
            errs.append(f"row {i}: region_count {rc!r}")
        if a["region_count_bin"] not in CBINS:
            errs.append(f"row {i}: bad count bin {a['region_count_bin']!r}")
        oc = a["occupied_cells"]
        if not isinstance(oc, list) or any(not isinstance(c, str) or not cell_ok(c) for c in oc):
            errs.append(f"row {i}: bad cell name(s)")
        elif oc != sorted(oc):
            errs.append(f"row {i}: cells not sorted")
        elif len(set(oc)) != len(oc):
            errs.append(f"row {i}: duplicate cells")
        for nm, keys in (("zone_counts", ZONES), ("area_bin_counts", ABINS)):
            d = a[nm]
            if not isinstance(d, dict) or set(d) != set(keys):
                errs.append(f"row {i}: {nm} keys"); continue
            for k in keys:
                v = d[k]
                if not isinstance(v, int) or isinstance(v, bool) or not (0 <= v <= 12):
                    errs.append(f"row {i}: {nm}[{k}]={v!r}")
        cards = a["region_cards"]
        if not isinstance(cards, list) or len(cards) > 8:
            errs.append(f"row {i}: region_cards length {len(cards) if isinstance(cards,list) else '?'}")
            continue
        ncards.append(len(cards))
        for j, c in enumerate(cards):
            if set(c) != CARDF:
                errs.append(f"row {i} card {j}: fields"); continue
            if c["region_rank"] != j + 1:
                errs.append(f"row {i} card {j}: rank {c['region_rank']} != {j+1}")
            if not cell_ok(c["center_cell"]):
                errs.append(f"row {i} card {j}: center_cell {c['center_cell']!r}")
            if c["zone"] not in ZONES or c["area_bin"] not in ABINS:
                errs.append(f"row {i} card {j}: zone/area_bin")
            b = c["bbox"]
            if (not isinstance(b, list) or len(b) != 4
                    or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in b)):
                errs.append(f"row {i} card {j}: bbox type"); continue
            if not (0 <= b[0] <= 1 and 0 <= b[1] <= 1 and 0 <= b[2] <= 1 and 0 <= b[3] <= 1):
                errs.append(f"row {i} card {j}: bbox range {b}")
            if not (b[2] > b[0] and b[3] > b[1]):
                errs.append(f"row {i} card {j}: bbox not positive-area {b}")

    print(f"rows {len(sub)}   cards/row mean {sum(ncards)/max(len(ncards),1):.2f}")
    if errs:
        print(f"FAILED: {len(errs)} problems")
        for e in errs[:25]: print("  ", e)
    else:
        print("PASSED: submission is well-formed")
    return len(errs)


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1]) else 0)
