#!/usr/bin/env python3
"""Rebuild a correctly-labelled, patient-grouped, verifiably disjoint CXR corpus."""
import re, sys, hashlib
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[1]
TBX  = ROOT / "data/tbx11k/imgs"
POOL = ROOT / "experiments/pools/source_pool_rel.csv"
OUT  = ROOT / "experiments/pools_clean"
SEED, DRY = 42, "--write" not in sys.argv
rng = np.random.default_rng(SEED)

rows, unmatched = [], []
def add(f, cls, src, grp): rows.append(dict(abs_path=str(f), Class=cls, source=src, group=grp))

for f in sorted((TBX/"tb").glob("*")):
    if f.is_file(): add(f, "TB", "tbx11k", f.stem)
for f in sorted((TBX/"health").glob("*")):
    if f.is_file(): add(f, "Normal", "tbx11k", f.stem)

for f in sorted((TBX/"extra/mc+shenzhen").rglob("*")):
    if not f.is_file(): continue
    m = re.search(r"_(\d)\.[A-Za-z]+$", f.name)
    if m: add(f, "TB" if m.group(1)=="1" else "Normal", "mc+shenzhen", f.stem)
    else: unmatched.append(str(f))

for f in sorted((TBX/"extra/da+db").rglob("*")):
    if not f.is_file(): continue
    c = f.name[0].lower()
    if   c == "n": add(f, "Normal", "da+db", f.stem)
    elif c == "p": add(f, "TB",     "da+db", f.stem)
    else: unmatched.append(str(f))

pool = pd.read_csv(POOL)
chex = pool[pool.rel_path.str.startswith("train/patient")]
for r in chex.itertuples():
    add(ROOT/"data/chexpert"/r.rel_path, r.Class, "chexpert",
        re.search(r"(patient\d+)", r.rel_path).group(1))

df = pd.DataFrame(rows)
print("=== label rules applied ===")
print(df.groupby(["Class","source"]).size().to_string())
if unmatched:
    print(f"\nUNMATCHED ({len(unmatched)}) - refusing to proceed:")
    for u in unmatched[:20]: print("  ", u)
    sys.exit(1)

missing = [p for p in df.abs_path if not Path(p).exists()]
if missing:
    print(f"\nMISSING ({len(missing)}) e.g. {missing[:3]}"); sys.exit(1)

df["hash"] = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in df.abs_path]
before = len(df); df = df.drop_duplicates("hash")
print(f"\nexact-duplicate images removed: {before-len(df)}")

cap = df.groupby("Class").size().min()
print(f"balancing to {cap} per class")
keep = []
for cls, g in df.groupby("Class"):
    order = rng.permutation(g.group.unique()); n, sel = 0, []
    for grp in order:
        if n >= cap: break
        sub = g[g.group == grp]; sel.append(sub); n += len(sub)
    keep.append(pd.concat(sel).head(cap))
df = pd.concat(keep).reset_index(drop=True)

parts = {"train": [], "val": [], "test": []}
for cls, g in df.groupby("Class"):
    order = rng.permutation(g.group.unique()); n = len(g); c = 0
    for grp in order:
        sub = g[g.group == grp]
        split = "train" if c < 0.70*n else ("val" if c < 0.85*n else "test")
        parts[split].append(sub); c += len(sub)
splits = {k: pd.concat(v).reset_index(drop=True) for k, v in parts.items()}

for a in splits:
    for b in splits:
        if a >= b: continue
        assert not (set(splits[a].group)  & set(splits[b].group)),  f"group leak {a}/{b}"
        assert not (set(splits[a].hash)   & set(splits[b].hash)),   f"hash leak {a}/{b}"
        assert not (set(splits[a].abs_path)&set(splits[b].abs_path)),f"path leak {a}/{b}"
print("\nassertions passed: splits are disjoint by path, content hash and patient group")
for k, v in splits.items():
    print(f"{k:5s} {len(v):5d}  " + " ".join(f"{c}={n}" for c, n in v.Class.value_counts().items()))

if DRY: print("\nDRY RUN - rerun with --write to save"); sys.exit(0)
OUT.mkdir(exist_ok=True)
for k, v in splits.items(): v.to_csv(OUT/f"{k}_split.csv", index=False)
print(f"\nwritten to {OUT}")
