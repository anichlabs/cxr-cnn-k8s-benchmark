#!/usr/bin/env python3
"""
audit_near_duplicates.py  (v2)
------------------------------
Detect near-identical images across the clean corpus splits by direct pixel
comparison.

Why not perceptual hashing
--------------------------
The first version of this script used an 8x8 difference hash. That failed
badly on chest radiographs: every CXR shares the same gross structure, so at
8x8 resolution they are nearly indistinguishable and a 64-bit hash cannot
separate them. It reported 37,580 "near-duplicate" pairs whose class labels
agreed only 50 per cent of the time, which is what chance looks like.

This version compares images directly. Each image is reduced to 64x64
greyscale and z-scored, so brightness and contrast differences are removed.
The similarity between two images is then the Pearson correlation of their
pixel vectors, computed for the whole corpus in a single matrix product.
Correlation is interpretable and has a meaningful scale: 1.0 is identical,
around 0.0 is unrelated.

Mirrored copies are handled by also correlating against the horizontally
flipped corpus and taking the larger value.

Choosing a threshold
--------------------
Run with --diagnose first. It prints the distribution of similarities across
all pairs, split by whether the pair shares a class and a source. A usable
threshold sits well above the bulk of the distribution. If no such gap exists,
there are no near-duplicates to find and the check should be reported as
negative rather than forced.

Usage
-----
    python scripts/audit_near_duplicates.py --diagnose
    python scripts/audit_near_duplicates.py --threshold 0.95
    python scripts/audit_near_duplicates.py --threshold 0.95 --emit-groups

Outputs
-------
    reports/tables/near_duplicate_pairs.csv     pairs at or above threshold
    reports/tables/near_duplicate_groups.csv    cluster assignment (--emit-groups)

Exit status is 1 if any cross-split pair is found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "experiments" / "pools_clean"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

SPLITS = ("train", "val", "test")
SIDE = 64  # Comparison resolution.


def load_corpus() -> pd.DataFrame:
    """Load the three splits into one frame, tagged by split name."""
    frames = []
    for split in SPLITS:
        path = SPLITS_DIR / f"{split}_split.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        frame = pd.read_csv(path)
        frame["split"] = split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_matrix(df: pd.DataFrame, side: int = SIDE) -> tuple[np.ndarray, np.ndarray]:
    """
    Load every image as a z-scored vector, returning the corpus matrix and its
    horizontally mirrored counterpart.

    Z-scoring each image individually removes global brightness and contrast,
    so the correlation reflects structure rather than exposure.
    """
    n = len(df)
    straight = np.zeros((n, side * side), dtype=np.float32)
    mirrored = np.zeros((n, side * side), dtype=np.float32)

    for i, path in enumerate(df["abs_path"]):
        if i % 250 == 0:
            print(f"  loading {i:,} / {n:,}", flush=True)
        try:
            img = Image.open(path).convert("L").resize(
                (side, side), Image.Resampling.LANCZOS
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not read {path}: {exc}") from exc

        arr = np.asarray(img, dtype=np.float32)
        flip = arr[:, ::-1]

        for target, source in ((straight, arr), (mirrored, flip)):
            vec = source.flatten()
            vec = vec - vec.mean()
            norm = np.linalg.norm(vec)
            # A constant image has zero variance and no meaningful correlation.
            target[i] = vec / norm if norm > 0 else vec

    print(f"  loading {n:,} / {n:,}")
    return straight, mirrored


def similarity_matrix(straight: np.ndarray, mirrored: np.ndarray) -> np.ndarray:
    """
    Pairwise similarity for the whole corpus.

    Vectors are already centred and unit-norm, so the dot product is the
    Pearson correlation. The mirrored comparison catches left-right flips.
    """
    sim = straight @ straight.T
    np.maximum(sim, straight @ mirrored.T, out=sim)
    np.fill_diagonal(sim, -1.0)
    return sim


def describe(sim: np.ndarray, df: pd.DataFrame) -> None:
    """Print the similarity distribution so a threshold can be chosen on evidence."""
    iu = np.triu_indices(len(sim), k=1)
    vals = sim[iu]

    same_class = (df["Class"].values[iu[0]] == df["Class"].values[iu[1]])
    same_source = (df["source_true"].values[iu[0]] == df["source_true"].values[iu[1]])

    print()
    print("--- SIMILARITY DISTRIBUTION ---")
    print(f"pairs compared: {len(vals):,}")
    print()
    print(f"{'percentile':>12s} {'all':>9s} {'same class':>11s} {'same source':>12s}")
    for p in (50, 90, 99, 99.9, 99.99):
        print(f"{p:>11.2f}% {np.percentile(vals, p):9.4f} "
              f"{np.percentile(vals[same_class], p):11.4f} "
              f"{np.percentile(vals[same_source], p):12.4f}")
    print(f"{'max':>12s} {vals.max():9.4f} "
          f"{vals[same_class].max():11.4f} {vals[same_source].max():12.4f}")

    print()
    print("Pair counts above candidate thresholds:")
    print(f"{'threshold':>10s} {'pairs':>10s} {'same class':>11s} {'agreement':>10s}")
    for t in (0.90, 0.93, 0.95, 0.97, 0.99):
        mask = vals >= t
        count = int(mask.sum())
        if count == 0:
            print(f"{t:>10.2f} {count:>10,} {'-':>11s} {'-':>10s}")
            continue
        agree = int(same_class[mask].sum())
        print(f"{t:>10.2f} {count:>10,} {agree:>11,} {agree / count:>9.1%}")

    print()
    print("Interpretation: genuine duplicates agree on class almost always. "
          "A threshold whose pairs agree near chance is picking up ordinary "
          "anatomical similarity, not duplication.")


def cluster(pairs: list[tuple[int, int, float]], n: int) -> np.ndarray:
    """Assign a cluster id per row via union-find over the reported pairs."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    return np.array([find(i) for i in range(n)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Minimum correlation to treat as a near-duplicate "
                             "(default 0.95). Run --diagnose to choose.")
    parser.add_argument("--diagnose", action="store_true",
                        help="Print the similarity distribution and exit.")
    parser.add_argument("--emit-groups", action="store_true",
                        help="Write a cluster assignment for build_clean_corpus.py "
                             "to consume as a group override.")
    args = parser.parse_args()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading splits ...")
    df = load_corpus()
    print(f"  {len(df):,} images across {df['split'].nunique()} splits")

    print(f"Loading images at {SIDE}x{SIDE} ...")
    straight, mirrored = load_matrix(df)

    print("Computing pairwise similarity ...")
    sim = similarity_matrix(straight, mirrored)

    if args.diagnose:
        describe(sim, df)
        return 0

    iu = np.triu_indices(len(sim), k=1)
    mask = sim[iu] >= args.threshold
    rows, cols, vals = iu[0][mask], iu[1][mask], sim[iu][mask]
    print(f"  {len(rows):,} pairs at or above correlation {args.threshold}")

    if len(rows) == 0:
        print()
        print(f"No near-duplicates at correlation {args.threshold}. Splits are "
              "clean at this threshold.")
        return 0

    records = []
    for i, j, val in zip(rows, cols, vals):
        a, b = df.iloc[int(i)], df.iloc[int(j)]
        records.append({
            "similarity": round(float(val), 4),
            "cross_split": a["split"] != b["split"],
            "split_a": a["split"], "split_b": b["split"],
            "class_a": a["Class"], "class_b": b["Class"],
            "source_a": a["source_true"], "source_b": b["source_true"],
            "path_a": a["rel_path"], "path_b": b["rel_path"],
        })

    pairs_df = pd.DataFrame(records).sort_values(
        ["cross_split", "similarity"], ascending=[False, False]
    )
    out = TABLE_DIR / "near_duplicate_pairs.csv"
    pairs_df.to_csv(out, index=False)

    cross = pairs_df[pairs_df["cross_split"]]
    within = pairs_df[~pairs_df["cross_split"]]
    agree = int((pairs_df["class_a"] == pairs_df["class_b"]).sum())

    print()
    print("--- NEAR-DUPLICATE REPORT ---")
    print(f"within-split pairs : {len(within):,}")
    print(f"cross-split pairs  : {len(cross):,}")
    print(f"class agreement    : {agree:,} / {len(pairs_df):,} "
          f"({agree / len(pairs_df):.1%})")

    if agree / len(pairs_df) < 0.8:
        print()
        print("WARNING: class agreement is low. At this threshold the script is "
              "likely matching ordinary anatomical similarity rather than "
              "duplication. Raise the threshold and re-run --diagnose before "
              "acting on these pairs.")

    if len(cross):
        print()
        print("Cross-split pairs by source:")
        print(cross.groupby(["source_a", "source_b"]).size().to_string())
        print()
        print("Closest ten:")
        print(cross.head(10)[
            ["similarity", "split_a", "split_b", "class_a", "class_b",
             "path_a", "path_b"]
        ].to_string(index=False))

    print()
    print(f"Full pair list written to: {out.relative_to(PROJECT_ROOT)}")

    if args.emit_groups:
        ids = cluster(list(zip(rows.tolist(), cols.tolist(), vals.tolist())), len(df))
        groups = pd.DataFrame({
            "rel_path": df["rel_path"],
            "source_true": df["source_true"],
            "original_group": df["group"],
            "merged_group": [f"dupclust_{i}" for i in ids],
        })
        sizes = groups.groupby("merged_group").size()
        gpath = TABLE_DIR / "near_duplicate_groups.csv"
        groups.to_csv(gpath, index=False)
        print(f"Group override written to: {gpath.relative_to(PROJECT_ROOT)}")
        print(f"  {int((sizes > 1).sum()):,} clusters contain more than one image")

    if len(cross):
        print()
        print("RESULT: cross-split near-duplicates detected. Inspect the pair "
              "list before rebuilding: confirm visually that the closest pairs "
              "really are the same radiograph.")
        return 1

    print()
    print("RESULT: no cross-split near-duplicates. Splits are sound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
