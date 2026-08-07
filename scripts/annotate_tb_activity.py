#!/usr/bin/env python3
"""
annotate_tb_activity.py
-----------------------
Add a `tb_activity` column to the clean-corpus split CSVs.

Why
---
TBX11K annotates tuberculosis at three levels of activity, not one:

    ActiveTuberculosis              current disease
    ObsoletePulmonaryTuberculosis   healed, inactive; radiographic scarring
    PulmonaryTuberculosis           unresolved / general

These are clinically distinct. A screening model that flags healed scarring as
tuberculosis is not detecting infectious disease. The paper must therefore say
what its TB class actually contains, and cannot describe it simply as
"tuberculosis" without qualification.

The TB images drawn from Montgomery, Shenzhen, DA and DB carry a binary
positive/negative label with no activity sub-classification, so they are
recorded as `unlabelled`. That is a property of those datasets, not a gap in
this script, and must itself be disclosed.

What it does
------------
Reads the COCO-style annotation files shipped with TBX11K, resolves the set of
activity categories present per image, and writes the result back into
train/val/test split CSVs as a new column. Nothing else in the CSVs is touched
and no rows are added or removed, so the splits stay byte-comparable on every
other field.

Values written to `tb_activity`:

    active        ActiveTuberculosis only
    obsolete      ObsoletePulmonaryTuberculosis only
    both          both categories present on the same image
    pulmonary     PulmonaryTuberculosis only (unresolved category)
    unlabelled    TB image from a source without activity annotation
    n/a           not a TB image

Usage
-----
    python scripts/annotate_tb_activity.py --dry-run
    python scripts/annotate_tb_activity.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "experiments" / "pools_clean"
ANNO_DIR = PROJECT_ROOT / "data" / "tbx11k" / "annotations" / "json"

SPLITS = ("train", "val", "test")

# The annotation files that carry activity categories. all_trainval is the
# broadest set with released ground truth; all_test is withheld by TBX11K for
# their online competition and is not consulted.
ANNO_FILES = ("all_trainval.json", "all_train.json", "all_val.json",
              "TBX11K_trainval.json", "TBX11K_train.json", "TBX11K_val.json")

ACTIVE = "ActiveTuberculosis"
OBSOLETE = "ObsoletePulmonaryTuberculosis"
PULMONARY = "PulmonaryTuberculosis"


def build_activity_map() -> dict[str, str]:
    """
    Map file name to activity label by unioning every available annotation file.

    Files are read in order and merged. A given image should carry consistent
    annotations across files, since they are subsets of one labelling effort;
    any disagreement is reported rather than silently resolved.
    """
    per_image: dict[str, set[str]] = collections.defaultdict(set)
    files_read = []

    for name in ANNO_FILES:
        path = ANNO_DIR / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if "categories" not in data or "annotations" not in data:
            continue

        cats = {c["id"]: c["name"] for c in data["categories"]}
        imgs = {i["id"]: i["file_name"] for i in data["images"]}

        for ann in data["annotations"]:
            fname = imgs.get(ann["image_id"])
            cat = cats.get(ann["category_id"])
            if fname and cat:
                per_image[fname].add(cat)

        files_read.append(name)

    if not files_read:
        raise FileNotFoundError(
            f"No usable annotation files found in {ANNO_DIR}. "
            "Expected the TBX11K COCO-style JSON annotations."
        )

    print(f"Annotation files read: {', '.join(files_read)}")
    print(f"Images with TB annotations: {len(per_image):,}")

    resolved = {}
    for fname, cats in per_image.items():
        has_active = ACTIVE in cats
        has_obsolete = OBSOLETE in cats
        if has_active and has_obsolete:
            resolved[fname] = "both"
        elif has_active:
            resolved[fname] = "active"
        elif has_obsolete:
            resolved[fname] = "obsolete"
        elif PULMONARY in cats:
            resolved[fname] = "pulmonary"
        else:
            resolved[fname] = "unlabelled"

    return resolved


def classify(row: pd.Series, activity: dict[str, str]) -> str:
    """Resolve one corpus row to an activity label."""
    if row["Class"] != "TB":
        return "n/a"

    rel = row["rel_path"]

    # Annotation file names are relative to imgs/, matching rel_path directly
    # for the TBX11K-native folders. Try the path as given, then the bare stem
    # as a fallback for any naming variation.
    if rel in activity:
        return activity[rel]

    stem = Path(rel).name
    for key, val in activity.items():
        if Path(key).name == stem:
            return val

    # TB images from Montgomery, Shenzhen, DA and DB. Those datasets label TB
    # positivity without distinguishing activity.
    return "unlabelled"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the breakdown without writing the CSVs.")
    args = parser.parse_args()

    activity = build_activity_map()
    print()

    frames = {}
    tagged = []
    for split in SPLITS:
        path = SPLITS_DIR / f"{split}_split.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        df = pd.read_csv(path)
        if "tb_activity" in df.columns:
            df = df.drop(columns=["tb_activity"])
        df["tb_activity"] = df.apply(lambda r: classify(r, activity), axis=1)
        frames[split] = df
        tagged.append(df.assign(_split=split))

    combined = pd.concat(tagged, ignore_index=True)
    tb = combined[combined["Class"] == "TB"]

    print("--- TB ACTIVITY BREAKDOWN ---")
    print(f"TB images in corpus: {len(tb):,}")
    print()
    counts = tb["tb_activity"].value_counts()
    for label, n in counts.items():
        print(f"  {label:12s} {n:5,}  ({n / len(tb):5.1%})")

    print()
    print("By source:")
    print(pd.crosstab(tb["source_true"], tb["tb_activity"]).to_string())

    print()
    print("By split:")
    print(pd.crosstab(tb["_split"], tb["tb_activity"]).to_string())

    annotated = int((tb["tb_activity"] != "unlabelled").sum())
    print()
    print(f"Activity-annotated: {annotated:,} of {len(tb):,} "
          f"({annotated / len(tb):.1%})")
    print(f"Unlabelled (Montgomery/Shenzhen/DA/DB): {len(tb) - annotated:,}")

    if args.dry_run:
        print()
        print("Dry run. No files written.")
        return 0

    for split, df in frames.items():
        path = SPLITS_DIR / f"{split}_split.csv"
        df.to_csv(path, index=False)
        print(f"Wrote {path.relative_to(PROJECT_ROOT)} "
              f"({len(df):,} rows, {len(df.columns)} columns)")

    print()
    print("The tb_activity column is now available to NB04 for per-subclass "
          "recall, and to Section 3.1 for the class definition.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
