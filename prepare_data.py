"""
prepare_data.py

Splits a Kaggle-style image dataset (one subfolder per class, full of images)
into a train/ and val/ split, keeping the same class-folder structure so
Keras' ImageDataGenerator.flow_from_directory can consume it directly.

Expected input layout (what `kaggle datasets download ... --unzip` gives you):

    data_raw/
        happy/
            img001.jpg
            img002.jpg
            ...
        sad/
            ...
        angry/
            ...

Produces:

    data/
        train/
            happy/...
            sad/...
        val/
            happy/...
            sad/...

Usage:
    python prepare_data.py --source data_raw --dest data --val_split 0.2
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_class_dirs(source: Path):
    """Return every immediate subdirectory of `source` that contains images.

    Handles datasets that nest everything one extra level deep (a common
    Kaggle quirk, e.g. data_raw/DogEmotion/happy/... instead of
    data_raw/happy/...).
    """
    candidates = [d for d in source.iterdir() if d.is_dir()]

    def has_images(d: Path) -> bool:
        return any(p.suffix.lower() in IMAGE_EXTS for p in d.glob("*"))

    class_dirs = [d for d in candidates if has_images(d)]
    if class_dirs:
        return class_dirs

    # nothing at this level has images directly -- descend one level
    nested = []
    for d in candidates:
        nested.extend(find_class_dirs(d))
    return nested


def split_class(class_dir: Path, dest: Path, val_split: float, seed: int):
    images = [p for p in class_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        print(f"  [skip] {class_dir} has no images")
        return 0, 0

    random.Random(seed).shuffle(images)
    n_val = max(1, int(len(images) * val_split))
    val_files = images[:n_val]
    train_files = images[n_val:]

    class_name = class_dir.name
    train_dir = dest / "train" / class_name
    val_dir = dest / "val" / class_name
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for f in train_files:
        shutil.copy2(f, train_dir / f.name)
    for f in val_files:
        shutil.copy2(f, val_dir / f.name)

    return len(train_files), len(val_files)


def main():
    ap = argparse.ArgumentParser(description="Split raw image dataset into train/val")
    ap.add_argument("--source", required=True, help="Path to raw downloaded dataset")
    ap.add_argument("--dest", required=True, help="Where to write train/ and val/")
    ap.add_argument("--val_split", type=float, default=0.2, help="Fraction held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    if not source.exists():
        raise SystemExit(f"Source folder not found: {source}")

    if dest.exists():
        print(f"Note: {dest} already exists, files will be added/overwritten.")

    class_dirs = find_class_dirs(source)
    if not class_dirs:
        raise SystemExit(
            f"No class subfolders with images found under {source}. "
            "Check the unzipped dataset layout."
        )

    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

    total_train, total_val = 0, 0
    for class_dir in sorted(class_dirs):
        n_train, n_val = split_class(class_dir, dest, args.val_split, args.seed)
        print(f"  {class_dir.name:15s} -> train={n_train:5d}  val={n_val:5d}")
        total_train += n_train
        total_val += n_val

    print(f"\nDone. train={total_train} images, val={total_val} images")
    print(f"Data ready at: {dest}/train and {dest}/val")


if __name__ == "__main__":
    main()
