from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Sample:
    sample_id: str
    x_path: str
    y_path: str
    source: str


def source_from_stem(stem: str) -> str:
    if "_" not in stem:
        return stem
    return stem.rsplit("_", 1)[0]


def collect_samples(x_dir: Path, y_dir: Path) -> List[Sample]:
    x_files = {p.stem: p for p in x_dir.glob("*.npy")}
    y_files = {p.stem: p for p in y_dir.glob("*.npy")}

    paired_stems = sorted(set(x_files) & set(y_files))

    return [
        Sample(
            sample_id=stem,
            x_path=str(Path("X") / x_files[stem].name),
            y_path=str(Path("Y") / y_files[stem].name),
            source=source_from_stem(stem),
        )
        for stem in paired_stems
    ]


def split_by_source(
    samples: Sequence[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Sample]]:
    grouped: Dict[str, List[Sample]] = {}
    for s in samples:
        grouped.setdefault(s.source, []).append(s)

    sources = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(sources)

    if len(sources) < 3:
        return split_by_sample(samples, train_ratio, val_ratio, seed)

    n = len(sources)
    n_train = max(1, int(round(train_ratio * n)))
    n_val = max(1, int(round(val_ratio * n)))

    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)

    train_sources = set(sources[:n_train])
    val_sources = set(sources[n_train:n_train + n_val])
    test_sources = set(sources[n_train + n_val:])

    if not test_sources:
        moved = next(iter(val_sources))
        val_sources.remove(moved)
        test_sources.add(moved)

    out = {"train": [], "val": [], "test": []}
    for s in samples:
        if s.source in train_sources:
            out["train"].append(s)
        elif s.source in val_sources:
            out["val"].append(s)
        else:
            out["test"].append(s)

    return out


def split_by_sample(
    samples: Sequence[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Sample]]:
    items = list(samples)
    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))

    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val

    if n_test <= 0:
        n_val -= 1
        n_test = 1

    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }


def write_outputs(splits: Dict[str, List[Sample]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "x_path", "y_path", "source", "split"])
        for split_name in ("train", "val", "test"):
            for s in splits[split_name]:
                writer.writerow([s.sample_id, s.x_path, s.y_path, s.source, split_name])

    for split_name in ("train", "val", "test"):
        txt_path = out_dir / f"{split_name}.txt"
        with txt_path.open("w", encoding="utf-8") as f:
            for s in splits[split_name]:
                f.write(f"{s.sample_id}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic train/val/test splits for X/Y tiles.")
    parser.add_argument("--x-dir", type=Path, default=Path("kaggle_upload/X"))
    parser.add_argument("--y-dir", type=Path, default=Path("kaggle_upload/Y"))
    parser.add_argument("--out-dir", type=Path, default=Path("kaggle_upload/splits"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy",
        choices=["source", "sample"],
        default="source",
        help="source: split by source tile prefix when possible; sample: split by random sample.",
    )
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if min(args.train_ratio, args.val_ratio, test_ratio) <= 0:
        raise ValueError("train/val/test ratios must all be positive and sum to < 1.")

    samples = collect_samples(args.x_dir, args.y_dir)
    if len(samples) < 3:
        raise ValueError("Need at least 3 paired samples to create train/val/test splits.")

    if args.strategy == "source":
        splits = split_by_source(samples, args.train_ratio, args.val_ratio, args.seed)
        used_group_split = len({s.source for s in samples}) >= 3
    else:
        splits = split_by_sample(samples, args.train_ratio, args.val_ratio, args.seed)
        used_group_split = False

    write_outputs(splits, args.out_dir)

    print("Split creation complete")
    print(f"Total paired samples: {len(samples)}")
    print(f"Train: {len(splits['train'])}")
    print(f"Val:   {len(splits['val'])}")
    print(f"Test:  {len(splits['test'])}")
    print(f"Strategy used: {'source' if used_group_split else 'sample'}")
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
