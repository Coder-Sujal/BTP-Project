from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


def list_npy_stems(folder: Path) -> Set[str]:
    return {p.stem for p in folder.glob("*.npy")}


def source_from_stem(stem: str) -> str:
    if "_" not in stem:
        return stem
    return stem.rsplit("_", 1)[0]


def check_dataset(x_dir: Path, y_dir: Path, limit: int | None = None) -> Dict:
    x_stems = list_npy_stems(x_dir)
    y_stems = list_npy_stems(y_dir)

    paired = sorted(x_stems & y_stems)
    only_x = sorted(x_stems - y_stems)
    only_y = sorted(y_stems - x_stems)

    report: Dict = {
        "total_x": len(x_stems),
        "total_y": len(y_stems),
        "paired": len(paired),
        "only_x": len(only_x),
        "only_y": len(only_y),
        "examples_only_x": only_x[:10],
        "examples_only_y": only_y[:10],
        "shape_mismatches": [],
        "x_non_finite": 0,
        "mask_non_binary": 0,
        "mask_unique_values": {},
        "x_dtype_counts": {},
        "y_dtype_counts": {},
        "channel_min": None,
        "channel_max": None,
        "channel_mean": None,
        "channel_std": None,
        "mask_positive_pixels": 0,
        "mask_total_pixels": 0,
        "source_counts": {},
    }

    if not paired:
        return report

    if limit is not None:
        paired = paired[:limit]

    sample_for_stats = []
    channel_sum = None
    channel_sq_sum = None
    channel_count = 0

    for stem in paired:
        x_path = x_dir / f"{stem}.npy"
        y_path = y_dir / f"{stem}.npy"

        x = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")

        report["x_dtype_counts"][str(x.dtype)] = report["x_dtype_counts"].get(str(x.dtype), 0) + 1
        report["y_dtype_counts"][str(y.dtype)] = report["y_dtype_counts"].get(str(y.dtype), 0) + 1

        src = source_from_stem(stem)
        report["source_counts"][src] = report["source_counts"].get(src, 0) + 1

        if x.shape[:2] != y.shape[:2]:
            report["shape_mismatches"].append(
                {
                    "sample": stem,
                    "x_shape": tuple(int(v) for v in x.shape),
                    "y_shape": tuple(int(v) for v in y.shape),
                }
            )
            continue

        if x.ndim != 3:
            report["shape_mismatches"].append(
                {
                    "sample": stem,
                    "x_shape": tuple(int(v) for v in x.shape),
                    "y_shape": tuple(int(v) for v in y.shape),
                    "reason": "X must be HxWxC",
                }
            )
            continue

        y_vals = np.unique(y)
        report["mask_unique_values"][stem] = [int(v) for v in y_vals.tolist()]

        if not np.all(np.isin(y_vals, [0, 1])):
            report["mask_non_binary"] += 1

        if not np.isfinite(x).all():
            report["x_non_finite"] += 1

        x_f = np.asarray(x, dtype=np.float64)

        if channel_sum is None:
            c = x_f.shape[-1]
            channel_sum = np.zeros(c, dtype=np.float64)
            channel_sq_sum = np.zeros(c, dtype=np.float64)
            report["channel_min"] = [float("inf")] * c
            report["channel_max"] = [float("-inf")] * c

        reshaped = x_f.reshape(-1, x_f.shape[-1])
        channel_sum += reshaped.sum(axis=0)
        channel_sq_sum += np.square(reshaped).sum(axis=0)
        channel_count += reshaped.shape[0]

        ch_min = reshaped.min(axis=0)
        ch_max = reshaped.max(axis=0)
        report["channel_min"] = [float(min(a, b)) for a, b in zip(report["channel_min"], ch_min)]
        report["channel_max"] = [float(max(a, b)) for a, b in zip(report["channel_max"], ch_max)]

        y_arr = np.asarray(y)
        report["mask_positive_pixels"] += int(np.count_nonzero(y_arr))
        report["mask_total_pixels"] += int(y_arr.size)

        if len(sample_for_stats) < 10:
            sample_for_stats.append(stem)

    if channel_count > 0 and channel_sum is not None and channel_sq_sum is not None:
        mean = channel_sum / channel_count
        var = np.maximum(channel_sq_sum / channel_count - np.square(mean), 0.0)
        std = np.sqrt(var)
        report["channel_mean"] = [float(v) for v in mean]
        report["channel_std"] = [float(v) for v in std]

    if report["mask_total_pixels"] > 0:
        report["mask_positive_ratio"] = report["mask_positive_pixels"] / report["mask_total_pixels"]
    else:
        report["mask_positive_ratio"] = 0.0

    # This can be huge; keep only examples for readability.
    report["mask_unique_values_examples"] = {
        k: report["mask_unique_values"][k] for k in list(report["mask_unique_values"].keys())[:20]
    }
    del report["mask_unique_values"]

    report["checked_samples"] = len(paired)
    report["sampled_examples"] = sample_for_stats

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate X/Y npy segmentation dataset integrity.")
    parser.add_argument("--x-dir", type=Path, default=Path("kaggle_upload/X"))
    parser.add_argument("--y-dir", type=Path, default=Path("kaggle_upload/Y"))
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on samples to inspect.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("kaggle_upload/integrity_report.json"),
        help="Where to write report JSON.",
    )
    args = parser.parse_args()

    report = check_dataset(args.x_dir, args.y_dir, limit=args.limit)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Integrity check complete")
    print(f"Paired samples: {report['paired']}")
    print(f"Checked samples: {report.get('checked_samples', 0)}")
    print(f"Only in X: {report['only_x']}, only in Y: {report['only_y']}")
    print(f"Shape mismatches: {len(report['shape_mismatches'])}")
    print(f"X non-finite samples: {report['x_non_finite']}")
    print(f"Mask non-binary samples: {report['mask_non_binary']}")
    print(f"Mask positive ratio: {report.get('mask_positive_ratio', 0.0):.6f}")
    print(f"Report saved: {args.output_json}")


if __name__ == "__main__":
    main()
