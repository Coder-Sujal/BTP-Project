from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import KAGGLE, MODELS, PROCESSED

PRED_TILE_DIR = PROCESSED / "path_planning" / "pred_tiles"
PRED_TILE_DIR.mkdir(parents=True, exist_ok=True)


def load_model_robust(model_path: Path):
    # Try tf.keras first.
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e_tf:
        tf_error = e_tf

    # Fallback to standalone keras when available (often needed for newer .keras archives).
    try:
        import keras

        return keras.models.load_model(model_path, compile=False)
    except Exception as e_keras:
        raise RuntimeError(
            "Failed to load model with both tf.keras and standalone keras. "
            "This is usually a version mismatch between training/runtime environments.\n"
            f"tf.keras error: {tf_error}\n"
            f"keras error: {e_keras}\n"
            "Run inference in the same environment family used to train/save the model "
            "(for example, your Kaggle runtime), or re-save/export the model in a compatible format."
        ) from e_keras


def resolve_model(model_path: Path | None) -> Path:
    if model_path is not None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return model_path

    candidates = sorted(
        list(MODELS.glob("*.keras")) + list(MODELS.glob("*.h5")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No model found in models/. Provide --model.")
    return candidates[0]


def collect_x_files(x_dir: Path, region: str | None = None) -> list[Path]:
    files = sorted(x_dir.glob("*.npy"))
    if region is None:
        return files
    prefix = f"dem_{region}_"
    return [p for p in files if p.stem.startswith(prefix)]


def _resolve_manifest_x_paths(manifest_csv: Path, data_root: Path) -> tuple[list[Path], list[Path]]:
    train_paths: list[Path] = []
    all_paths: list[Path] = []
    with manifest_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = Path(row["x_path"])
            if not p.is_absolute():
                p = (data_root / p).resolve()
            all_paths.append(p)
            if row.get("split", "") == "train":
                train_paths.append(p)
    return train_paths, all_paths


def compute_mean_std(paths: list[Path], max_files: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise ValueError("No files provided for statistics computation.")

    ch_sum = None
    ch_sq = None
    count = 0

    for p in paths:
        x = np.load(p).astype(np.float64)
        flat = x.reshape(-1, x.shape[-1])
        if ch_sum is None:
            ch_sum = np.zeros(flat.shape[-1], dtype=np.float64)
            ch_sq = np.zeros(flat.shape[-1], dtype=np.float64)
        ch_sum += flat.sum(axis=0)
        ch_sq += np.square(flat).sum(axis=0)
        count += flat.shape[0]

    mean = ch_sum / count
    var = np.maximum(ch_sq / count - np.square(mean), 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def save_stats(stats_json: Path, mean: np.ndarray, std: np.ndarray) -> None:
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2),
        encoding="utf-8",
    )


def load_stats(stats_json: Path) -> tuple[np.ndarray, np.ndarray]:
    d = json.loads(stats_json.read_text(encoding="utf-8"))
    mean = np.asarray(d["mean"], dtype=np.float32)
    std = np.asarray(d["std"], dtype=np.float32)
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model inference on X tiles and save crater probability tiles.")
    parser.add_argument("--model", type=Path, default=None, help="Path to .keras/.h5 model. Defaults to newest in models/.")
    parser.add_argument("--x-dir", type=Path, default=KAGGLE / "X")
    parser.add_argument("--out-dir", type=Path, default=PRED_TILE_DIR)
    parser.add_argument("--region", type=str, default=None, help="Optional region suffix filter, e.g. 80s")
    parser.add_argument("--manifest", type=Path, default=KAGGLE / "splits" / "manifest.csv")
    parser.add_argument("--stats-json", type=Path, default=MODELS / "normalization_stats.json")
    parser.add_argument("--stats-max-files", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of tiles for smoke testing.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_path = resolve_model(args.model)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    x_files = collect_x_files(args.x_dir, args.region)
    if args.limit is not None:
        x_files = x_files[: args.limit]
    if not x_files:
        raise ValueError("No input X tiles found to run inference.")

    if args.stats_json.exists():
        mean, std = load_stats(args.stats_json)
        stats_source = str(args.stats_json)
    else:
        if not args.manifest.exists():
            raise FileNotFoundError(
                f"Stats file not found and manifest missing: {args.stats_json}, {args.manifest}"
            )
        data_root = args.manifest.parent.parent
        train_paths, all_paths = _resolve_manifest_x_paths(args.manifest, data_root)
        base = train_paths if train_paths else all_paths
        mean, std = compute_mean_std(base, max_files=args.stats_max_files)
        save_stats(args.stats_json, mean, std)
        stats_source = f"computed from manifest and saved to {args.stats_json}"

    print(f"Loading model: {model_path}")
    model = load_model_robust(model_path)

    mean = mean.reshape(1, 1, 1, -1).astype(np.float32)
    std = std.reshape(1, 1, 1, -1).astype(np.float32)

    pending_files: list[Path] = []
    for p in x_files:
        out_p = out_dir / p.name
        if out_p.exists() and not args.overwrite:
            continue
        pending_files.append(p)

    if not pending_files:
        print("No pending tiles. Use --overwrite to force regeneration.")
        return

    print(f"Stats source: {stats_source}")
    print(f"Tiles selected: {len(x_files)} | Pending: {len(pending_files)}")

    total = len(pending_files)
    for i in range(0, total, args.batch_size):
        batch_files = pending_files[i : i + args.batch_size]
        x_batch = np.stack([np.load(p).astype(np.float32) for p in batch_files], axis=0)
        x_batch = (x_batch - mean) / (std + 1e-6)

        pred = model.predict(x_batch, verbose=0)
        if pred.ndim == 4 and pred.shape[-1] == 1:
            pred = pred[..., 0]
        pred = np.clip(pred.astype(np.float32), 0.0, 1.0)

        for p, y in zip(batch_files, pred):
            np.save(out_dir / p.name, y)

        done = min(i + args.batch_size, total)
        print(f"Inference progress: {done}/{total}")

    print("Inference complete")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
