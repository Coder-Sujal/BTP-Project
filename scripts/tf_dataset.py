from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required for tf_dataset.py. Install it in Kaggle or your training environment."
    ) from exc


AUTOTUNE = tf.data.AUTOTUNE


@dataclass(frozen=True)
class Record:
    sample_id: str
    x_path: str
    y_path: str
    source: str
    split: str


def _resolve_data_path(path_str: str, data_root: Path) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((data_root / p).resolve())


def load_manifest(manifest_csv: Path, data_root: Optional[Path] = None) -> List[Record]:
    records: List[Record] = []
    if data_root is None:
        data_root = manifest_csv.parent.parent

    with manifest_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "x_path", "y_path", "source", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        for row in reader:
            records.append(
                Record(
                    sample_id=row["sample_id"],
                    x_path=_resolve_data_path(row["x_path"], data_root),
                    y_path=_resolve_data_path(row["y_path"], data_root),
                    source=row["source"],
                    split=row["split"],
                )
            )

    if not records:
        raise ValueError(f"Manifest has no rows: {manifest_csv}")

    return records


def filter_split(records: Sequence[Record], split: str) -> List[Record]:
    out = [r for r in records if r.split == split]
    if not out:
        raise ValueError(f"No records found for split='{split}'")
    return out


def compute_channel_stats(
    records: Sequence[Record],
    max_files: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    subset = list(records[:max_files]) if max_files is not None else list(records)

    channel_sum = None
    channel_sq_sum = None
    count = 0

    for rec in subset:
        x = np.load(rec.x_path).astype(np.float64)
        if x.ndim != 3:
            raise ValueError(f"Expected HxWxC for X, got {x.shape} in {rec.x_path}")

        flat = x.reshape(-1, x.shape[-1])

        if channel_sum is None:
            channel_sum = np.zeros(flat.shape[-1], dtype=np.float64)
            channel_sq_sum = np.zeros(flat.shape[-1], dtype=np.float64)

        channel_sum += flat.sum(axis=0)
        channel_sq_sum += np.square(flat).sum(axis=0)
        count += flat.shape[0]

    if count == 0 or channel_sum is None or channel_sq_sum is None:
        raise ValueError("No data found while computing channel statistics.")

    mean = channel_sum / count
    var = np.maximum(channel_sq_sum / count - np.square(mean), 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def _load_pair_np(x_path: bytes, y_path: bytes) -> Tuple[np.ndarray, np.ndarray]:
    x = np.load(x_path.decode("utf-8")).astype(np.float32)
    y = np.load(y_path.decode("utf-8")).astype(np.float32)

    if y.ndim == 2:
        y = np.expand_dims(y, axis=-1)
    elif y.ndim != 3 or y.shape[-1] != 1:
        raise ValueError(f"Expected mask shape HxW or HxWx1, got {y.shape}")

    # Enforce binary masks for segmentation loss stability.
    y = (y > 0.5).astype(np.float32)
    return x, y


def _tf_load_pair(x_path: tf.Tensor, y_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x, y = tf.numpy_function(_load_pair_np, [x_path, y_path], [tf.float32, tf.float32])

    x.set_shape([None, None, None])
    y.set_shape([None, None, 1])

    return x, y


def _normalize(x: tf.Tensor, y: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = (x - mean) / (std + 1e-6)
    return x, y


def _augment(x: tf.Tensor, y: tf.Tensor, seed: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    # Stateless ops keep augmentations deterministic per element given a seed.
    seed1 = tf.stack([seed[0], seed[1]])
    seed2 = tf.stack([seed[0], seed[1] + 1])
    seed3 = tf.stack([seed[0], seed[1] + 2])

    do_lr = tf.random.stateless_uniform([], seed1) > 0.5
    do_ud = tf.random.stateless_uniform([], seed2) > 0.5
    k = tf.random.stateless_uniform([], seed3, minval=0, maxval=4, dtype=tf.int32)

    x = tf.cond(do_lr, lambda: tf.image.flip_left_right(x), lambda: x)
    y = tf.cond(do_lr, lambda: tf.image.flip_left_right(y), lambda: y)

    x = tf.cond(do_ud, lambda: tf.image.flip_up_down(x), lambda: x)
    y = tf.cond(do_ud, lambda: tf.image.flip_up_down(y), lambda: y)

    x = tf.image.rot90(x, k=k)
    y = tf.image.rot90(y, k=k)

    return x, y


def build_dataset(
    records: Sequence[Record],
    batch_size: int,
    mean: np.ndarray,
    std: np.ndarray,
    training: bool,
    shuffle_buffer: int = 2048,
    cache: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    x_paths = [r.x_path for r in records]
    y_paths = [r.y_path for r in records]

    ds = tf.data.Dataset.from_tensor_slices((x_paths, y_paths))

    if training:
        ds = ds.shuffle(buffer_size=min(shuffle_buffer, len(records)), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(_tf_load_pair, num_parallel_calls=AUTOTUNE)

    mean_tf = tf.constant(mean.reshape((1, 1, -1)), dtype=tf.float32)
    std_tf = tf.constant(std.reshape((1, 1, -1)), dtype=tf.float32)
    ds = ds.map(lambda x, y: _normalize(x, y, mean_tf, std_tf), num_parallel_calls=AUTOTUNE)

    if training:
        counter = tf.data.experimental.Counter(start=0)
        seeds = counter.map(lambda c: tf.stack([tf.cast(seed, tf.int64), c]))
        ds = tf.data.Dataset.zip((ds, seeds))
        ds = ds.map(lambda xy, s: _augment(xy[0], xy[1], s), num_parallel_calls=AUTOTUNE)

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_train_val_test_datasets(
    manifest_csv: Path,
    data_root: Optional[Path] = None,
    batch_size: int = 16,
    train_stats_max_files: Optional[int] = None,
    cache: bool = False,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray]:
    records = load_manifest(manifest_csv, data_root=data_root)

    train_records = filter_split(records, "train")
    val_records = filter_split(records, "val")
    test_records = filter_split(records, "test")

    mean, std = compute_channel_stats(train_records, max_files=train_stats_max_files)

    train_ds = build_dataset(train_records, batch_size, mean, std, training=True, cache=cache, seed=seed)
    val_ds = build_dataset(val_records, batch_size, mean, std, training=False, cache=cache, seed=seed)
    test_ds = build_dataset(test_records, batch_size, mean, std, training=False, cache=cache, seed=seed)

    return train_ds, val_ds, test_ds, mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and sanity-check tf.data datasets from manifest.csv")
    parser.add_argument("--manifest", type=Path, default=Path("kaggle_upload/splits/manifest.csv"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root that contains X/ and Y/. Defaults to manifest parent parent.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stats-max-files", type=int, default=None)
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    train_ds, val_ds, test_ds, mean, std = build_train_val_test_datasets(
        args.manifest,
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_stats_max_files=args.stats_max_files,
        cache=args.cache,
    )

    train_batch = next(iter(train_ds.take(1)))
    x_b, y_b = train_batch

    print("tf.data pipeline ready")
    print(f"mean: {mean.tolist()}")
    print(f"std:  {std.tolist()}")
    print(f"train batch X shape: {x_b.shape}")
    print(f"train batch Y shape: {y_b.shape}")
    print(f"val batches: {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"test batches: {tf.data.experimental.cardinality(test_ds).numpy()}")


if __name__ == "__main__":
    main()
