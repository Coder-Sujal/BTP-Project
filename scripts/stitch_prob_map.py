from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from config import DEM, PROCESSED

PATH_DIR = PROCESSED / "path_planning"
PRED_TILE_DIR = PATH_DIR / "pred_tiles"
PROB_MAP_DIR = PATH_DIR / "prob_maps"
META_DIR = PATH_DIR / "meta"

for d in (PATH_DIR, PRED_TILE_DIR, PROB_MAP_DIR, META_DIR):
    d.mkdir(parents=True, exist_ok=True)


def parse_tile_index(stem: str, region: str) -> int | None:
    prefix = f"dem_{region}_"
    if not stem.startswith(prefix):
        return None
    idx_str = stem[len(prefix) :]
    if not idx_str.isdigit():
        return None
    return int(idx_str)


def stitch_region(pred_dir: Path, dem_file: Path, region: str, tile_size: int) -> tuple[np.ndarray, np.ndarray, dict]:
    with rasterio.open(dem_file) as src:
        h, w = src.shape
        profile = src.profile.copy()

    n_rows = ((h - tile_size) // tile_size) + 1
    n_cols = ((w - tile_size) // tile_size) + 1
    expected_tiles = n_rows * n_cols

    prob_sum = np.zeros((h, w), dtype=np.float32)
    prob_count = np.zeros((h, w), dtype=np.uint16)

    seen = 0
    skipped = 0
    for p in sorted(pred_dir.glob("*.npy")):
        idx = parse_tile_index(p.stem, region)
        if idx is None:
            continue

        tr = idx // n_cols
        tc = idx % n_cols
        if tr < 0 or tc < 0 or tr >= n_rows or tc >= n_cols:
            skipped += 1
            continue

        tile = np.load(p).astype(np.float32)
        if tile.ndim == 3 and tile.shape[-1] == 1:
            tile = tile[..., 0]
        if tile.shape != (tile_size, tile_size):
            skipped += 1
            continue

        r0 = tr * tile_size
        c0 = tc * tile_size
        r1 = r0 + tile_size
        c1 = c0 + tile_size

        prob_sum[r0:r1, c0:c1] += tile
        prob_count[r0:r1, c0:c1] += 1
        seen += 1

    prob = np.zeros((h, w), dtype=np.float32)
    valid = prob_count > 0
    prob[valid] = prob_sum[valid] / prob_count[valid]
    prob[~valid] = -1.0  # nodata for uncovered borders

    meta = {
        "region": region,
        "tile_size": tile_size,
        "dem_shape": [int(h), int(w)],
        "grid_rows": int(n_rows),
        "grid_cols": int(n_cols),
        "expected_tiles": int(expected_tiles),
        "tiles_loaded": int(seen),
        "tiles_skipped": int(skipped),
        "coverage_fraction": float(np.mean(valid)),
        "missing_tile_count": int(max(0, expected_tiles - seen)),
    }

    return prob, profile, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch predicted probability tiles into full-scene probability raster.")
    parser.add_argument("--region", type=str, required=True, help="Region suffix, e.g. 80s or 85s")
    parser.add_argument("--pred-dir", type=Path, default=PRED_TILE_DIR)
    parser.add_argument("--dem", type=Path, default=None, help="Optional DEM path")
    parser.add_argument("--tile-size", type=int, default=256)
    args = parser.parse_args()

    dem_file = args.dem if args.dem is not None else DEM / f"dem_{args.region}.tif"
    if not dem_file.exists():
        raise FileNotFoundError(f"DEM not found: {dem_file}")
    if not args.pred_dir.exists():
        raise FileNotFoundError(f"Prediction tile directory not found: {args.pred_dir}")

    prob, profile, meta = stitch_region(args.pred_dir, dem_file, args.region, args.tile_size)

    prob_out = PROB_MAP_DIR / f"prob_{args.region}.tif"
    meta_out = META_DIR / f"prob_{args.region}.json"

    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.float32, count=1, nodata=-1.0)

    with rasterio.open(prob_out, "w", **out_profile) as dst:
        dst.write(prob.astype(np.float32), 1)

    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Stitching complete")
    print(f"DEM: {dem_file}")
    print(f"Prediction tiles: {args.pred_dir}")
    print(f"Probability raster: {prob_out}")
    print(f"Metadata: {meta_out}")
    print(f"Coverage fraction: {meta['coverage_fraction']:.6f}")
    print(f"Tiles loaded: {meta['tiles_loaded']} / {meta['expected_tiles']}")


if __name__ == "__main__":
    main()

