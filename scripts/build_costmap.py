from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

from config import MASK, PROCESSED, SLOPE

PATH_DIR = PROCESSED / "path_planning"
COSTMAP_DIR = PATH_DIR / "costmaps"
BLOCKED_DIR = PATH_DIR / "blocked"
META_DIR = PATH_DIR / "meta"

for d in (PATH_DIR, COSTMAP_DIR, BLOCKED_DIR, META_DIR):
    d.mkdir(parents=True, exist_ok=True)


def normalize_slope(slope: np.ndarray, upper_q: float = 99.0) -> np.ndarray:
    slope_f = slope.astype(np.float32)
    slope_f = np.nan_to_num(slope_f, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = np.percentile(slope_f, upper_q)
    if vmax <= 0:
        return np.zeros_like(slope_f, dtype=np.float32)
    return np.clip(slope_f / vmax, 0.0, 1.0)


def build_costmap(
    crater_prob: np.ndarray,
    slope: np.ndarray,
    inflation_radius_px: int,
    crater_block_threshold: float,
    w_dist: float,
    w_crater: float,
    w_slope: float,
    w_clearance: float,
    max_clearance_px: float,
    blocked_cost: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    crater_prob = np.clip(crater_prob.astype(np.float32), 0.0, 1.0)
    slope_norm = normalize_slope(slope)

    hard_block = crater_prob >= crater_block_threshold

    # Distance to nearest hard-blocked crater pixel.
    dist_to_block = distance_transform_edt(~hard_block).astype(np.float32)

    inflated_block = dist_to_block <= float(inflation_radius_px)

    # High penalty near obstacles; decays with distance.
    clearance_penalty = np.clip(1.0 - (dist_to_block / max_clearance_px), 0.0, 1.0)

    cost = (
        w_dist
        + w_crater * crater_prob
        + w_slope * slope_norm
        + w_clearance * clearance_penalty
    ).astype(np.float32)

    cost[inflated_block] = blocked_cost

    meta = {
        "inflation_radius_px": int(inflation_radius_px),
        "crater_block_threshold": float(crater_block_threshold),
        "weights": {
            "w_dist": float(w_dist),
            "w_crater": float(w_crater),
            "w_slope": float(w_slope),
            "w_clearance": float(w_clearance),
        },
        "max_clearance_px": float(max_clearance_px),
        "blocked_cost": float(blocked_cost),
        "blocked_fraction": float(np.mean(inflated_block)),
        "cost_min": float(np.min(cost[np.isfinite(cost)])),
        "cost_max": float(np.max(cost[np.isfinite(cost)])),
    }

    return cost, inflated_block.astype(np.uint8), meta


def _resolve_inputs(region: str, crater_prob_path: Path | None, slope_path: Path | None) -> tuple[Path, Path]:
    cp = crater_prob_path if crater_prob_path is not None else MASK / f"mask_{region}.tif"
    sp = slope_path if slope_path is not None else SLOPE / f"slope_{region}.tif"
    if not cp.exists():
        raise FileNotFoundError(f"Crater probability/mask file not found: {cp}")
    if not sp.exists():
        raise FileNotFoundError(f"Slope file not found: {sp}")
    return cp, sp


def main() -> None:
    parser = argparse.ArgumentParser(description="Build traversability costmap from crater map + slope.")
    parser.add_argument("--region", type=str, default="80s", help="Region suffix, e.g. 80s, 85s")
    parser.add_argument("--crater-prob", type=Path, default=None, help="Optional crater probability GeoTIFF")
    parser.add_argument("--slope", type=Path, default=None, help="Optional slope GeoTIFF")

    parser.add_argument("--inflation-radius-px", type=int, default=6)
    parser.add_argument("--crater-block-threshold", type=float, default=0.5)

    parser.add_argument("--w-dist", type=float, default=1.0)
    parser.add_argument("--w-crater", type=float, default=8.0)
    parser.add_argument("--w-slope", type=float, default=3.0)
    parser.add_argument("--w-clearance", type=float, default=2.0)
    parser.add_argument("--max-clearance-px", type=float, default=24.0)
    parser.add_argument("--blocked-cost", type=float, default=1e6)
    args = parser.parse_args()

    crater_prob_file, slope_file = _resolve_inputs(args.region, args.crater_prob, args.slope)

    with rasterio.open(crater_prob_file) as cp_src, rasterio.open(slope_file) as sl_src:
        crater_prob = cp_src.read(1).astype(np.float32)
        slope = sl_src.read(1).astype(np.float32)

        if cp_src.shape != sl_src.shape:
            raise ValueError(f"Shape mismatch: crater={cp_src.shape}, slope={sl_src.shape}")

        if cp_src.transform != sl_src.transform:
            raise ValueError("Transform mismatch between crater and slope rasters.")

        cost, blocked, meta = build_costmap(
            crater_prob=crater_prob,
            slope=slope,
            inflation_radius_px=args.inflation_radius_px,
            crater_block_threshold=args.crater_block_threshold,
            w_dist=args.w_dist,
            w_crater=args.w_crater,
            w_slope=args.w_slope,
            w_clearance=args.w_clearance,
            max_clearance_px=args.max_clearance_px,
            blocked_cost=args.blocked_cost,
        )

        cost_profile = cp_src.profile.copy()
        cost_profile.update(dtype=rasterio.float32, count=1, nodata=None)

        blocked_profile = cp_src.profile.copy()
        blocked_profile.update(dtype=rasterio.uint8, count=1, nodata=0)

    cost_out = COSTMAP_DIR / f"costmap_{args.region}.tif"
    blocked_out = BLOCKED_DIR / f"blocked_{args.region}.tif"
    meta_out = META_DIR / f"costmap_{args.region}.json"

    with rasterio.open(cost_out, "w", **cost_profile) as dst:
        dst.write(cost.astype(np.float32), 1)

    with rasterio.open(blocked_out, "w", **blocked_profile) as dst:
        dst.write(blocked.astype(np.uint8), 1)

    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Costmap generation complete")
    print(f"Crater input: {crater_prob_file}")
    print(f"Slope input: {slope_file}")
    print(f"Costmap: {cost_out}")
    print(f"Blocked: {blocked_out}")
    print(f"Metadata: {meta_out}")
    print(f"Blocked fraction: {meta['blocked_fraction']:.4f}")


if __name__ == "__main__":
    main()
