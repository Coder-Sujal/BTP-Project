from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import binary_dilation

from config import DEM, PROCESSED, RESULTS

PATH_DIR = PROCESSED / "path_planning"
BLOCKED_DIR = PATH_DIR / "blocked"
ROUTE_DIR = PATH_DIR / "routes"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize planned route over DEM and blocked map.")
    parser.add_argument("--region", type=str, default="80s")
    parser.add_argument("--dem", type=Path, default=None)
    parser.add_argument("--blocked", type=Path, default=None)
    parser.add_argument("--route", type=Path, default=None)
    parser.add_argument("--route-csv", type=Path, default=None)
    parser.add_argument("--downsample", type=int, default=8, help="Integer factor for fast plotting")
    parser.add_argument("--route-dilate", type=int, default=2, help="Dilate route pixels before downsampling")
    args = parser.parse_args()

    dem_file = args.dem if args.dem is not None else DEM / f"dem_{args.region}.tif"
    blocked_file = args.blocked if args.blocked is not None else BLOCKED_DIR / f"blocked_{args.region}.tif"
    route_file = args.route if args.route is not None else ROUTE_DIR / f"route_{args.region}.tif"
    route_csv_file = args.route_csv if args.route_csv is not None else ROUTE_DIR / f"route_{args.region}.csv"

    if not dem_file.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_file}")
    if not blocked_file.exists():
        raise FileNotFoundError(f"Blocked file not found: {blocked_file}")
    if not route_file.exists():
        raise FileNotFoundError(f"Route file not found: {route_file}")

    with rasterio.open(dem_file) as dsrc, rasterio.open(blocked_file) as bsrc, rasterio.open(route_file) as rsrc:
        dem = dsrc.read(1)
        blocked = bsrc.read(1).astype(bool)
        route = rsrc.read(1).astype(bool)

    if dem.shape != blocked.shape or dem.shape != route.shape:
        raise ValueError("DEM, blocked, and route rasters must have identical shape.")

    ds = max(1, int(args.downsample))
    if args.route_dilate > 0:
        route = binary_dilation(route, iterations=int(args.route_dilate))

    dem_v = dem[::ds, ::ds]
    blocked_v = blocked[::ds, ::ds]
    route_v = route[::ds, ::ds]

    dem_v = dem_v.astype(np.float32)
    dem_v = (dem_v - np.nanmin(dem_v)) / (np.nanmax(dem_v) - np.nanmin(dem_v) + 1e-8)

    plt.figure(figsize=(10, 10))
    plt.imshow(dem_v, cmap="gray")

    blocked_overlay = np.zeros((*blocked_v.shape, 4), dtype=np.float32)
    blocked_overlay[..., 0] = 1.0
    blocked_overlay[..., 3] = blocked_v.astype(np.float32) * 0.25
    plt.imshow(blocked_overlay)

    route_overlay = np.zeros((*route_v.shape, 4), dtype=np.float32)
    route_overlay[..., 1] = 1.0
    route_overlay[..., 2] = 0.2
    route_overlay[..., 3] = route_v.astype(np.float32) * 0.95
    plt.imshow(route_overlay)

    if route_csv_file.exists():
        rows = []
        with route_csv_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if rows:
            srow = float(rows[0]["row"]) / ds
            scol = float(rows[0]["col"]) / ds
            grow = float(rows[-1]["row"]) / ds
            gcol = float(rows[-1]["col"]) / ds
            plt.scatter([scol], [srow], c="cyan", s=30, marker="o", label="start")
            plt.scatter([gcol], [grow], c="yellow", s=30, marker="x", label="goal")
            plt.legend(loc="lower right")

    plt.title(f"Route Overlay ({args.region})")
    plt.axis("off")

    out = RESULTS / f"route_overlay_{args.region}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close()

    print("Route visualization saved")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
