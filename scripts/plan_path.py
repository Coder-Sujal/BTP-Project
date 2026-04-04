from __future__ import annotations

import argparse
import csv
import heapq
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import xy
from scipy.ndimage import distance_transform_edt

from config import PROCESSED

PATH_DIR = PROCESSED / "path_planning"
COSTMAP_DIR = PATH_DIR / "costmaps"
BLOCKED_DIR = PATH_DIR / "blocked"
ROUTE_DIR = PATH_DIR / "routes"
META_DIR = PATH_DIR / "meta"

for d in (PATH_DIR, ROUTE_DIR, META_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class Node:
    f: float
    g: float
    r: int
    c: int


NEIGHBORS = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, 2 ** 0.5),
    (-1, 1, 2 ** 0.5),
    (1, -1, 2 ** 0.5),
    (1, 1, 2 ** 0.5),
]


def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def nearest_free(blocked: np.ndarray, r: int, c: int) -> tuple[int, int]:
    if not blocked[r, c]:
        return r, c

    free = ~blocked
    if not np.any(free):
        raise ValueError("No free cells available in map.")

    _, inds = distance_transform_edt(blocked, return_indices=True)
    rr = int(inds[0, r, c])
    cc = int(inds[1, r, c])
    return rr, cc


def astar(cost: np.ndarray, blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = cost.shape

    sr, sc = start
    gr, gc = goal

    g_score = np.full((h, w), np.inf, dtype=np.float64)
    parent_r = np.full((h, w), -1, dtype=np.int32)
    parent_c = np.full((h, w), -1, dtype=np.int32)
    closed = np.zeros((h, w), dtype=bool)

    g_score[sr, sc] = 0.0
    pq: list[tuple[float, float, int, int]] = []
    heapq.heappush(pq, (heuristic(start, goal), 0.0, sr, sc))

    while pq:
        f_cur, g_cur, r, c = heapq.heappop(pq)
        if closed[r, c]:
            continue
        closed[r, c] = True

        if (r, c) == (gr, gc):
            break

        for dr, dc, step_len in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= h or nc >= w:
                continue
            if blocked[nr, nc]:
                continue

            step_cost = step_len * 0.5 * (float(cost[r, c]) + float(cost[nr, nc]))
            tentative = g_cur + step_cost

            if tentative < g_score[nr, nc]:
                g_score[nr, nc] = tentative
                parent_r[nr, nc] = r
                parent_c[nr, nc] = c
                f_new = tentative + heuristic((nr, nc), goal)
                heapq.heappush(pq, (f_new, tentative, nr, nc))

    if parent_r[gr, gc] == -1 and (sr, sc) != (gr, gc):
        return []

    path: list[tuple[int, int]] = []
    r, c = gr, gc
    path.append((r, c))
    while (r, c) != (sr, sc):
        pr = int(parent_r[r, c])
        pc = int(parent_c[r, c])
        if pr < 0 or pc < 0:
            return []
        r, c = pr, pc
        path.append((r, c))
    path.reverse()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A* path planning on generated costmap.")
    parser.add_argument("--region", type=str, default="80s")
    parser.add_argument("--costmap", type=Path, default=None)
    parser.add_argument("--blocked", type=Path, default=None)

    parser.add_argument("--start-row", type=int, required=True)
    parser.add_argument("--start-col", type=int, required=True)
    parser.add_argument("--goal-row", type=int, required=True)
    parser.add_argument("--goal-col", type=int, required=True)

    parser.add_argument("--allow-nearest-free", action="store_true")
    args = parser.parse_args()

    costmap_file = args.costmap if args.costmap is not None else COSTMAP_DIR / f"costmap_{args.region}.tif"
    blocked_file = args.blocked if args.blocked is not None else BLOCKED_DIR / f"blocked_{args.region}.tif"

    if not costmap_file.exists():
        raise FileNotFoundError(f"Costmap not found: {costmap_file}")
    if not blocked_file.exists():
        raise FileNotFoundError(f"Blocked map not found: {blocked_file}")

    with rasterio.open(costmap_file) as csrc, rasterio.open(blocked_file) as bsrc:
        cost = csrc.read(1).astype(np.float32)
        blocked = bsrc.read(1).astype(bool)
        transform = csrc.transform
        crs = csrc.crs
        profile = csrc.profile.copy()

    if cost.shape != blocked.shape:
        raise ValueError("Costmap and blocked map shapes do not match.")

    h, w = cost.shape

    start = (args.start_row, args.start_col)
    goal = (args.goal_row, args.goal_col)

    for name, (r, c) in (("start", start), ("goal", goal)):
        if r < 0 or c < 0 or r >= h or c >= w:
            raise ValueError(f"{name} outside map bounds: {(r, c)} shape={cost.shape}")

    if args.allow_nearest_free:
        start = nearest_free(blocked, *start)
        goal = nearest_free(blocked, *goal)
    else:
        if blocked[start]:
            raise ValueError("Start is blocked. Re-run with --allow-nearest-free.")
        if blocked[goal]:
            raise ValueError("Goal is blocked. Re-run with --allow-nearest-free.")

    path = astar(cost=cost, blocked=blocked, start=start, goal=goal)
    if not path:
        raise RuntimeError("No path found.")

    route_mask = np.zeros_like(blocked, dtype=np.uint8)
    for r, c in path:
        route_mask[r, c] = 1

    route_tif = ROUTE_DIR / f"route_{args.region}.tif"
    route_csv = ROUTE_DIR / f"route_{args.region}.csv"
    route_meta = META_DIR / f"route_{args.region}.json"

    route_profile = profile.copy()
    route_profile.update(dtype=rasterio.uint8, count=1, nodata=0)

    with rasterio.open(route_tif, "w", **route_profile) as dst:
        dst.write(route_mask, 1)

    total_cost = 0.0
    if len(path) > 1:
        for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
            step_len = float(np.hypot(r1 - r0, c1 - c0))
            total_cost += step_len * 0.5 * (float(cost[r0, c0]) + float(cost[r1, c1]))

    with route_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "row", "col", "x", "y"])
        for idx, (r, c) in enumerate(path):
            x, y = xy(transform, r, c, offset="center")
            writer.writerow([idx, r, c, float(x), float(y)])

    meta = {
        "region": args.region,
        "costmap": str(costmap_file),
        "blocked": str(blocked_file),
        "route_tif": str(route_tif),
        "route_csv": str(route_csv),
        "crs": str(crs),
        "start": {"row": int(start[0]), "col": int(start[1])},
        "goal": {"row": int(goal[0]), "col": int(goal[1])},
        "num_points": int(len(path)),
        "path_pixel_length": float(sum(np.hypot(r1 - r0, c1 - c0) for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]))),
        "path_cost": float(total_cost),
    }
    route_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Path planning complete")
    print(f"Start: {start} Goal: {goal}")
    print(f"Path points: {len(path)}")
    print(f"Path cost: {total_cost:.3f}")
    print(f"Route raster: {route_tif}")
    print(f"Route CSV: {route_csv}")
    print(f"Route metadata: {route_meta}")


if __name__ == "__main__":
    main()
