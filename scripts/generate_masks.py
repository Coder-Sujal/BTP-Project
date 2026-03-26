from config import DEM, PROCESSED
import rasterio
import numpy as np
import pandas as pd
import cv2
from pyproj import Transformer
import os
os.environ["PROJ_IGNORE_CELESTIAL_BODY"] = "YES"

MASK_DIR = PROCESSED / "masks"
MASK_DIR.mkdir(parents=True, exist_ok=True)

CRATER_CSV = "../craters_filtered.csv"


def generate_mask(dem_path, crater_df, output_path):
    print(f"\nProcessing: {dem_path.name}")

    with rasterio.open(dem_path) as src:
        height, width = src.shape
        pixel_size = src.res[0]

        print("CRS:", src.crs)
        print("Resolution:", src.res)

        # 🔥 Create transformer (lat/lon → DEM CRS)
        transformer = Transformer.from_crs(
            "EPSG:4326", src.crs, always_xy=True
        )

        mask = np.zeros((height, width), dtype=np.uint8)

        valid = 0
        skipped = 0

        for _, row in crater_df.iterrows():
            lat = row["lat"]
            lon = row["lon"]
            diameter = row["diameter"]

            # 🔥 Convert coordinates
            try:
                x, y = transformer.transform(lon, lat)
                r, c = src.index(x, y)
            except:
                skipped += 1
                continue

            # Skip if outside DEM
            if r < 0 or c < 0 or r >= height or c >= width:
                skipped += 1
                continue

            # Compute radius
            radius = int((diameter / 2) / pixel_size)

            # 🔥 Filter unrealistic radii
            if radius < 3 or radius > 80:
                continue

            cv2.circle(mask, (c, r), radius, 1, -1)
            valid += 1

        print("Total craters:", len(crater_df))
        print("Valid craters drawn:", valid)
        print("Skipped:", skipped)

        # Save mask
        profile = src.profile.copy()
        profile.update({
            "dtype": rasterio.uint8,
            "nodata": 0,
            "count": 1
        })

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask, 1)

    print(f"✅ Mask saved: {output_path}")


if __name__ == "__main__":
    crater_df = pd.read_csv(CRATER_CSV)

    # 🔥 IMPORTANT: Longitude fix (0–360 → -180–180)
    crater_df["lon"] = crater_df["lon"].apply(
        lambda x: x - 360 if x > 180 else x
    )

    # 🔥 Diameter filtering
    crater_df = crater_df[
        (crater_df["diameter"] > 1000) &
        (crater_df["diameter"] < 20000)
    ]

    print("Final dataset size:", len(crater_df))

    for dem_path in DEM.glob("*.tif"):
        name = dem_path.stem.split("_")[1]
        output_path = MASK_DIR / f"mask_{name}.tif"

        generate_mask(dem_path, crater_df, output_path)