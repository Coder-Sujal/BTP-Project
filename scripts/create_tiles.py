from config import DEM, HILLSHADE, SOS, KAGGLE, PROCESSED
import numpy as np
import rasterio

TILE_SIZE = 256

X_DIR = KAGGLE / "X"
Y_DIR = KAGGLE / "Y"

X_DIR.mkdir(parents=True, exist_ok=True)
Y_DIR.mkdir(parents=True, exist_ok=True)


def create_tiles(dem_path, hillshade_path, sos_path, mask_path, prefix):
    print(f"\nProcessing: {prefix}")

    with rasterio.open(dem_path) as d, \
         rasterio.open(hillshade_path) as h, \
         rasterio.open(sos_path) as s, \
         rasterio.open(mask_path) as m:

        dem = d.read(1)
        hill = h.read(1)
        sos = s.read(1)
        mask = m.read(1)

    print("DEM shape:", dem.shape)
    print("Mask unique values:", np.unique(mask))

    X = np.stack([dem, hill, sos], axis=-1)

    h, w = dem.shape
    count = 0

    for i in range(0, h - TILE_SIZE + 1, TILE_SIZE):
        for j in range(0, w - TILE_SIZE + 1, TILE_SIZE):

            x_tile = X[i:i+TILE_SIZE, j:j+TILE_SIZE]
            y_tile = mask[i:i+TILE_SIZE, j:j+TILE_SIZE]

            if x_tile.shape[:2] != (TILE_SIZE, TILE_SIZE):
                continue

            np.save(X_DIR / f"{prefix}_{count}.npy", x_tile)
            np.save(Y_DIR / f"{prefix}_{count}.npy", y_tile)

            count += 1

    print(f"✅ Saved {count} tiles for {prefix}")


# 🚀 MAIN DRIVER
if __name__ == "__main__":

    dem_files = list(DEM.glob("*.tif"))

    print(f"Found {len(dem_files)} DEM files")

    for dem_path in dem_files:

        name = dem_path.stem  # e.g., dem_80s, dem_85

        hillshade_path = HILLSHADE / f"hillshade_{name.split('_')[1]}.tif"
        sos_path = SOS / f"sos_{name.split('_')[1]}.tif"
        mask_path = PROCESSED / "masks" / f"mask_{name.split('_')[1]}.tif"
        print(hillshade_path, sos_path, mask_path)

        # 🔍 Check if files exist
        if not hillshade_path.exists():
            print(f"❌ Missing hillshade for {name}")
            continue

        if not sos_path.exists():
            print(f"❌ Missing SOS for {name}")
            continue

        if not mask_path.exists():
            print(f"❌ Missing mask for {name}")
            continue

        create_tiles(
            dem_path,
            hillshade_path,
            sos_path,
            mask_path,
            prefix=name
        )