from config import DEM, HILLSHADE, SOS, TILES, KAGGLE
import numpy as np
import rasterio

TILE_SIZE = 256

X_DIR = KAGGLE / "X"
Y_DIR = KAGGLE / "Y"

X_DIR.mkdir(parents=True, exist_ok=True)
Y_DIR.mkdir(parents=True, exist_ok=True)


def create_tiles(dem_path, hillshade_path, sos_path, mask, prefix):
    with rasterio.open(dem_path) as d, \
         rasterio.open(hillshade_path) as h, \
         rasterio.open(sos_path) as s:

        dem = d.read(1)
        hill = h.read(1)
        sos = s.read(1)

    X = np.stack([dem, hill, sos], axis=-1)

    h, w = dem.shape
    count = 0

    for i in range(0, h, TILE_SIZE):
        for j in range(0, w, TILE_SIZE):

            x_tile = X[i:i+TILE_SIZE, j:j+TILE_SIZE]
            y_tile = mask[i:i+TILE_SIZE, j:j+TILE_SIZE]

            if x_tile.shape[:2] != (TILE_SIZE, TILE_SIZE):
                continue

            np.save(X_DIR / f"{prefix}_{count}.npy", x_tile)
            np.save(Y_DIR / f"{prefix}_{count}.npy", y_tile)

            count += 1

    print(f"Saved {count} tiles to kaggle_upload/")