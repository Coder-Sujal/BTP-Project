import rasterio
import numpy as np
from config import MASK

mask_files = list(MASK.glob("*.tif"))

for mask_path in mask_files:
    with rasterio.open(mask_path) as m:
        mask = m.read(1)
        print(f"{mask_path.name} - Unique values: {np.unique(mask)}")
