import numpy as np
import rasterio
from scipy.ndimage import sobel, gaussian_filter
from config import SLOPE, SOS

def compute_sos(input_path, output_path):
    with rasterio.open(input_path) as src:
        slope = src.read(1)
        profile = src.profile

    slope = slope.astype(np.float32)

    # Handle NoData
    slope[slope == -9999] = np.nan
    slope = np.nan_to_num(slope, nan=np.nanmean(slope))

    # Smooth
    slope = gaussian_filter(slope, sigma=1)

    dx = sobel(slope, axis=0)
    dy = sobel(slope, axis=1)

    sos = np.hypot(dx, dy)

    sos = gaussian_filter(sos, sigma=1)

    # Normalize
    sos = (sos - np.min(sos)) / (np.max(sos) - np.min(sos) + 1e-8)

    # Enhance
    sos = np.power(sos, 1.5)
    sos[sos < 0.1] = 0

    profile.update(dtype=rasterio.float32)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(sos.astype(np.float32), 1)


def main():
    for slope_file in SLOPE.glob("*.tif"):
        output_file = SOS / slope_file.name.replace("slope", "sos")
        print(f"Processing: {slope_file.name}")
        compute_sos(slope_file, output_file)


if __name__ == "__main__":
    main()