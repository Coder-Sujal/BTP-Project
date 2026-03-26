from config import MASK, RESULTS
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

# Create result directory if not exists
RESULTS.mkdir(parents=True, exist_ok=True)

def save_mask_images():
    mask_files = list(MASK.glob("*.tif"))

    print(f"Found {len(mask_files)} mask files")

    for mask_path in mask_files:
        print(f"Processing: {mask_path.name}")

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        # Output PNG path
        output_path = RESULTS / f"{mask_path.stem}.png"

        # Save image
        plt.figure(figsize=(6, 6))
        plt.imshow(mask, cmap="gray")
        plt.title(mask_path.stem)
        plt.axis("off")

        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        
        print(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    save_mask_images()