from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Raw data
RAW_DATA = BASE_DIR / "raw_data" / "LOLA"

# Processed data
PROCESSED = BASE_DIR / "processed"
HILLSHADE = PROCESSED / "hillshade"
SLOPE = PROCESSED / "slope"
SOS = PROCESSED / "sos"
TILES = PROCESSED / "tiles"
DEM = PROCESSED / "dem"
DEM.mkdir(parents=True, exist_ok=True)
MASK = PROCESSED / "masks"
MASK.mkdir(parents=True, exist_ok=True)

# Results
RESULTS = BASE_DIR / "result_images" / "data"

# Models
MODELS = BASE_DIR / "models"

# Kaggle upload
KAGGLE = BASE_DIR / "kaggle_upload"

# Ensure directories exist
for path in [HILLSHADE, SLOPE, SOS, TILES, RESULTS, MODELS, KAGGLE]:
    path.mkdir(parents=True, exist_ok=True)