# 🌕 ChandraMarg – Lunar Crater Detection (LTKAU-Net)

An AI-driven pipeline for **lunar crater detection** using Digital Elevation Models (DEM) and deep learning (LTKAU-Net), built as part of an autonomous lunar navigation system.

---

## 📌 Project Overview

This project focuses on detecting lunar craters using **topographic data** instead of image-based approaches.

We use:
- **DEM (Digital Elevation Model)** → Elevation information
- **Hillshade** → Illumination-based terrain features
- **SOS (Slope-of-Slope)** → Curvature / second derivative of terrain

These are combined into a **3-channel input** for a deep learning segmentation model.

---

## 🚀 Pipeline Overview

```
RAW LOLA DATA (.img + .lbl)
        ↓
  Convert → DEM (.tif)
        ↓
  Generate → Hillshade / Slope
        ↓
  Compute → SOS
        ↓
  Generate → Masks (from crater dataset)
        ↓
  Create → Tiles (256×256)
        ↓
  Upload → Kaggle
        ↓
  Train → LTKAU-Net (GPU)
```

---

## 📁 Project Structure

```
project/
├── kaggle_upload/
│   ├── X/
│   └── Y/
├── models/
├── processed/
│   ├── dem/
│   ├── hillshade/
│   ├── slope/
│   ├── sos/
│   └── tiles/
├── raw_data/
│   └── LOLA/
├── result_images/
│   └── data/
├── scripts/
│   ├── config.py
│   ├── compute_sos.py
│   ├── create_tiles.py
│   ├── generate_masks.py
│   └── ...
└── README.md
```

> ⚠️ **Important:** Large datasets are ignored via `.gitignore`.

---

## ⚙️ Environment Setup

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd project
```

### 2. Install Dependencies

```bash
pip install numpy rasterio matplotlib scipy opencv-python
```

### 3. Install GDAL (Required)

```bash
sudo apt update
sudo apt install gdal-bin
```

Verify installation:

```bash
gdalinfo --version
```

---

## ☁️ GCP VM Setup (Recommended)

We use GCP VM for preprocessing large geospatial data.

- Create a VM instance
- Install Python + GDAL + rasterio
- Connect via SSH or VS Code Remote SSH

---

## 📥 Data Setup

### Download LOLA Dataset

**Source:** NASA LOLA DEM dataset

Place files in:

```
raw_data/LOLA/
```

Example files:

```
ldem_80s_40m.img
ldem_80s_40m.lbl
```

---

## 🔄 Data Processing Pipeline

### Step 1 — Convert IMG → GeoTIFF

```bash
gdal_translate raw_data/LOLA/ldem_80s_40m.lbl processed/dem/dem_80s.tif
```

### Step 2 — Generate Hillshade

```bash
gdaldem hillshade processed/dem/dem_80s.tif processed/hillshade/hillshade_80s.tif -az 180 -alt 5
```

### Step 3 — Generate Slope

```bash
gdaldem slope processed/dem/dem_80s.tif processed/slope/slope_80s.tif -p
```

### Step 4 — Compute SOS

```bash
python scripts/compute_sos.py
```

### Step 5 — Generate Masks *(Critical Step)*

- Input: crater dataset (lat, lon, diameter)
- Convert geographic coordinates → pixel coordinates
- Draw circular masks

Output format:

```
mask → (H, W)
```

### Step 6 — Create Tiles (256×256)

```bash
python scripts/create_tiles.py
```

Output:

```
kaggle_upload/
├── X/
└── Y/
```

---

## 🧠 Kaggle Setup (GPU Training)

### 1. Install Kaggle CLI

```bash
pip install kaggle
```

### 2. Add API Key

Download `kaggle.json` from your Kaggle account settings, then:

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Create Dataset

```bash
cd kaggle_upload
kaggle datasets init -p .
```

Edit the generated `dataset-metadata.json`:

```json
{
  "title": "lunar-crater-dataset",
  "id": "yourusername/lunar-crater-dataset",
  "licenses": [{"name": "CC0-1.0"}]
}
```

### 4. Upload Dataset

```bash
kaggle datasets create -p .
```

### 5. Train Model on Kaggle

1. Open a Kaggle Notebook
2. Enable GPU accelerator
3. Add your dataset
4. Train LTKAU-Net model

---

## 💾 Model Storage

After training, store model weights in:

```
models/
```

Example:

```
models/model.h5
```

---

## 🤝 Team Collaboration

### Git Workflow

```bash
git pull
git add .
git commit -m "your message"
git push
```

### Data Sharing

Since datasets are large, share via:

- Google Drive
- Kaggle datasets
- Shared GCP VM

---

## ⚠️ Important Guidelines

- Always use `scripts/config.py` for all file paths — never hardcode paths
- Ensure correct folder structure is maintained
- Normalize all data before training
- Verify mask alignment visually before training

---

## 🧪 Debug Tips

- Check file existence before processing
- Visualize DEM + mask overlay to confirm alignment
- Confirm lat/lon → pixel coordinate conversion accuracy

---

## 🔥 Future Work

- LTKAU-Net full implementation
- Hazard map generation
- Costmap creation
- Path planning algorithms:
  - A\*
  - D\*
  - RRT\*

---

## 📌 Technical Notes

- DEM-based learning improves robustness in shadow regions
- SOS highlights crater rims effectively
- Hillshade enhances edge detection

---

## 🙌 Contributors

| Name | Role |
|------|------|
| Sujal Chodvadiya | — |
| Atharv Mishra | — |
| Sujal Saraswat | — |
| Devendra Kumar | — |

---

## 📞 Support

If you face issues:

1. Verify folder structure matches the layout above
2. Check all paths in `config.py`
3. Ensure required raw data is available in `raw_data/LOLA/`

---

🚀 **Happy Building!**