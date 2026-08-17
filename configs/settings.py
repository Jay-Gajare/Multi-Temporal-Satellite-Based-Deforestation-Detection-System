"""
Configuration for Deforestation Early Warning Pipeline.
Central place for all tunable parameters.
"""

import calendar
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / "exports"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# ── ROI: Rondônia, Brazil ──────────────────────────────────────────
# Hotspot of deforestation in the southern Amazon
ROI_COORDINATES = [
    [-63.5, -12.5],  # South-West  [lon, lat]
    [-59.5, -12.5],  # South-East
    [-59.5, -10.0],  # North-East
    [-63.5, -10.0],  # North-West
    [-63.5, -12.5],  # close ring
]
ROI_CENTER = [-61.5, -11.25]

# ── Temporal ────────────────────────────────────────────────────────
DATE_START = "2023-01-01"
DATE_END = "2023-12-31"
YEAR = 2023

# Pre-computed monthly ranges: {(month_num, "YYYY-MM-DD", "YYYY-MM-DD")}
MONTHLY_RANGES = []
for _m in range(1, 13):
    _last_day = calendar.monthrange(YEAR, _m)[1]
    MONTHLY_RANGES.append(
        (_m, f"{YEAR}-{_m:02d}-01", f"{YEAR}-{_m:02d}-{_last_day:02d}")
    )

# ── Spatial ─────────────────────────────────────────────────────────
SCALE = 30            # metres per pixel (Sentinel-2 analysis resolution)
PATCH_SIZE = 64       # pixels per side for model input patches
IMAGE_SIZE = 256      # crop size exported for labelling

# ── Sentinel-2 bands ────────────────────────────────────────────────
S2_BANDS = [
    "B2", "B3", "B4",       # Blue, Green, Red  (10 m)
    "B5", "B6", "B7",       # Red Edge 1-3      (20 m)
    "B8",                   # NIR                (10 m)
    "B8A",                  # NIR narrow         (20 m)
    "B11", "B12",           # SWIR 1, 2          (20 m)
]
S2_BAND_NAMES = [
    "blue", "green", "red",
    "re1", "re2", "re3",
    "nir", "nir_narrow",
    "swir1", "swir2",
]

# ── Hansen GFC bands ────────────────────────────────────────────────
HANSEN_BANDS = [
    "treecover2000",
    "lossyear",
    "loss",
    "gain",
    "datamask",
]

# ── Hansen dataset version ───────────────────────────────────────────
HANSEN_ASSET_ID = "UMD/hansen/global_forest_change_2025_v1_13"

# ── Cloud masking ────────────────────────────────────────────────────
CLOUD_PROB_THRESHOLD = 40   # s2cloudless probability threshold (0-100)
S2_CLOUD_PROB_COLLECTION = "COPERNICUS/S2_CLOUD_PROBABILITY"

# ── Feature engineering bands ────────────────────────────────────────
# Bands required for index computation (subset of S2_BANDS)
INDEX_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]

# All bands after feature engineering: 6 spectral + 3 indices
FEATURE_BANDS = INDEX_BANDS + ["NDVI", "NBR", "NDMI"]

# ── Seasonal definitions (Rondonia) ────────────────────────────────
DRY_SEASON_MONTHS = [4, 5, 6, 7, 8, 9]    # April – September
WET_SEASON_MONTHS = [10, 11, 12, 1, 2, 3]  # October – March

# ── Visualization ────────────────────────────────────────────────────
VIS_RGB_BANDS = ["B4", "B3", "B2"]           # True color
VIS_FALSE_COLOR_BANDS = ["B8", "B4", "B3"]   # NIR, Red, Green
VIS_MIN = 0.0
VIS_MAX = 0.4
VIS_GAMMA = 1.4

# ── Validation ────────────────────────────────────────────────────────
INDEX_VALID_MIN = -1.0
INDEX_VALID_MAX = 1.0

# ── Finalized Dataset ────────────────────────────────────────────────
# After export and balancing, 5,001 patches were selected for training.
# These are spatially stratified across rows 0-31 of the Rondonia grid.
# Always use FINAL_PATCH_IDS or FINAL_METADATA_PATH to determine which
# patches are in the dataset — never assume a count.
FINAL_METADATA_PATH = EXPORT_DIR / "final_patch_metadata.csv"
FINAL_PATCH_DIR = EXPORT_DIR / "patches"
FINAL_TOTAL_PATCHES = 5001  # updated by preprocessing/finalize_dataset.py
FINAL_PATCH_SIZE = 64       # pixels per side
FINAL_MONTHS = 12           # Jan–Dec 2023
FINAL_BANDS = FEATURE_BANDS  # 9 bands: B2,B3,B4,B8,B11,B12,NDVI,NBR,NDMI
