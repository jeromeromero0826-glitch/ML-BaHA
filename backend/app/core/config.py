from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = BACKEND_DIR / "app"

ASSETS_DIR = BACKEND_DIR / "assets"
MODEL_DIR = ASSETS_DIR / "model"
TABLES_DIR = ASSETS_DIR / "tables"

OUTPUTS_DIR = BACKEND_DIR / "outputs"
OUTPUT_RASTERS_DIR = OUTPUTS_DIR / "rasters"
OUTPUT_CSV_DIR = OUTPUTS_DIR / "csv"
OUTPUT_LOGS_DIR = OUTPUTS_DIR / "logs"

MODEL_PATH = MODEL_DIR / "xgboost_twi_v2_best_tuned_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"
HAZARD_CONFIG_PATH = MODEL_DIR / "hazard_config.json"

GRID_TABLE_PATH = TABLES_DIR / "grid_cells.csv"
GRID_METADATA_PATH = TABLES_DIR / "grid_metadata.json"

# Precomputed cell -> barangay assignment (see scripts/build_barangay_index.py)
BARANGAY_META_PATH = TABLES_DIR / "barangay_meta.json"
BARANGAY_INDEX_PATH = TABLES_DIR / "barangay_index.npz"

# How many recent scenarios keep their output files on disk. The history endpoint
# only ever lists scenarios whose files still exist, so this is the disk budget,
# the length of the history panel, and the depth of the prediction cache all at
# once: a repeat of any retained scenario is served without recomputing.
# Each scenario is roughly 8.6 MB on disk (depth raster 1.9, hazard raster 0.5,
# per-cell CSV 6.1, overlay PNG and summary JSON the remainder), so 8 is about
# 70 MB. Raise it for a higher cache hit rate if the instance has the disk.
MAX_RETAINED_SCENARIOS = 8

DEPTH_NODATA = -9999.0
DEFAULT_HAZARD_NODATA = 255