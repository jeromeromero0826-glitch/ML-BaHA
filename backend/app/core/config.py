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

# How many recent scenarios keep their output files on disk. The history
# endpoint only ever lists scenarios whose files still exist, so this is both
# the disk budget and the length of the history panel.
MAX_RETAINED_SCENARIOS = 5

DEPTH_NODATA = -9999.0
DEFAULT_HAZARD_NODATA = 255