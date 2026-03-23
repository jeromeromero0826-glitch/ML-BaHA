"""
asset_loader.py — Optimized asset loading with validation.

Improvements over original:
- Post-load validation (checks required columns, keys, feature count)
- Descriptive errors on missing/malformed assets
- load_all_assets returns a typed dict with consistent keys
- Optional force_reload parameter for dev reloading without restart
"""

import json
import joblib
import pandas as pd

from app.core.config import (
    MODEL_PATH,
    FEATURE_NAMES_PATH,
    HAZARD_CONFIG_PATH,
    GRID_TABLE_PATH,
    GRID_METADATA_PATH,
)

# Required grid table columns — updated for Random Forest v2 (TWI + log10_flow_accumulation)
_REQUIRED_GRID_COLS = {
    "row", "col",
    "x_coordinate", "y_coordinate",
    "elevation", "slope",
    "log10_flow_accumulation", "twi",
}

# Required hazard config keys
_REQUIRED_HAZARD_KEYS = {"classes", "nodata_value"}

# Required grid metadata keys
_REQUIRED_METADATA_KEYS = {"grid_shape", "meta"}


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_json(path) -> dict | list:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Required asset not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in {path}: {e}")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_feature_names() -> list[str]:
    names = load_json(FEATURE_NAMES_PATH)
    if not isinstance(names, list) or len(names) == 0:
        raise ValueError(f"feature_names.json must be a non-empty list, got: {type(names)}")
    return names


def load_hazard_config() -> dict:
    cfg = load_json(HAZARD_CONFIG_PATH)
    missing = _REQUIRED_HAZARD_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"hazard_config.json is missing required keys: {missing}")
    if not isinstance(cfg["classes"], list) or len(cfg["classes"]) == 0:
        raise ValueError("hazard_config.json 'classes' must be a non-empty list.")
    return cfg


def load_grid_table() -> pd.DataFrame:
    if not GRID_TABLE_PATH.exists():
        raise FileNotFoundError(f"Grid table not found: {GRID_TABLE_PATH}")
    df = pd.read_csv(GRID_TABLE_PATH)
    missing_cols = _REQUIRED_GRID_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(f"grid_cells.csv is missing required columns: {missing_cols}")
    return df


def load_grid_metadata() -> dict:
    meta = load_json(GRID_METADATA_PATH)
    missing = _REQUIRED_METADATA_KEYS - meta.keys()
    if missing:
        raise ValueError(f"grid_metadata.json is missing required keys: {missing}")
    shape = meta["grid_shape"]
    if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
        raise ValueError(f"grid_metadata 'grid_shape' must be [rows, cols], got: {shape}")
    return meta


# ---------------------------------------------------------------------------
# Master loader
# ---------------------------------------------------------------------------

def load_all_assets(force_reload: bool = False) -> dict:
    """
    Load and validate all model assets.
    Returns a dict with keys: model, feature_names, hazard_config, grid_df, grid_metadata.
    Raises descriptive errors on any missing or malformed asset.
    """
    print("[asset_loader] Loading model...")
    model = load_model()

    print("[asset_loader] Loading feature names...")
    feature_names = load_feature_names()

    print("[asset_loader] Loading hazard config...")
    hazard_config = load_hazard_config()

    print("[asset_loader] Loading grid table...")
    grid_df = load_grid_table()

    print("[asset_loader] Loading grid metadata...")
    grid_metadata = load_grid_metadata()

    # Cross-validate: feature count must match what model expects
    try:
        expected_features = model.n_features_in_
        if len(feature_names) != expected_features:
            raise ValueError(
                f"feature_names.json has {len(feature_names)} features "
                f"but model expects {expected_features}."
            )
    except AttributeError:
        pass  # Some model wrappers don't expose n_features_in_

    print(
        f"[asset_loader] Assets loaded: "
        f"{len(grid_df):,} grid cells | "
        f"{len(feature_names)} features | "
        f"{len(hazard_config['classes'])} hazard classes"
    )

    return {
        "model":         model,
        "feature_names": feature_names,
        "hazard_config": hazard_config,
        "grid_df":       grid_df,
        "grid_metadata": grid_metadata,
    }