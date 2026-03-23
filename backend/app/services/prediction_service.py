"""
prediction_service.py — Optimized flood hazard prediction pipeline.

Improvements over original:
- LRU cache for repeated rainfall input combinations
- Vectorized numpy ops (no Python loops in critical path)
- Single-pass summary computation
- Separated pure-compute from I/O (save_outputs now decoupled)
- Scenario registry for history endpoint
- Cleaner error surface
- Random Forest v2 support: TWI + log10_flow_accumulation features (12 total)
"""

from datetime import datetime
from functools import lru_cache
from pathlib import Path
import json
import hashlib
import threading

import numpy as np
import pandas as pd

from app.core.config import (
    DEPTH_NODATA,
    DEFAULT_HAZARD_NODATA,
    OUTPUT_RASTERS_DIR,
    OUTPUT_CSV_DIR,
    OUTPUT_LOGS_DIR,
    OUTPUTS_DIR,
)
from app.services.rainfall_features import validate_inputs, compute_rainfall_features
from app.services.hazard_mapper import (
    classify_hazard,
    convert_no_hazard_to_nodata,
    build_hazard_lookup,
)
from app.services.geotiff_writer import (
    reconstruct_raster,
    clean_raster_metadata,
    save_geotiff,
)
from app.services.map_renderer import render_hazard_png

# ---------------------------------------------------------------------------
# In-memory scenario registry  (thread-safe append-only list)
# ---------------------------------------------------------------------------
_scenario_registry: list[dict] = []
_registry_lock = threading.Lock()


def get_scenario_history() -> list[dict]:
    """Return a copy of all completed scenario summaries (newest first)."""
    with _registry_lock:
        return list(reversed(_scenario_registry))


def _register_scenario(entry: dict) -> None:
    with _registry_lock:
        _scenario_registry.append(entry)


# ---------------------------------------------------------------------------
# Rainfall cache key
# ---------------------------------------------------------------------------

def _rainfall_cache_key(duration: float, depth: float, antecedent: float) -> str:
    """Stable string key for caching rainfall feature dicts."""
    return f"{duration:.4f}_{depth:.4f}_{antecedent:.4f}"


# Simple dict-based cache (avoids hashing large arrays).
_rainfall_cache: dict[str, dict] = {}


def _get_rainfall_features(duration: float, depth: float, antecedent: float) -> dict:
    key = _rainfall_cache_key(duration, depth, antecedent)
    if key not in _rainfall_cache:
        _rainfall_cache[key] = compute_rainfall_features(duration, depth, antecedent)
    return _rainfall_cache[key]


# ---------------------------------------------------------------------------
# Feature matrix construction  (fully vectorised, no Python loop)
# ---------------------------------------------------------------------------

def build_feature_dataframe(
    grid_df: pd.DataFrame,
    rainfall: dict,
    feature_names: list,
) -> pd.DataFrame:
    n = len(grid_df)
    scalar_cols = {
        "duration":         np.float32(rainfall["duration"]),
        "depth":            np.float32(rainfall["depth"]),
        "antecedent":       np.float32(rainfall["antecedent"]),
        "intensity":        np.float32(rainfall["intensity"]),
        "total_rain":       np.float32(rainfall["total_rain"]),
        "antecedent_ratio": np.float32(rainfall["antecedent_ratio"]),
    }
    # Build scalar columns with np.full once each
    data = {col: np.full(n, val, dtype=np.float32) for col, val in scalar_cols.items()}

    # Grid columns — cast once
    data["x_coordinate"] = grid_df["x_coordinate"].to_numpy(dtype=np.float32)
    data["y_coordinate"] = grid_df["y_coordinate"].to_numpy(dtype=np.float32)
    data["elevation"]    = grid_df["elevation"].to_numpy(dtype=np.float32)
    data["slope"]        = grid_df["slope"].to_numpy(dtype=np.float32)

    # TWI and log10 flow accumulation (added in Random Forest v2)
    if "log10_flow_accumulation" in grid_df.columns:
        data["log10_flow_accumulation"] = grid_df["log10_flow_accumulation"].to_numpy(dtype=np.float32)
    elif "log10_flow_accumulation" in feature_names:
        raise ValueError(
            "Model expects 'log10_flow_accumulation' but it is missing from grid_cells.csv. "
            "Re-export your grid with the TWI columns included."
        )

    if "twi" in grid_df.columns:
        data["twi"] = grid_df["twi"].to_numpy(dtype=np.float32)
    elif "twi" in feature_names:
        raise ValueError(
            "Model expects 'twi' but it is missing from grid_cells.csv. "
            "Re-export your grid with the TWI columns included."
        )

    return pd.DataFrame(data)[feature_names]


# ---------------------------------------------------------------------------
# Summary  (single-pass, avoids repeated boolean masking)
# ---------------------------------------------------------------------------

def build_summary(
    depth_array: np.ndarray,
    hazard_array: np.ndarray,
    hazard_config: dict,
    rainfall: dict,
) -> dict:
    flooded_mask   = depth_array > 0
    n_flooded      = int(flooded_mask.sum())
    n_no_hazard    = int((hazard_array == 0).sum())

    summary = {
        "timestamp":                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_rainfall":             rainfall,
        "n_cells":                    int(len(depth_array)),
        "n_flooded_cells":            n_flooded,
        "n_no_hazard_cells":          n_no_hazard,
        "max_depth_m":                float(depth_array.max()),
        "mean_depth_all_cells_m":     float(depth_array.mean()),
        "mean_depth_flooded_cells_m": float(depth_array[flooded_mask].mean()) if n_flooded else 0.0,
    }

    # Vectorised class count using np.bincount on uint8 codes
    codes       = hazard_array.astype(np.intp)
    max_code    = int(codes.max()) + 1 if len(codes) else 1
    bin_counts  = np.bincount(codes, minlength=max_code)

    class_stats = [
        {
            "code":       cls["code"],
            "name":       cls.get("name", f"H{cls['code']}"),
            "label":      cls["label"],
            "cell_count": int(bin_counts[cls["code"]]) if cls["code"] < len(bin_counts) else 0,
            "color":      cls.get("color"),
        }
        for cls in hazard_config["classes"]
    ]
    summary["hazard_class_counts"] = class_stats
    return summary


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def ensure_output_dirs() -> None:
    for d in (OUTPUT_RASTERS_DIR, OUTPUT_CSV_DIR, OUTPUT_LOGS_DIR, OUTPUTS_DIR / "maps"):
        d.mkdir(parents=True, exist_ok=True)


def _cleanup_old_outputs() -> None:
    """Delete all previous prediction output files, keeping only the latest."""
    patterns = [
        (OUTPUT_RASTERS_DIR, "*.tif"),
        (OUTPUT_CSV_DIR,     "*.csv"),
        (OUTPUT_LOGS_DIR,    "*.json"),
        (OUTPUTS_DIR / "maps", "*.png"),
    ]
    for folder, pattern in patterns:
        if folder.exists():
            for f in folder.glob(pattern):
                try:
                    f.unlink()
                except OSError:
                    pass  # skip if file is locked


def build_scenario_id(duration: float, depth: float, antecedent: float) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"D{duration:g}_R{depth:g}_A{antecedent:g}_{ts}"


# ---------------------------------------------------------------------------
# I/O — decoupled from compute so it can later move to a background task
# ---------------------------------------------------------------------------

def save_outputs(
    scenario_id: str,
    depth_raster: np.ndarray,
    hazard_raster: np.ndarray,
    hazard_nodata: int,
    output_df: pd.DataFrame,
    summary: dict,
    grid_metadata: dict,
    hazard_config: dict,
) -> dict:
    ensure_output_dirs()
    _cleanup_old_outputs()   # delete previous prediction files before saving new ones

    depth_raster_path  = OUTPUT_RASTERS_DIR / f"{scenario_id}_depth.tif"
    hazard_raster_path = OUTPUT_RASTERS_DIR / f"{scenario_id}_hazard.tif"
    overlay_png_path   = OUTPUTS_DIR / "maps" / f"{scenario_id}_hazard.png"
    csv_path           = OUTPUT_CSV_DIR  / f"{scenario_id}_predicted_cells.csv"
    summary_path       = OUTPUT_LOGS_DIR / f"{scenario_id}_summary.json"

    raster_meta = clean_raster_metadata(grid_metadata["meta"])

    save_geotiff(depth_raster_path,  depth_raster,  raster_meta, dtype="float32", nodata=DEPTH_NODATA)
    save_geotiff(hazard_raster_path, hazard_raster, raster_meta, dtype="uint8",   nodata=hazard_nodata)

    output_df.to_csv(csv_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    map_output = render_hazard_png(
        hazard_raster_path=str(hazard_raster_path),
        output_png_path=str(overlay_png_path),
        hazard_config=hazard_config,
    )

    return {
        "outputs": {
            "scenario_id":   scenario_id,
            "depth_raster":  f"/outputs/rasters/{depth_raster_path.name}",
            "hazard_raster": f"/outputs/rasters/{hazard_raster_path.name}",
            "prediction_csv":f"/outputs/csv/{csv_path.name}",
            "summary_json":  f"/outputs/logs/{summary_path.name}",
            "hazard_png":    f"/outputs/maps/{overlay_png_path.name}",
        },
        "map_outputs": {
            "hazard_png": f"/outputs/maps/{overlay_png_path.name}",
            "bounds":     map_output["bounds"],
            "width":      map_output["width"],
            "height":     map_output["height"],
        },
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_prediction(duration: float, depth: float, antecedent: float, assets: dict) -> dict:
    validate_inputs(duration, depth, antecedent)

    model         = assets["model"]
    feature_names = assets["feature_names"]
    hazard_config = assets["hazard_config"]
    grid_df       = assets["grid_df"]
    grid_metadata = assets["grid_metadata"]

    # --- Rainfall features (cached for repeated identical inputs) ---
    rainfall    = _get_rainfall_features(duration, depth, antecedent)
    feature_df  = build_feature_dataframe(grid_df, rainfall, feature_names)

    # --- Inference ---
    predicted_depth = model.predict(feature_df.values).astype(np.float32)
    np.clip(predicted_depth, 0, None, out=predicted_depth)          # in-place clip

    # --- Classification ---
    predicted_hazard = classify_hazard(predicted_depth, hazard_config)

    hazard_nodata = int(hazard_config.get("nodata_value", DEFAULT_HAZARD_NODATA))
    predicted_hazard_for_raster = convert_no_hazard_to_nodata(predicted_hazard, hazard_nodata)

    # --- Reconstruct rasters ---
    grid_shape  = tuple(grid_metadata["grid_shape"])
    rows        = grid_df["row"].to_numpy(dtype=int)
    cols        = grid_df["col"].to_numpy(dtype=int)

    depth_raster  = reconstruct_raster(predicted_depth,              rows, cols, grid_shape, fill_value=np.float32(DEPTH_NODATA))
    hazard_raster = reconstruct_raster(predicted_hazard_for_raster,  rows, cols, grid_shape, fill_value=np.uint8(hazard_nodata))

    # --- Build output dataframe ---
    code_to_name, code_to_label = build_hazard_lookup(hazard_config)
    output_df = grid_df.copy()
    output_df["predicted_depth_m"] = predicted_depth
    output_df["hazard_code"]       = predicted_hazard
    output_df["hazard_name"]       = output_df["hazard_code"].map(code_to_name)
    output_df["hazard_label"]      = output_df["hazard_code"].map(code_to_label)

    # --- Summary & scenario ---
    summary     = build_summary(predicted_depth, predicted_hazard, hazard_config, rainfall)
    scenario_id = build_scenario_id(duration, depth, antecedent)

    saved = save_outputs(
        scenario_id=scenario_id,
        depth_raster=depth_raster,
        hazard_raster=hazard_raster,
        hazard_nodata=hazard_nodata,
        output_df=output_df,
        summary=summary,
        grid_metadata=grid_metadata,
        hazard_config=hazard_config,
    )

    # --- Register in history ---
    _register_scenario({
        "scenario_id":   scenario_id,
        "timestamp":     summary["timestamp"],
        "rainfall":      rainfall,
        "summary":       summary,
        "outputs":       saved["outputs"],
        "map_outputs":   saved["map_outputs"],
    })

    return {
        "rainfall":           rainfall,
        "summary":            summary,
        "hazard_class_counts":summary["hazard_class_counts"],
        "outputs":            saved["outputs"],
        "map_outputs":        saved["map_outputs"],
    }