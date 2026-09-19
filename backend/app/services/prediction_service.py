"""
prediction_service.py — Optimized flood hazard prediction pipeline.

Improvements over original:
- LRU cache for repeated rainfall input combinations
- Vectorized numpy ops (no Python loops in critical path)
- Single-pass summary computation
- Separated pure-compute from I/O (save_outputs now decoupled)
- Scenario registry for history endpoint
- Cleaner error surface
- Model v2 support: TWI + log10_flow_accumulation features (12 total)
- Per-barangay summary computed server-side (was a 23 MB client-side download)
"""

from datetime import datetime
from pathlib import Path
import json
import threading
import uuid

import numpy as np
import pandas as pd

from app.core.config import (
    DEPTH_NODATA,
    DEFAULT_HAZARD_NODATA,
    MAX_RETAINED_SCENARIOS,
    OUTPUT_RASTERS_DIR,
    OUTPUT_CSV_DIR,
    OUTPUT_LOGS_DIR,
    OUTPUTS_DIR,
)
from app.services.rainfall_features import (
    validate_inputs,
    check_extrapolation,
    compute_rainfall_features,
)
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
from app.services.hover_grid import save_hover_grid
from app.services.barangay_summary import build_barangay_summary, build_area_totals

# ---------------------------------------------------------------------------
# In-memory scenario registry  (thread-safe append-only list)
# ---------------------------------------------------------------------------
_scenario_registry: list[dict] = []
_registry_lock = threading.Lock()


def get_scenario_history() -> list[dict]:
    """Return a copy of all retained scenario summaries (newest first)."""
    with _registry_lock:
        return list(reversed(_scenario_registry))


def _scenario_files(entry: dict) -> list[Path]:
    """Map a scenario's public output URLs back to their paths on disk."""
    paths = []
    for url in (entry.get("outputs") or {}).values():
        if isinstance(url, str) and url.startswith("/outputs/"):
            paths.append(OUTPUTS_DIR / url[len("/outputs/"):])
    return paths


def _find_cached_scenario(duration: float, depth: float, antecedent: float) -> dict | None:
    """
    Return a retained scenario computed from exactly these inputs, or None.

    A prediction is deterministic in its three rainfall inputs, so repeating a
    scenario recomputes an identical answer. On a small instance that is roughly
    26 seconds of CPU for a result already sitting on disk.

    The cache is the scenario registry rather than a separate structure, which
    makes a whole class of bug impossible: eviction removes the registry entry
    and deletes its files together under one lock, so there is no way to serve a
    hit whose outputs have already been deleted. The overlay is still checked on
    disk before serving, in case a file was removed from outside the process.
    """
    key = _rainfall_cache_key(duration, depth, antecedent)
    with _registry_lock:
        for entry in reversed(_scenario_registry):      # newest first
            r = entry.get("rainfall") or {}
            if _rainfall_cache_key(r.get("duration", -1),
                                   r.get("depth", -1),
                                   r.get("antecedent", -1)) != key:
                continue
            overlay = (entry.get("map_outputs") or {}).get("hazard_png", "")
            if overlay.startswith("/outputs/") and not (OUTPUTS_DIR / overlay[len("/outputs/"):]).exists():
                break       # files are gone; fall through and recompute
            return entry
    return None


def scenario_is_cached(duration: float, depth: float, antecedent: float) -> bool:
    """
    Whether this scenario would be served from cache. Used by the rate limiter so
    that repeating a scenario someone else already ran costs no compute budget.
    Side-effect free.
    """
    return _find_cached_scenario(duration, depth, antecedent) is not None


def _register_scenario(entry: dict) -> None:
    """
    Record a scenario and retire the oldest ones past the retention cap.

    Output files used to be wiped wholesale at the start of every prediction while
    the history endpoint went on advertising the deleted ones, so loading any
    scenario but the newest gave a 404 map and dead download links. Deletion is now
    tied to eviction from the registry: whatever history lists, you can still open.
    """
    evicted = []
    with _registry_lock:
        _scenario_registry.append(entry)
        while len(_scenario_registry) > MAX_RETAINED_SCENARIOS:
            evicted.append(_scenario_registry.pop(0))

    for old in evicted:
        for path in _scenario_files(old):
            try:
                path.unlink()
            except OSError:
                pass  # already gone, or held open by a download in flight


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

    # TWI and log10 flow accumulation (added in model v2)
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
    for d in (OUTPUT_RASTERS_DIR, OUTPUT_CSV_DIR, OUTPUT_LOGS_DIR,
              OUTPUTS_DIR / "maps", OUTPUTS_DIR / "grids"):
        d.mkdir(parents=True, exist_ok=True)


def purge_output_dirs() -> int:
    """
    Clear every output file. Called once at startup, when the in-memory registry
    is empty and therefore nothing can reference what is left over from a previous
    process. Keeps disk from growing across restarts on a small instance.
    """
    ensure_output_dirs()
    removed = 0
    for folder, pattern in (
        (OUTPUT_RASTERS_DIR, "*.tif"),
        (OUTPUT_CSV_DIR,     "*.csv"),
        (OUTPUT_LOGS_DIR,    "*.json"),
        (OUTPUTS_DIR / "maps",  "*.png"),
        (OUTPUTS_DIR / "grids", "*.bin"),
    ):
        if folder.exists():
            for f in folder.glob(pattern):
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
    return removed


def build_scenario_id(duration: float, depth: float, antecedent: float) -> str:
    """
    Unique per request. The timestamp alone has one-second resolution, so two
    predictions with the same inputs arriving together would have written to the
    same filenames and corrupted each other's outputs. The short suffix keeps
    concurrent requests on disjoint files.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    token = uuid.uuid4().hex[:4]
    return f"D{duration:g}_R{depth:g}_A{antecedent:g}_{ts}_{token}"


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

    depth_raster_path  = OUTPUT_RASTERS_DIR / f"{scenario_id}_depth.tif"
    hazard_raster_path = OUTPUT_RASTERS_DIR / f"{scenario_id}_hazard.tif"
    overlay_png_path   = OUTPUTS_DIR / "maps" / f"{scenario_id}_hazard.png"
    csv_path           = OUTPUT_CSV_DIR  / f"{scenario_id}_predicted_cells.csv"
    summary_path       = OUTPUT_LOGS_DIR / f"{scenario_id}_summary.json"
    hover_grid_path    = OUTPUTS_DIR / "grids" / f"{scenario_id}_hover.bin"

    raster_meta = clean_raster_metadata(grid_metadata["meta"])

    save_geotiff(depth_raster_path,  depth_raster,  raster_meta, dtype="float32", nodata=DEPTH_NODATA)
    save_geotiff(hazard_raster_path, hazard_raster, raster_meta, dtype="uint8",   nodata=hazard_nodata)

    output_df.to_csv(csv_path, index=False)

    # Packed per-cell grid for the map hover readout. Two orders of magnitude
    # smaller than shipping the CSV to the browser for the same purpose.
    save_hover_grid(
        path=hover_grid_path,
        depth_raster=depth_raster,
        hazard_raster=hazard_raster,
        hazard_nodata=hazard_nodata,
        depth_nodata_in=DEPTH_NODATA,
    )

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
            "hover_grid":    f"/outputs/grids/{hover_grid_path.name}",
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

    # Accepted, but is it inside what the surrogate was trained on? Recomputed
    # per request rather than stored, since it depends only on the three inputs.
    extrapolation = check_extrapolation(duration, depth, antecedent)

    # Identical inputs give an identical answer, so reuse one we still hold.
    hit = _find_cached_scenario(duration, depth, antecedent)
    if hit is not None:
        return {
            "extrapolation":       extrapolation,
            "rainfall":            hit["rainfall"],
            "summary":             hit["summary"],
            "hazard_class_counts": hit["summary"]["hazard_class_counts"],
            "barangay_summary":    hit.get("barangay_summary", []),
            "outputs":             hit["outputs"],
            "map_outputs":         hit["map_outputs"],
            "cached":              True,
        }

    model         = assets["model"]
    feature_names = assets["feature_names"]
    hazard_config = assets["hazard_config"]
    grid_df       = assets["grid_df"]
    grid_metadata = assets["grid_metadata"]
    barangay      = assets["barangay"]

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

    # --- Per-barangay aggregation (milliseconds; the index is precomputed) ---
    barangay_summary = build_barangay_summary(predicted_depth, predicted_hazard, barangay)
    area_totals      = build_area_totals(predicted_hazard, barangay)
    summary["area_totals"] = area_totals
    summary["channel_min_flow_acc_log10"] = barangay.get("channel_threshold")
    summary["n_channel_cells"] = barangay.get("n_channel_cells", 0)

    # Municipality-wide land-only maximum, so the headline number is not an
    # in-channel water depth. Channel cells are still in every count above.
    land_mask = ~barangay["cell_is_channel"]
    summary["max_depth_land_m"] = (
        float(predicted_depth[land_mask].max()) if land_mask.any() else 0.0
    )

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
        "scenario_id":      scenario_id,
        "timestamp":        summary["timestamp"],
        "rainfall":         rainfall,
        "summary":          summary,
        "barangay_summary": barangay_summary,
        "outputs":          saved["outputs"],
        "map_outputs":      saved["map_outputs"],
    })

    return {
        "extrapolation":      extrapolation,
        "rainfall":           rainfall,
        "summary":            summary,
        "hazard_class_counts":summary["hazard_class_counts"],
        "barangay_summary":   barangay_summary,
        "outputs":            saved["outputs"],
        "map_outputs":        saved["map_outputs"],
        "cached":             False,
    }