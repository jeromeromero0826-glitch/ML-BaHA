import os
import json
import joblib
import numpy as np
import pandas as pd
import rasterio
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\Thesis\Python\Machine Learning\flood-hazard-app\backend"

MODEL_PATH = os.path.join(BASE_DIR, "assets", "model", "xgboost_best_tuned_model.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "assets", "model", "feature_names.json")
HAZARD_CONFIG_PATH = os.path.join(BASE_DIR, "assets", "model", "hazard_config.json")

GRID_TABLE_PATH = os.path.join(BASE_DIR, "assets", "tables", "grid_cells.csv")
GRID_METADATA_PATH = os.path.join(BASE_DIR, "assets", "tables", "grid_metadata.json")

DEPTH_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "rasters", "predicted_depth.tif")
HAZARD_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "rasters", "predicted_hazard.tif")
CSV_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "csv", "predicted_cells.csv")
SUMMARY_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "logs", "prediction_summary.json")


# ============================================================
# USER INPUTS
# ============================================================

DURATION = 12.0
DEPTH = 180.0
ANTECEDENT = 60.0


# ============================================================
# HELPERS
# ============================================================

def ensure_output_dirs():
    os.makedirs(os.path.dirname(DEPTH_OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(HAZARD_OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SUMMARY_OUTPUT_PATH), exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_rainfall_features(duration, depth, antecedent):
    intensity = depth / (duration + 1e-6)
    total_rain = depth + antecedent
    antecedent_ratio = antecedent / (depth + 1e-6)

    return {
        "duration": float(duration),
        "depth": float(depth),
        "antecedent": float(antecedent),
        "intensity": float(intensity),
        "total_rain": float(total_rain),
        "antecedent_ratio": float(antecedent_ratio),
    }


def validate_inputs(duration, depth, antecedent):
    if duration <= 0:
        raise ValueError("Duration must be greater than 0.")
    if depth < 0:
        raise ValueError("Rainfall depth cannot be negative.")
    if antecedent < 0:
        raise ValueError("Antecedent rainfall cannot be negative.")


def classify_hazard(depth_array, hazard_config):
    nodata_value = hazard_config.get("nodata_value", 255)
    hazard = np.full(depth_array.shape, nodata_value, dtype=np.uint8)

    # Zero or negative depth = No Hazard
    hazard[depth_array <= 0] = 0

    for cls in hazard_config["classes"]:
        code = cls["code"]

        # Skip explicit No Hazard class because it was already assigned above
        if code == 0:
            continue

        dmin = cls["min_depth"]
        dmax = cls["max_depth"]

        # Open-ended last class (e.g., >5 m)
        if dmax >= 999:
            mask = depth_array > dmin
        else:
            mask = (depth_array > dmin) & (depth_array <= dmax)

        hazard[mask] = code

    return hazard


def reconstruct_raster(values, rows, cols, grid_shape, fill_value):
    raster = np.full(grid_shape, fill_value, dtype=values.dtype)
    raster[rows, cols] = values
    return raster


def build_summary(depth_array, hazard_array, hazard_config):
    flooded_mask = depth_array > 0
    no_hazard_mask = hazard_array == 0

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_cells": int(len(depth_array)),
        "n_flooded_cells": int(np.sum(flooded_mask)),
        "n_no_hazard_cells": int(np.sum(no_hazard_mask)),
        "max_depth_m": float(np.max(depth_array)),
        "mean_depth_all_cells_m": float(np.mean(depth_array)),
        "mean_depth_flooded_cells_m": float(np.mean(depth_array[flooded_mask])) if np.any(flooded_mask) else 0.0,
    }

    class_stats = []
    for cls in hazard_config["classes"]:
        code = cls["code"]
        name = cls.get("name", f"H{code}")
        label = cls["label"]
        count = int(np.sum(hazard_array == code))

        class_stats.append({
            "code": code,
            "name": name,
            "label": label,
            "cell_count": count
        })

    summary["hazard_class_counts"] = class_stats
    return summary


def save_geotiff(output_path, array, metadata, dtype, nodata):
    meta = metadata.copy()
    meta.update({
        "count": 1,
        "dtype": dtype,
        "nodata": nodata
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(array, 1)


def clean_raster_metadata(metadata):
    cleaned = metadata.copy()

    # Remove keys that may cause trouble when rewriting
    keys_to_remove = [
        "blockxsize", "blockysize", "tiled", "compress", "interleave"
    ]
    for key in keys_to_remove:
        cleaned.pop(key, None)

    return cleaned


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("RUN LOCAL FLOOD HAZARD PREDICTION")
    print("=" * 60)

    ensure_output_dirs()

    validate_inputs(DURATION, DEPTH, ANTECEDENT)

    print("\nLoading assets...")
    model = joblib.load(MODEL_PATH)
    feature_names = load_json(FEATURE_NAMES_PATH)
    hazard_config = load_json(HAZARD_CONFIG_PATH)
    grid_df = pd.read_csv(GRID_TABLE_PATH)
    grid_metadata = load_json(GRID_METADATA_PATH)

    print(f"Loaded model from: {MODEL_PATH}")
    print(f"Loaded grid table with {len(grid_df):,} cells")

    rainfall = compute_rainfall_features(DURATION, DEPTH, ANTECEDENT)

    print("\nComputed rainfall features:")
    for k, v in rainfall.items():
        print(f"  {k}: {v:.6f}")

    # Build model input
    feature_df = pd.DataFrame({
        "duration": np.full(len(grid_df), rainfall["duration"], dtype=np.float32),
        "depth": np.full(len(grid_df), rainfall["depth"], dtype=np.float32),
        "antecedent": np.full(len(grid_df), rainfall["antecedent"], dtype=np.float32),
        "intensity": np.full(len(grid_df), rainfall["intensity"], dtype=np.float32),
        "total_rain": np.full(len(grid_df), rainfall["total_rain"], dtype=np.float32),
        "antecedent_ratio": np.full(len(grid_df), rainfall["antecedent_ratio"], dtype=np.float32),
        "x_coordinate": grid_df["x_coordinate"].values.astype(np.float32),
        "y_coordinate": grid_df["y_coordinate"].values.astype(np.float32),
        "elevation": grid_df["elevation"].values.astype(np.float32),
        "slope": grid_df["slope"].values.astype(np.float32),
    })

    # Enforce exact feature order
    feature_df = feature_df[feature_names]

    print(f"\nFeature matrix shape: {feature_df.shape}")
    print("Predicting flood depth...")

    predicted_depth = model.predict(feature_df.values).astype(np.float32)

    # Clip negative predictions to zero
    predicted_depth = np.where(predicted_depth < 0, 0, predicted_depth).astype(np.float32)

    # Classify hazard
    predicted_hazard = classify_hazard(predicted_depth, hazard_config)

    # Prepare raster outputs
    grid_shape = tuple(grid_metadata["grid_shape"])
    rows = grid_df["row"].values.astype(int)
    cols = grid_df["col"].values.astype(int)

    hazard_nodata = np.uint8(hazard_config.get("nodata_value", 255))

    # For raster display only: make No Hazard transparent
    predicted_hazard_for_raster = predicted_hazard.copy()
    predicted_hazard_for_raster[predicted_hazard_for_raster == 0] = hazard_nodata

    depth_raster = reconstruct_raster(
        predicted_depth,
        rows,
        cols,
        grid_shape,
        fill_value=np.float32(-9999.0)
    )

    hazard_raster = reconstruct_raster(
        predicted_hazard_for_raster,
        rows,
        cols,
        grid_shape,
        fill_value=hazard_nodata
    )

    # Raster metadata
    raster_meta = grid_metadata["meta"]
    if "transform" in raster_meta:
        raster_meta["transform"] = rasterio.Affine(*raster_meta["transform"])

    raster_meta = clean_raster_metadata(raster_meta)

    # Save rasters
    save_geotiff(
        DEPTH_OUTPUT_PATH,
        depth_raster,
        raster_meta,
        dtype="float32",
        nodata=-9999.0
    )

    save_geotiff(
        HAZARD_OUTPUT_PATH,
        hazard_raster,
        raster_meta,
        dtype="uint8",
        nodata=int(hazard_nodata)
    )

    print(f"Saved depth raster to: {DEPTH_OUTPUT_PATH}")
    print(f"Saved hazard raster to: {HAZARD_OUTPUT_PATH}")

    # Save per-cell CSV
    output_df = grid_df.copy()
    output_df["predicted_depth_m"] = predicted_depth
    output_df["hazard_code"] = predicted_hazard

    code_to_name = {c["code"]: c.get("name", f"H{c['code']}") for c in hazard_config["classes"]}
    code_to_label = {c["code"]: c["label"] for c in hazard_config["classes"]}

    output_df["hazard_name"] = output_df["hazard_code"].map(code_to_name)
    output_df["hazard_label"] = output_df["hazard_code"].map(code_to_label)

    output_df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"Saved prediction table to: {CSV_OUTPUT_PATH}")

    # Save summary
    summary = build_summary(predicted_depth, predicted_hazard, hazard_config)
    summary["input_rainfall"] = rainfall

    with open(SUMMARY_OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Saved summary to: {SUMMARY_OUTPUT_PATH}")

    print("\nPrediction completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()