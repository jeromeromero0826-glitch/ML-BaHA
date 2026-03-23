import os
import json
import joblib
import numpy as np
import pandas as pd
import rasterio


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\Thesis\Python\Machine Learning\flood-hazard-app\backend"

FILTERED_DATASET_PATH = os.path.join(
    BASE_DIR, "assets", "data", "filtered_dataset.pkl"
)

DEM_PATH = os.path.join(
    BASE_DIR, "assets", "rasters", "Sipocot_DEM.tif"
)

SLOPE_PATH = os.path.join(
    BASE_DIR, "assets", "rasters", "Sipocot_Slope.tif"
)

OUTPUT_TABLE_PATH = os.path.join(
    BASE_DIR, "assets", "tables", "grid_cells.csv"
)

OUTPUT_METADATA_PATH = os.path.join(
    BASE_DIR, "assets", "tables", "grid_metadata.json"
)


# ============================================================
# HELPERS
# ============================================================

def ensure_directories():
    os.makedirs(os.path.dirname(OUTPUT_TABLE_PATH), exist_ok=True)


def sample_raster_values(raster_path, coords, label):
    print(f"\nSampling {label} from: {raster_path}")

    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"{label} raster not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        sampled = list(src.sample(coords))
        values = np.array([v[0] for v in sampled], dtype=np.float32)

        nodata = src.nodata
        if nodata is not None:
            values = np.where(values == nodata, np.nan, values)

        nan_count = np.isnan(values).sum()
        if nan_count > 0:
            print(f"{label}: {nan_count} nodata values found. Filling with median.")
            fill_value = np.nanmedian(values)
            values = np.where(np.isnan(values), fill_value, values)

    print(
        f"{label} stats -> "
        f"min: {np.min(values):.3f}, "
        f"max: {np.max(values):.3f}, "
        f"mean: {np.mean(values):.3f}"
    )

    return values.astype(np.float32)


def serialize_meta(meta):
    out = {}
    for key, value in meta.items():
        if key == "transform":
            out[key] = list(value)
        else:
            try:
                json.dumps(value)
                out[key] = value
            except TypeError:
                out[key] = str(value)
    return out


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("BUILDING STATIC GRID FEATURES")
    print("=" * 60)

    ensure_directories()

    if not os.path.exists(FILTERED_DATASET_PATH):
        raise FileNotFoundError(f"Filtered dataset not found: {FILTERED_DATASET_PATH}")

    print(f"\nLoading filtered dataset from:\n{FILTERED_DATASET_PATH}")
    dataset = joblib.load(FILTERED_DATASET_PATH)

    coords_filtered = dataset["coords_filtered"]
    cell_indices = dataset["cell_indices"]
    grid_shape = dataset["grid_shape"]
    meta = dataset["meta"]

    nrows, ncols = grid_shape

    print("\nLoaded dataset contents:")
    print(f"  Number of filtered cells: {len(cell_indices):,}")
    print(f"  Grid shape: {grid_shape}")
    print(f"  Coordinate shape: {coords_filtered.shape}")

    # Reconstruct row and col from flattened cell indices
    rows = cell_indices // ncols
    cols = cell_indices % ncols

    # Sample terrain rasters
    elevation = sample_raster_values(DEM_PATH, coords_filtered, "Elevation")
    slope = sample_raster_values(SLOPE_PATH, coords_filtered, "Slope")

    # Build grid feature table
    grid_df = pd.DataFrame({
        "cell_id": np.arange(len(cell_indices), dtype=np.int32),
        "cell_index": cell_indices.astype(np.int32),
        "row": rows.astype(np.int32),
        "col": cols.astype(np.int32),
        "x_coordinate": coords_filtered[:, 0].astype(np.float32),
        "y_coordinate": coords_filtered[:, 1].astype(np.float32),
        "elevation": elevation,
        "slope": slope
    })

    print("\nGrid feature table preview:")
    print(grid_df.head())

    # Save CSV
    grid_df.to_csv(OUTPUT_TABLE_PATH, index=False)
    print(f"\nGrid feature table saved to:\n{OUTPUT_TABLE_PATH}")

    # Save metadata JSON
    metadata = {
        "grid_shape": [int(nrows), int(ncols)],
        "n_filtered_cells": int(len(cell_indices)),
        "meta": serialize_meta(meta)
    }

    with open(OUTPUT_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Grid metadata saved to:\n{OUTPUT_METADATA_PATH}")

    print("\nDone.")
    print("=" * 60)


if __name__ == "__main__":
    main()