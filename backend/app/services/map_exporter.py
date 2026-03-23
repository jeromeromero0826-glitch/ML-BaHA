import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import rasterio
from PIL import Image


def hazard_to_rgba(hazard_array: np.ndarray) -> np.ndarray:
    """
    Convert hazard class raster to RGBA image array.

    Class mapping:
    0 = No Hazard -> transparent
    1 = H1 Very Low
    2 = H2 Low
    3 = H3 Medium
    4 = H4 High
    5 = H5 Extreme
    """
    h, w = hazard_array.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    color_map = {
        0: (0, 0, 0, 0),           # transparent
        1: (173, 216, 230, 180),   # light blue
        2: (255, 255, 0, 180),     # yellow
        3: (255, 165, 0, 180),     # orange
        4: (255, 69, 0, 180),      # orange-red
        5: (220, 20, 60, 180),     # crimson
    }

    for cls, color in color_map.items():
        rgba[hazard_array == cls] = color

    return rgba


def get_raster_bounds(raster_path: Path) -> Dict[str, float]:
    """
    Return Leaflet-friendly bounds from raster.
    """
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        return {
            "west": float(bounds.left),
            "south": float(bounds.bottom),
            "east": float(bounds.right),
            "north": float(bounds.top),
        }


def export_hazard_overlay(
    hazard_raster_path: Path,
    output_png_path: Path,
    output_bounds_path: Path,
) -> Dict[str, Any]:
    """
    Convert hazard raster TIFF to colored PNG overlay and save bounds JSON.
    """
    with rasterio.open(hazard_raster_path) as src:
        hazard = src.read(1)

    rgba = hazard_to_rgba(hazard)
    img = Image.fromarray(rgba, mode="RGBA")
    img.save(output_png_path)

    bounds = get_raster_bounds(hazard_raster_path)
    with open(output_bounds_path, "w", encoding="utf-8") as f:
        json.dump(bounds, f, indent=4)

    return {
        "overlay_png": str(output_png_path),
        "overlay_bounds_json": str(output_bounds_path),
        "bounds": bounds,
    }