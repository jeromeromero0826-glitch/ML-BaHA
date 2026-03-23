from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from PIL import Image


def hex_to_rgba(hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, alpha)


def build_color_lookup(hazard_config: Dict) -> Dict[int, Tuple[int, int, int, int]]:
    classes = hazard_config.get("classes", [])
    color_lookup = {}

    for cls in classes:
        code = int(cls["code"])
        color = cls.get("color", "#000000")

        if code == 0:
            color_lookup[code] = (0, 0, 0, 0)  # transparent for no hazard
        else:
            color_lookup[code] = hex_to_rgba(color, alpha=190)

    return color_lookup


def raster_bounds_to_latlon_bounds(bounds, src_crs) -> List[List[float]]:
    """
    Returns Leaflet-style bounds:
    [
        [south, west],
        [north, east]
    ]
    """
    if str(src_crs).upper() == "EPSG:4326":
        return [
            [bounds.bottom, bounds.left],
            [bounds.top, bounds.right],
        ]

    from rasterio.warp import transform_bounds

    left, bottom, right, top = transform_bounds(
        src_crs,
        "EPSG:4326",
        bounds.left,
        bounds.bottom,
        bounds.right,
        bounds.top,
    )

    return [
        [bottom, left],
        [top, right],
    ]


def render_hazard_png(
    hazard_raster_path: str,
    output_png_path: str,
    hazard_config: Dict,
) -> Dict:
    """
    Convert classified hazard raster to transparent PNG for web map overlay.
    Returns metadata including web map bounds.
    """
    hazard_raster_path = Path(hazard_raster_path)
    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    color_lookup = build_color_lookup(hazard_config)

    with rasterio.open(hazard_raster_path) as src:
        hazard = src.read(1)
        bounds = src.bounds
        crs = src.crs
        height, width = hazard.shape

    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    unique_codes = np.unique(hazard)

    for code in unique_codes:
        code_int = int(code)
        mask = hazard == code

        if code_int in color_lookup:
            rgba[mask] = color_lookup[code_int]
        else:
            rgba[mask] = (0, 0, 0, 0)

    image = Image.fromarray(rgba, mode="RGBA")
    image.save(output_png_path)

    leaflet_bounds = raster_bounds_to_latlon_bounds(bounds, crs)

    return {
        "png_path": str(output_png_path),
        "bounds": leaflet_bounds,
        "width": int(width),
        "height": int(height),
    }