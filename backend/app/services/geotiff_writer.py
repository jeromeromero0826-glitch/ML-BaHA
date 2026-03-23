import rasterio
import numpy as np


def reconstruct_raster(values, rows, cols, grid_shape, fill_value):
    raster = np.full(grid_shape, fill_value, dtype=values.dtype)
    raster[rows, cols] = values
    return raster


def clean_raster_metadata(metadata):
    cleaned = metadata.copy()

    keys_to_remove = [
        "blockxsize", "blockysize", "tiled", "compress", "interleave"
    ]
    for key in keys_to_remove:
        cleaned.pop(key, None)

    if "transform" in cleaned:
        cleaned["transform"] = rasterio.Affine(*cleaned["transform"])

    return cleaned


def save_geotiff(output_path, array, metadata, dtype, nodata):
    meta = metadata.copy()
    meta.update({
        "count": 1,
        "dtype": dtype,
        "nodata": nodata
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(array, 1)