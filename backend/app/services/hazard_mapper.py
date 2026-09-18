import numpy as np


def classify_hazard(depth_array, hazard_config):
    nodata_value = hazard_config.get("nodata_value", 255)
    hazard = np.full(depth_array.shape, nodata_value, dtype=np.uint8)

    # Zero or negative depth = No Hazard
    hazard[depth_array <= 0] = 0

    for cls in hazard_config["classes"]:
        code = cls["code"]

        if code == 0:
            continue

        dmin = cls["min_depth"]
        dmax = cls["max_depth"]

        if dmax >= 999:
            mask = depth_array > dmin
        else:
            mask = (depth_array > dmin) & (depth_array <= dmax)

        hazard[mask] = code

    # Safety net. Depths that fall in no class interval (for example a value
    # sitting exactly on a boundary, or below the lowest class minimum) would
    # otherwise keep the nodata code and silently vanish from the class counts,
    # leaving the totals short of the cell count. Any wet cell still unassigned
    # is folded into the lowest hazard class.
    lowest = min((c["code"] for c in hazard_config["classes"] if c["code"] != 0), default=1)
    stranded = (hazard == nodata_value) & (depth_array > 0)
    if stranded.any():
        hazard[stranded] = lowest

    return hazard


def convert_no_hazard_to_nodata(hazard_array, nodata_value):
    hazard_for_raster = hazard_array.copy()
    hazard_for_raster[hazard_for_raster == 0] = nodata_value
    return hazard_for_raster


def build_hazard_lookup(hazard_config):
    code_to_name = {
        c["code"]: c.get("name", f"H{c['code']}")
        for c in hazard_config["classes"]
    }
    code_to_label = {
        c["code"]: c["label"]
        for c in hazard_config["classes"]
    }
    return code_to_name, code_to_label