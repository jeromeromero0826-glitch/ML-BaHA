def validate_inputs(duration: float, depth: float, antecedent: float) -> None:
    if duration <= 0:
        raise ValueError("Duration must be greater than 0.")
    if depth < 0:
        raise ValueError("Rainfall depth cannot be negative.")
    if antecedent < 0:
        raise ValueError("Antecedent rainfall cannot be negative.")


def compute_rainfall_features(duration: float, depth: float, antecedent: float) -> dict:
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