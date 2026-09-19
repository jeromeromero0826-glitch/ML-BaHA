"""
rainfall_features.py — rainfall inputs, their valid range, and the derived
features the model consumes.

The surrogate was trained on 109 HEC-RAS 2D events. Outside the range those
events covered, the model is extrapolating and its output is not evidence about
anything. The bounds below come straight from that training set, so the API can
say which of the two situations a request is in instead of returning a confident
hazard map for rainfall the model has never seen.
"""

# Observed range of the 109 training events (assets/data/filtered_dataset.pkl).
# Inside this band the model is interpolating.
TRAINING_RANGE = {
    "duration":   (7.0, 40.0),     # hours
    "depth":      (50.0, 839.0),   # mm
    "antecedent": (8.0, 206.0),    # mm
    "intensity":  (3.87, 41.25),   # mm/h, derived
}

# Outside this wider band the request is rejected. The gap between the two is
# where a prediction is still returned but flagged as extrapolation. The ceiling
# is roughly 1.5x the largest training event, which is already beyond any
# recorded storm in the Bicol River Basin.
ACCEPTED_RANGE = {
    "duration":   (0.5, 72.0),
    "depth":      (0.0, 1250.0),
    "antecedent": (0.0, 400.0),
    "intensity":  (0.0, 100.0),
}


def validate_inputs(duration: float, depth: float, antecedent: float) -> None:
    """Raise ValueError if the scenario is outside what the API will answer at all."""
    for name, value in (("duration", duration), ("depth", depth), ("antecedent", antecedent)):
        lo, hi = ACCEPTED_RANGE[name]
        if value != value:                       # NaN
            raise ValueError(f"{name.capitalize()} must be a number.")
        if value < lo or value > hi:
            unit = "hours" if name == "duration" else "mm"
            raise ValueError(
                f"{name.capitalize()} must be between {lo:g} and {hi:g} {unit}. "
                f"Received {value:g}."
            )
    if duration <= 0:
        raise ValueError("Duration must be greater than 0.")

    intensity = depth / duration
    lo, hi = ACCEPTED_RANGE["intensity"]
    if intensity > hi:
        raise ValueError(
            f"That is {intensity:.1f} mm/h, beyond the {hi:g} mm/h ceiling this model will answer for. "
            "Increase the duration or reduce the rainfall depth."
        )


def check_extrapolation(duration: float, depth: float, antecedent: float) -> list[str]:
    """
    Return a human-readable note for each input that sits outside the training
    range. An empty list means the scenario is inside the envelope the surrogate
    actually learned. Accepted but outside means the answer is still returned,
    labelled for what it is.
    """
    intensity = depth / duration if duration else 0.0
    checks = (
        ("Storm duration", duration,   "h",    *TRAINING_RANGE["duration"]),
        ("Rainfall depth", depth,      "mm",   *TRAINING_RANGE["depth"]),
        ("Antecedent rainfall", antecedent, "mm", *TRAINING_RANGE["antecedent"]),
        ("Rainfall intensity", intensity, "mm/h", *TRAINING_RANGE["intensity"]),
    )
    notes = []
    for label, value, unit, lo, hi in checks:
        if value < lo:
            notes.append(f"{label} {value:g} {unit} is below the trained range ({lo:g} to {hi:g} {unit}).")
        elif value > hi:
            notes.append(f"{label} {value:g} {unit} is above the trained range ({lo:g} to {hi:g} {unit}).")
    return notes


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
