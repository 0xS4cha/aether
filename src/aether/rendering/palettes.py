OPACITY_PRESETS = {
    "linear": [0.0, 0.1, 0.3, 0.6, 0.8, 1.0],
    "sigmoid": [0.0, 0.0, 0.05, 0.2, 0.6, 0.95],
    "exponential": [0.0, 0.0, 0.02, 0.1, 0.4, 1.0],
    "diffuse": [0.0, 0.05, 0.15, 0.35, 0.6, 0.85],
    "core": [0.0, 0.0, 0.0, 0.1, 0.5, 1.0],
}

CELESTIAL_COLORMAPS = [
    "plasma",
    "inferno",
    "magma",
    "viridis",
    "turbo",
    "gist_heat",
    "coolwarm",
]

def get_opacity_curve(name: str) -> list[float]:
    return OPACITY_PRESETS.get(name.lower(), OPACITY_PRESETS["sigmoid"])
