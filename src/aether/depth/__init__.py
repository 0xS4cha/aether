from aether.depth.base import BaseDepthEstimator
from aether.depth.midas import MiDaSDepthEstimator
from aether.depth.astronomical import AstronomicalDepthEstimator
from aether.depth.ensemble import HybridDepthEstimator

def create_depth_estimator(
    model_type: str = "hybrid",
    device: str = "auto",
    astro_weight: float = 0.4,
    radial_weight: float = 0.5,
    power: float = 1.4,
) -> BaseDepthEstimator:
    normalized = model_type.lower()
    if normalized == "astronomical":
        return AstronomicalDepthEstimator(radial_weight=radial_weight, power=power, device=device)
    elif normalized in ("midas_small", "dpt_hybrid", "dpt_large"):
        model_name_map = {
            "midas_small": "MiDaS_small",
            "dpt_hybrid": "DPT_Hybrid",
            "dpt_large": "DPT_Large",
        }
        return MiDaSDepthEstimator(model_type=model_name_map[normalized], device=device)
    elif normalized == "hybrid":
        return HybridDepthEstimator(
            model_type="MiDaS_small",
            astro_weight=astro_weight,
            radial_weight=radial_weight,
            power=power,
            device=device,
        )
    else:
        raise ValueError(f"Unknown depth model type: {model_type}")

__all__ = [
    "BaseDepthEstimator",
    "MiDaSDepthEstimator",
    "AstronomicalDepthEstimator",
    "HybridDepthEstimator",
    "create_depth_estimator",
]
