import numpy as np
from aether.depth.base import BaseDepthEstimator
from aether.depth.astronomical import AstronomicalDepthEstimator
from aether.depth.midas import MiDaSDepthEstimator

class HybridDepthEstimator(BaseDepthEstimator):
    def __init__(
        self,
        model_type: str = "MiDaS_small",
        astro_weight: float = 0.4,
        radial_weight: float = 0.5,
        power: float = 1.4,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        self.astro_weight = astro_weight
        self.astro_estimator = AstronomicalDepthEstimator(
            radial_weight=radial_weight,
            power=power,
            device=device,
        )
        self.nn_estimator: BaseDepthEstimator | None = None
        try:
            self.nn_estimator = MiDaSDepthEstimator(model_type=model_type, device=device)
        except Exception:
            self.nn_estimator = None

    def estimate(self, image: np.ndarray) -> np.ndarray:
        astro_depth = self.astro_estimator.estimate(image)
        if self.nn_estimator is None:
            return astro_depth

        try:
            nn_depth = self.nn_estimator.estimate(image)
            blended = (1.0 - self.astro_weight) * nn_depth + self.astro_weight * astro_depth
            return self.normalize_depth(blended)
        except Exception:
            return astro_depth
