import numpy as np
from scipy.ndimage import gaussian_filter
from aether.depth.base import BaseDepthEstimator
from aether.io.image import to_grayscale

class AstronomicalDepthEstimator(BaseDepthEstimator):
    def __init__(self, radial_weight: float = 0.5, power: float = 1.4, device: str = "cpu") -> None:
        super().__init__(device=device)
        self.radial_weight = radial_weight
        self.power = power

    def estimate(self, image: np.ndarray) -> np.ndarray:
        gray = to_grayscale(image)
        h, w = gray.shape

        threshold = np.percentile(gray, 70)
        mask = gray > threshold
        if np.any(mask):
            y_indices, x_indices = np.indices((h, w))
            total_mass = np.sum(gray[mask])
            cy = float(np.sum(y_indices[mask] * gray[mask]) / total_mass)
            cx = float(np.sum(x_indices[mask] * gray[mask]) / total_mass)
        else:
            cy, cx = h / 2.0, w / 2.0

        y_coords = (np.arange(h) - cy) / (h / 2.0)
        x_coords = (np.arange(w) - cx) / (w / 2.0)
        xx, yy = np.meshgrid(x_coords, y_coords)
        radius = np.sqrt(xx**2 + yy**2)

        max_radius = 1.2
        shell_depth = np.sqrt(np.maximum(0.0, max_radius**2 - radius**2))
        shell_depth = shell_depth / np.max(shell_depth)

        smooth_gas = gaussian_filter(gray, sigma=max(h, w) * 0.02)
        luminosity_depth = np.power(np.clip(smooth_gas, 0.0, 1.0), self.power)

        combined = (1.0 - self.radial_weight) * luminosity_depth + self.radial_weight * (shell_depth * luminosity_depth)
        
        detail_gas = gray - smooth_gas
        enhanced = combined + 0.3 * detail_gas

        return self.normalize_depth(enhanced)
