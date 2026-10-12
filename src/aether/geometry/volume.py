import numpy as np
import pyvista as pv
from aether.io.image import to_grayscale, resize_image

class VolumeReconstructor:
    def __init__(
        self,
        depth_layers: int = 48,
        thickness_factor: float = 1.0,
        gaussian_sigma: float = 1.8,
        resolution_scale: float = 0.5,
    ) -> None:
        self.depth_layers = max(8, depth_layers)
        self.thickness_factor = thickness_factor
        self.gaussian_sigma = max(0.1, gaussian_sigma)
        self.resolution_scale = min(1.0, max(0.1, resolution_scale))

    def reconstruct(self, rgb: np.ndarray, depth: np.ndarray) -> pv.ImageData:
        h, w = depth.shape
        target_w = max(16, int(w * self.resolution_scale))
        target_h = max(16, int(h * self.resolution_scale))

        resized_rgb = resize_image(rgb, (target_w, target_h))
        resized_depth = resize_image(depth, (target_w, target_h))
        gray = to_grayscale(resized_rgb)

        d = self.depth_layers
        z_indices = np.linspace(0.0, 1.0, d, dtype=np.float32)
        z_grid = z_indices[:, None, None]

        target_centers = resized_depth[None, :, :]
        dist_sq = (z_grid - target_centers) ** 2
        
        sigma_sq = (self.gaussian_sigma / float(d)) ** 2
        density_3d = gray[None, :, :] * np.exp(-dist_sq / (2.0 * sigma_sq * self.thickness_factor))

        grid = pv.ImageData()
        grid.dimensions = np.array([target_w, target_h, d])
        grid.origin = (-target_w / 2.0, -target_h / 2.0, -d / 2.0)
        grid.spacing = (1.0, 1.0, 1.0)

        flattened = density_3d.transpose(2, 1, 0).flatten(order="F")
        grid.point_data["density"] = flattened.astype(np.float32)

        return grid
