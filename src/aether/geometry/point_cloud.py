import numpy as np
from aether.io.image import to_grayscale, filter_stars

class PointCloudReconstructor:
    def __init__(
        self,
        depth_scale: float = 50.0,
        subsample_step: int = 1,
        threshold: float = 0.05,
        remove_star_outliers: bool = True,
    ) -> None:
        self.depth_scale = depth_scale
        self.subsample_step = max(1, subsample_step)
        self.threshold = threshold
        self.remove_star_outliers = remove_star_outliers

    def reconstruct(self, rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.remove_star_outliers:
            processed_rgb, _ = filter_stars(rgb)
        else:
            processed_rgb = rgb

        gray = to_grayscale(processed_rgb)
        h, w = gray.shape

        y_grid, x_grid = np.mgrid[0:h:self.subsample_step, 0:w:self.subsample_step]
        sub_rgb = processed_rgb[::self.subsample_step, ::self.subsample_step]
        sub_depth = depth[::self.subsample_step, ::self.subsample_step]
        sub_gray = gray[::self.subsample_step, ::self.subsample_step]

        mask = (sub_gray > self.threshold) | (sub_depth > self.threshold)

        x_coords = (x_grid[mask].astype(np.float32) - (w / 2.0))
        y_coords = - (y_grid[mask].astype(np.float32) - (h / 2.0))
        z_coords = (sub_depth[mask] - 0.5) * self.depth_scale

        points = np.column_stack((x_coords, y_coords, z_coords)).astype(np.float32)
        colors = sub_rgb[mask].reshape(-1, 3).astype(np.float32)

        return points, colors
