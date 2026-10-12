import numpy as np
import pyvista as pv
from aether.io.image import resize_image

class MeshReconstructor:
    def __init__(
        self,
        depth_scale: float = 40.0,
        resolution_scale: float = 0.5,
        smooth_iterations: int = 15,
        double_sided: bool = True,
    ) -> None:
        self.depth_scale = depth_scale
        self.resolution_scale = min(1.0, max(0.1, resolution_scale))
        self.smooth_iterations = smooth_iterations
        self.double_sided = double_sided

    def reconstruct(self, rgb: np.ndarray, depth: np.ndarray) -> pv.PolyData:
        h, w = depth.shape
        target_w = max(16, int(w * self.resolution_scale))
        target_h = max(16, int(h * self.resolution_scale))

        resized_rgb = resize_image(rgb, (target_w, target_h))
        resized_depth = resize_image(depth, (target_w, target_h))

        x_coords = np.linspace(-target_w / 2.0, target_w / 2.0, target_w, dtype=np.float32)
        y_coords = np.linspace(target_h / 2.0, -target_h / 2.0, target_h, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)

        z_front = (resized_depth - 0.5) * self.depth_scale
        points_front = np.column_stack((xx.ravel(), yy.ravel(), z_front.ravel()))
        colors_front = (resized_rgb.reshape(-1, 3) * 255.0).astype(np.uint8)

        grid_front = pv.StructuredGrid()
        grid_front.points = points_front
        grid_front.dimensions = [target_w, target_h, 1]
        mesh_front = grid_front.extract_surface(algorithm="dataset_surface")
        mesh_front.point_data["RGB"] = colors_front

        if not self.double_sided:
            if self.smooth_iterations > 0:
                mesh_front = mesh_front.smooth(n_iter=self.smooth_iterations)
            return mesh_front.compute_normals(point_normals=True, cell_normals=False)

        z_back = - (resized_depth - 0.5) * self.depth_scale
        points_back = np.column_stack((xx.ravel(), yy.ravel(), z_back.ravel()))
        colors_back = colors_front.copy()

        grid_back = pv.StructuredGrid()
        grid_back.points = points_back
        grid_back.dimensions = [target_w, target_h, 1]
        mesh_back = grid_back.extract_surface(algorithm="dataset_surface")
        mesh_back.point_data["RGB"] = colors_back

        combined = mesh_front + mesh_back
        if self.smooth_iterations > 0:
            combined = combined.smooth(n_iter=self.smooth_iterations)

        return combined.compute_normals(point_normals=True, cell_normals=False)
