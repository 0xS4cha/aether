import numpy as np
import pyvista as pv
from aether.spectral.decomposer import SpectralChannels
from aether.spectral.ionization import IonizationModel
from aether.io.image import resize_image

class SpectralReconstructor:
    def __init__(
        self,
        ionization_model: IonizationModel | None = None,
        depth_scale: float = 50.0,
        threshold: float = 0.08,
        subsample_step: int = 1,
    ) -> None:
        self.ionization_model = ionization_model or IonizationModel()
        self.depth_scale = depth_scale
        self.threshold = threshold
        self.subsample_step = max(1, subsample_step)

    def reconstruct_point_cloud(
        self,
        channels: SpectralChannels,
        depths: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        species_data = [
            ("oiii", channels.oiii, depths["oiii"], self.ionization_model.profile.oiii.color, 0),
            ("ha", channels.ha, depths["ha"], self.ionization_model.profile.ha.color, 1),
            ("sii", channels.sii, depths["sii"], self.ionization_model.profile.sii.color, 2),
        ]

        all_points: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []
        all_ids: list[np.ndarray] = []

        h, w = channels.ha.shape
        step = self.subsample_step
        y_grid, x_grid = np.mgrid[0:h:step, 0:w:step]

        for _, flux_map, depth_map, base_color, spec_id in species_data:
            sub_flux = flux_map[::step, ::step]
            sub_depth = depth_map[::step, ::step]

            mask = sub_flux > self.threshold
            if not np.any(mask):
                continue

            x_coords = (x_grid[mask].astype(np.float32) - (w / 2.0))
            y_coords = - (y_grid[mask].astype(np.float32) - (h / 2.0))
            z_coords = (sub_depth[mask] - 0.5) * self.depth_scale

            points = np.column_stack((x_coords, y_coords, z_coords)).astype(np.float32)

            f = sub_flux[mask][:, None]
            c = np.array(base_color, dtype=np.float32)[None, :]
            colors = np.clip(f * c * 1.3, 0.0, 1.0).astype(np.float32)

            ids = np.full(len(points), spec_id, dtype=np.int32)

            all_points.append(points)
            all_colors.append(colors)
            all_ids.append(ids)

        if not all_points:
            empty_pts = np.zeros((0, 3), dtype=np.float32)
            empty_clrs = np.zeros((0, 3), dtype=np.float32)
            empty_ids = np.zeros(0, dtype=np.int32)
            return empty_pts, empty_clrs, empty_ids

        return np.vstack(all_points), np.vstack(all_colors), np.concatenate(all_ids)

    def reconstruct_concentric_shells(
        self,
        channels: SpectralChannels,
        depths: dict[str, np.ndarray],
        resolution_scale: float = 0.5,
    ) -> dict[str, pv.PolyData]:
        species_data = [
            ("oiii", channels.oiii, depths["oiii"], self.ionization_model.profile.oiii.color),
            ("ha", channels.ha, depths["ha"], self.ionization_model.profile.ha.color),
            ("sii", channels.sii, depths["sii"], self.ionization_model.profile.sii.color),
        ]

        h, w = channels.ha.shape
        target_w = max(16, int(w * resolution_scale))
        target_h = max(16, int(h * resolution_scale))

        x_coords = np.linspace(-target_w / 2.0, target_w / 2.0, target_w, dtype=np.float32)
        y_coords = np.linspace(target_h / 2.0, -target_h / 2.0, target_h, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)

        shells: dict[str, pv.PolyData] = {}

        for name, flux_map, depth_map, base_color in species_data:
            res_flux = resize_image(flux_map, (target_w, target_h))
            res_depth = resize_image(depth_map, (target_w, target_h))

            z_front = (res_depth - 0.5) * self.depth_scale
            points_front = np.column_stack((xx.ravel(), yy.ravel(), z_front.ravel()))

            color_arr = np.array(base_color, dtype=np.float32)[None, :]
            flux_flat = res_flux.ravel()[:, None]
            rgb_front = np.clip(flux_flat * color_arr * 255.0, 0.0, 255.0).astype(np.uint8)

            grid_front = pv.StructuredGrid()
            grid_front.points = points_front
            grid_front.dimensions = [target_w, target_h, 1]
            mesh_front = grid_front.extract_surface(algorithm="dataset_surface")
            mesh_front.point_data["RGB"] = rgb_front

            z_back = - (res_depth - 0.5) * self.depth_scale
            points_back = np.column_stack((xx.ravel(), yy.ravel(), z_back.ravel()))
            grid_back = pv.StructuredGrid()
            grid_back.points = points_back
            grid_back.dimensions = [target_w, target_h, 1]
            mesh_back = grid_back.extract_surface(algorithm="dataset_surface")
            mesh_back.point_data["RGB"] = rgb_front.copy()

            combined = (mesh_front + mesh_back).smooth(n_iter=10)
            shells[name] = combined.compute_normals(point_normals=True, cell_normals=False)

        return shells

    def reconstruct_spectral_volume(
        self,
        channels: SpectralChannels,
        depths: dict[str, np.ndarray],
        depth_layers: int = 48,
        resolution_scale: float = 0.5,
    ) -> pv.ImageData:
        h, w = channels.ha.shape
        target_w = max(16, int(w * resolution_scale))
        target_h = max(16, int(h * resolution_scale))
        d = max(8, depth_layers)

        res_oiii_flux = resize_image(channels.oiii, (target_w, target_h))
        res_ha_flux = resize_image(channels.ha, (target_w, target_h))
        res_sii_flux = resize_image(channels.sii, (target_w, target_h))

        res_oiii_depth = resize_image(depths["oiii"], (target_w, target_h))
        res_ha_depth = resize_image(depths["ha"], (target_w, target_h))
        res_sii_depth = resize_image(depths["sii"], (target_w, target_h))

        z_indices = np.linspace(0.0, 1.0, d, dtype=np.float32)[:, None, None]

        sigma_sq = (1.6 / float(d)) ** 2
        oiii_dist_sq = (z_indices - res_oiii_depth[None, :, :]) ** 2
        ha_dist_sq = (z_indices - res_ha_depth[None, :, :]) ** 2
        sii_dist_sq = (z_indices - res_sii_depth[None, :, :]) ** 2

        vol_oiii = res_oiii_flux[None, :, :] * np.exp(-oiii_dist_sq / (2.0 * sigma_sq))
        vol_ha = res_ha_flux[None, :, :] * np.exp(-ha_dist_sq / (2.0 * sigma_sq))
        vol_sii = res_sii_flux[None, :, :] * np.exp(-sii_dist_sq / (2.0 * sigma_sq))

        total_density = vol_oiii + vol_ha + vol_sii

        color_oiii = np.array(self.ionization_model.profile.oiii.color, dtype=np.float32)
        color_ha = np.array(self.ionization_model.profile.ha.color, dtype=np.float32)
        color_sii = np.array(self.ionization_model.profile.sii.color, dtype=np.float32)

        rgb_3d = (
            vol_oiii[..., None] * color_oiii
            + vol_ha[..., None] * color_ha
            + vol_sii[..., None] * color_sii
        )
        rgb_3d = np.clip(rgb_3d * 255.0, 0.0, 255.0).astype(np.uint8)

        grid = pv.ImageData()
        grid.dimensions = np.array([target_w, target_h, d])
        grid.origin = (-target_w / 2.0, -target_h / 2.0, -d / 2.0)
        grid.spacing = (1.0, 1.0, 1.0)

        flattened_density = total_density.transpose(2, 1, 0).flatten(order="F")
        grid.point_data["density"] = flattened_density.astype(np.float32)

        flattened_rgb = rgb_3d.transpose(2, 1, 0, 3).reshape(-1, 3, order="F")
        grid.point_data["RGB"] = flattened_rgb

        return grid
