import numpy as np
import pyvista as pv
from scipy.ndimage import label, median_filter
from aether.io.image import resize_image, to_grayscale, filter_stars

class MeshReconstructor:
    def __init__(
        self,
        depth_scale: float = 40.0,
        resolution_scale: float = 0.5,
        smooth_iterations: int = 15,
        double_sided: bool = True,
        threshold: float = 0.03,
        remove_star_outliers: bool = True,
        close_mesh: bool = True,
        taper_edges: bool = True,
    ) -> None:
        self.depth_scale = depth_scale
        self.resolution_scale = min(1.0, max(0.1, resolution_scale))
        self.smooth_iterations = smooth_iterations
        self.double_sided = double_sided
        self.threshold = max(0.0, threshold)
        self.remove_star_outliers = remove_star_outliers
        self.close_mesh = close_mesh
        self.taper_edges = taper_edges

    def reconstruct(self, rgb: np.ndarray, depth: np.ndarray) -> pv.PolyData:
        h, w = depth.shape
        target_w = max(16, int(w * self.resolution_scale))
        target_h = max(16, int(h * self.resolution_scale))

        resized_rgb = resize_image(rgb, (target_w, target_h))
        resized_depth = resize_image(depth, (target_w, target_h))

        if self.remove_star_outliers:
            starless_depth = median_filter(resized_depth, size=5)
            working_depth = np.minimum(resized_depth, starless_depth + 0.15)
            clean_rgb, _ = filter_stars(resized_rgb, threshold=0.90, filter_size=5)
            gray = to_grayscale(clean_rgb)
        else:
            working_depth = resized_depth
            gray = to_grayscale(resized_rgb)

        if self.taper_edges and self.threshold > 0.0:
            taper = np.clip((gray - self.threshold * 0.4) / (self.threshold * 1.6), 0.0, 1.0)
            taper = 0.5 * (1.0 - np.cos(np.pi * taper))
            shaped_depth = working_depth * taper
        else:
            shaped_depth = working_depth

        x_coords = np.linspace(-target_w / 2.0, target_w / 2.0, target_w, dtype=np.float32)
        y_coords = np.linspace(target_h / 2.0, -target_h / 2.0, target_h, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)

        z_half_scale = self.depth_scale * 0.5
        z_front = (np.maximum(0.0, shaped_depth) * z_half_scale).astype(np.float32)
        pts_front = np.column_stack((xx.ravel(), yy.ravel(), z_front.ravel()))
        colors_front = (np.clip(resized_rgb.reshape(-1, 3), 0.0, 1.0) * 255.0).astype(np.uint8)

        u_coords = np.linspace(0.0, 1.0, target_w, dtype=np.float32)
        v_coords = np.linspace(1.0, 0.0, target_h, dtype=np.float32)
        uu, vv = np.meshgrid(u_coords, v_coords)
        uvs_front = np.column_stack((uu.ravel(), vv.ravel()))

        num_cells_x = target_w - 1
        num_cells_y = target_h - 1

        if self.threshold > 0.0:
            cell_flux = (
                gray[:-1, :-1] + gray[1:, :-1] + gray[:-1, 1:] + gray[1:, 1:]
            ) * 0.25
            cell_depth = (
                shaped_depth[:-1, :-1] + shaped_depth[1:, :-1] + shaped_depth[:-1, 1:] + shaped_depth[1:, 1:]
            ) * 0.25
            quad_mask = (cell_flux > self.threshold) | (cell_depth > self.threshold)

            labeled, num_features = label(quad_mask)
            if num_features > 1:
                sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
                max_label = int(np.argmax(sizes)) + 1
                max_size = sizes[max_label - 1]
                for i, s in enumerate(sizes, 1):
                    if s < max_size * 0.03:
                        quad_mask[labeled == i] = False

            if not np.any(quad_mask):
                quad_mask = np.ones((num_cells_y, num_cells_x), dtype=bool)
        else:
            quad_mask = np.ones((num_cells_y, num_cells_x), dtype=bool)

        r_idx = np.arange(num_cells_y)[:, None]
        c_idx = np.arange(num_cells_x)[None, :]
        v00 = r_idx * target_w + c_idx
        v01 = v00 + 1
        v10 = (r_idx + 1) * target_w + c_idx
        v11 = v10 + 1

        v00_sel = v00[quad_mask]
        v01_sel = v01[quad_mask]
        v10_sel = v10[quad_mask]
        v11_sel = v11[quad_mask]

        f_t1 = np.column_stack([np.full(len(v00_sel), 3), v00_sel, v10_sel, v01_sel])
        f_t2 = np.column_stack([np.full(len(v00_sel), 3), v01_sel, v10_sel, v11_sel])
        front_faces = np.vstack([f_t1, f_t2])

        if not self.double_sided:
            mesh = pv.PolyData(pts_front, front_faces)
            mesh.point_data["RGB"] = colors_front
            mesh.active_texture_coordinates = uvs_front
            mesh = mesh.clean()
            if self.smooth_iterations > 0:
                try:
                    mesh = mesh.smooth_taubin(n_iter=self.smooth_iterations)
                except Exception:
                    mesh = mesh.smooth(n_iter=self.smooth_iterations)
            return mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)

        z_back = -z_front
        pts_back = np.column_stack((xx.ravel(), yy.ravel(), z_back.ravel()))
        all_pts = np.vstack([pts_front, pts_back])
        all_colors = np.vstack([colors_front, colors_front.copy()])
        all_uvs = np.vstack([uvs_front, uvs_front.copy()])

        offset = target_w * target_h
        bv00_sel = v00_sel + offset
        bv01_sel = v01_sel + offset
        bv10_sel = v10_sel + offset
        bv11_sel = v11_sel + offset

        b_t1 = np.column_stack([np.full(len(bv00_sel), 3), bv00_sel, bv01_sel, bv10_sel])
        b_t2 = np.column_stack([np.full(len(bv00_sel), 3), bv01_sel, bv11_sel, bv10_sel])
        back_faces = np.vstack([b_t1, b_t2])

        face_list = [front_faces, back_faces]

        if self.close_mesh:
            top_boundary = quad_mask & np.pad(~quad_mask[:-1, :], ((1, 0), (0, 0)), constant_values=True)
            bot_boundary = quad_mask & np.pad(~quad_mask[1:, :], ((0, 1), (0, 0)), constant_values=True)
            left_boundary = quad_mask & np.pad(~quad_mask[:, :-1], ((0, 0), (1, 0)), constant_values=True)
            right_boundary = quad_mask & np.pad(~quad_mask[:, 1:], ((0, 0), (0, 1)), constant_values=True)

            e_top = np.column_stack([v00[top_boundary], v01[top_boundary]])
            e_right = np.column_stack([v01[right_boundary], v11[right_boundary]])
            e_bot = np.column_stack([v11[bot_boundary], v10[bot_boundary]])
            e_left = np.column_stack([v10[left_boundary], v00[left_boundary]])
            boundary_edges = np.vstack([e_top, e_right, e_bot, e_left])

            if len(boundary_edges) > 0:
                v0 = boundary_edges[:, 0]
                v1 = boundary_edges[:, 1]
                s_t1 = np.column_stack([np.full(len(v0), 3), v0, v1, v1 + offset])
                s_t2 = np.column_stack([np.full(len(v0), 3), v0, v1 + offset, v0 + offset])
                face_list.append(s_t1)
                face_list.append(s_t2)

        all_faces = np.vstack(face_list)
        mesh = pv.PolyData(all_pts, all_faces)
        mesh.point_data["RGB"] = all_colors
        mesh.active_texture_coordinates = all_uvs
        mesh = mesh.clean()

        if self.smooth_iterations > 0:
            try:
                mesh = mesh.smooth_taubin(n_iter=self.smooth_iterations)
            except Exception:
                mesh = mesh.smooth(n_iter=self.smooth_iterations)

        return mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)

