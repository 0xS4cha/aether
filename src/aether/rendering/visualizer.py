from pathlib import Path
import numpy as np
import pyvista as pv
from aether.rendering.palettes import get_opacity_curve

class NebulaVisualizer:
    def __init__(
        self,
        background_color: str = "black",
        show_axes: bool = True,
        off_screen: bool = False,
    ) -> None:
        self.background_color = background_color
        self.show_axes = show_axes
        self.off_screen = off_screen

    def create_starfield(self, num_stars: int = 1200, spread: float = 400.0) -> pv.PolyData:
        rng = np.random.default_rng(42)
        star_positions = rng.uniform(-spread, spread, size=(num_stars, 3)).astype(np.float32)
        
        radial_dist = np.linalg.norm(star_positions, axis=1)
        mask = radial_dist > (spread * 0.4)
        star_positions = star_positions[mask]

        star_poly = pv.PolyData(star_positions)
        star_brightness = rng.uniform(0.5, 1.0, size=len(star_positions)).astype(np.float32)
        star_poly.point_data["brightness"] = star_brightness
        return star_poly

    def _setup_plotter(self, add_stars: bool = True, star_spread: float = 400.0) -> pv.Plotter:
        plotter = pv.Plotter(off_screen=self.off_screen)
        plotter.set_background(self.background_color)
        
        if self.show_axes:
            plotter.add_axes()

        if add_stars:
            stars = self.create_starfield(spread=star_spread)
            plotter.add_mesh(
                stars,
                scalars="brightness",
                cmap="gray",
                point_size=1.5,
                render_points_as_spheres=True,
                show_scalar_bar=False,
            )

        return plotter

    def plot_point_cloud(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        point_size: float = 2.5,
        add_stars: bool = True,
        show: bool = True,
        save_path: str | Path | None = None,
    ) -> pv.Plotter:
        plotter = self._setup_plotter(add_stars=add_stars)
        
        poly = pv.PolyData(points)
        poly.point_data["RGB"] = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)

        plotter.add_mesh(
            poly,
            scalars="RGB",
            rgb=True,
            point_size=point_size,
            render_points_as_spheres=True,
        )

        plotter.reset_camera()
        
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plotter.screenshot(str(p))

        if show and not self.off_screen:
            plotter.show()

        return plotter

    def plot_mesh(
        self,
        mesh: pv.PolyData,
        add_stars: bool = True,
        show: bool = True,
        save_path: str | Path | None = None,
    ) -> pv.Plotter:
        plotter = self._setup_plotter(add_stars=add_stars)

        if "RGB" in mesh.point_data:
            plotter.add_mesh(
                mesh,
                scalars="RGB",
                rgb=True,
                smooth_shading=True,
            )
        else:
            plotter.add_mesh(mesh, smooth_shading=True)

        plotter.reset_camera()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plotter.screenshot(str(p))

        if show and not self.off_screen:
            plotter.show()

        return plotter

    def plot_volume(
        self,
        volume: pv.ImageData,
        cmap: str = "plasma",
        opacity_name: str = "sigmoid",
        add_stars: bool = False,
        show: bool = True,
        save_path: str | Path | None = None,
    ) -> pv.Plotter:
        plotter = self._setup_plotter(add_stars=add_stars)
        opacity = get_opacity_curve(opacity_name)

        plotter.add_volume(
            volume,
            scalars="density",
            cmap=cmap,
            opacity=opacity,
            show_scalar_bar=False,
        )

        plotter.reset_camera()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plotter.screenshot(str(p))

        if show and not self.off_screen:
            plotter.show()

        return plotter

    def plot_concentric_shells(
        self,
        shells: dict[str, pv.PolyData],
        opacities: dict[str, float] | None = None,
        add_stars: bool = True,
        show: bool = True,
        save_path: str | Path | None = None,
    ) -> pv.Plotter:
        plotter = self._setup_plotter(add_stars=add_stars)
        default_opacities = {"oiii": 0.85, "ha": 0.55, "sii": 0.35}
        active_opacities = opacities or default_opacities

        for name, mesh in shells.items():
            op = active_opacities.get(name, 0.5)
            if "RGB" in mesh.point_data:
                plotter.add_mesh(
                    mesh,
                    scalars="RGB",
                    rgb=True,
                    opacity=op,
                    smooth_shading=True,
                )
            else:
                plotter.add_mesh(mesh, opacity=op, smooth_shading=True)

        plotter.reset_camera()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plotter.screenshot(str(p))

        if show and not self.off_screen:
            plotter.show()

        return plotter

