from pathlib import Path
from typing import Any
import numpy as np
import pyvista as pv
from aether.config import AetherConfig, load_config
from aether.depth import create_depth_estimator, BaseDepthEstimator
from aether.geometry import PointCloudReconstructor, MeshReconstructor, VolumeReconstructor
from aether.spectral import (
    SpectralDecomposer,
    IonizationModel,
    IonizationProfile,
    IonSpeciesConfig,
    SpectralReconstructor,
)
from aether.morphology import (
    MorphologicalDeprojector,
    CameraPathGenerator,
    create_morphology_model,
)
from aether.io.image import load_image
from aether.io.exporters import export_point_cloud, export_mesh, export_volume
from aether.rendering import NebulaVisualizer

class NebulaPipeline:
    def __init__(self, config: AetherConfig | None = None) -> None:
        self.config = config or load_config()
        self.depth_estimator: BaseDepthEstimator = create_depth_estimator(
            model_type=self.config.depth.model_type,
            device=self.config.device,
            astro_weight=self.config.depth.astronomical_radial_weight,
            power=self.config.depth.astronomical_power,
        )
        self.pc_reconstructor = PointCloudReconstructor(
            depth_scale=self.config.geometry.point_cloud.depth_scale,
            subsample_step=self.config.geometry.point_cloud.subsample_step,
            threshold=self.config.geometry.point_cloud.threshold,
            remove_star_outliers=self.config.geometry.point_cloud.remove_star_outliers,
        )
        self.mesh_reconstructor = MeshReconstructor(
            depth_scale=self.config.geometry.mesh.depth_scale,
            resolution_scale=self.config.geometry.mesh.resolution_scale,
            smooth_iterations=self.config.geometry.mesh.smooth_iterations,
            double_sided=self.config.geometry.mesh.double_sided,
            threshold=self.config.geometry.mesh.threshold,
            remove_star_outliers=self.config.geometry.mesh.remove_star_outliers,
            close_mesh=self.config.geometry.mesh.close_mesh,
            taper_edges=self.config.geometry.mesh.taper_edges,
        )
        self.volume_reconstructor = VolumeReconstructor(
            depth_layers=self.config.geometry.volume.depth_layers,
            thickness_factor=self.config.geometry.volume.thickness_factor,
            gaussian_sigma=self.config.geometry.volume.gaussian_sigma,
        )
        self.spectral_decomposer = SpectralDecomposer(
            filter_sigma=self.config.spectral.filter_sigma
        )
        ion_profile = IonizationProfile(
            oiii=IonSpeciesConfig(self.config.spectral.oiii_radius, (0.08, 0.88, 0.95), 0.75),
            ha=IonSpeciesConfig(self.config.spectral.ha_radius, (0.95, 0.22, 0.32), 1.00),
            sii=IonSpeciesConfig(self.config.spectral.sii_radius, (0.96, 0.68, 0.12), 1.25),
        )
        self.ionization_model = IonizationModel(ion_profile)
        self.spectral_reconstructor = SpectralReconstructor(
            ionization_model=self.ionization_model,
            depth_scale=self.config.geometry.point_cloud.depth_scale,
            threshold=self.config.geometry.point_cloud.threshold,
            subsample_step=self.config.geometry.point_cloud.subsample_step,
        )
        morph_model = create_morphology_model(
            model_type=self.config.morphology.model_type,
            inclination_deg=self.config.morphology.inclination_deg,
            position_angle_deg=self.config.morphology.position_angle_deg,
        )
        self.deprojector = MorphologicalDeprojector(model=morph_model)
        self.visualizer = NebulaVisualizer(
            background_color=self.config.rendering.background_color,
            show_axes=self.config.rendering.show_axes,
        )

    def process(
        self,
        image_input: str | Path | np.ndarray,
        representation: str | None = None,
        spectral: bool | None = None,
        deproject: bool | None = None,
    ) -> dict[str, Any]:
        if isinstance(image_input, (str, Path)):
            rgb = load_image(image_input)
        else:
            rgb = image_input

        depth = self.depth_estimator.estimate(rgb)
        if self.config.depth.invert:
            depth = 1.0 - depth

        rep_mode = representation or self.config.geometry.representation
        use_spectral = spectral if spectral is not None else self.config.spectral.enabled
        use_deproject = deproject if deproject is not None else self.config.morphology.enabled

        result: dict[str, Any] = {
            "rgb": rgb,
            "depth": depth,
            "representation": rep_mode,
            "spectral": use_spectral,
            "deprojected": use_deproject,
        }

        if use_deproject:
            if rep_mode == "point_cloud":
                points, colors = self.deprojector.deproject_point_cloud(
                    rgb,
                    num_points=self.config.morphology.num_points,
                )
                result["points"] = points
                result["colors"] = colors
            elif rep_mode == "mesh":
                mesh = self.deprojector.deproject_surface_mesh(
                    rgb,
                    grid_size=self.config.morphology.grid_size,
                )
                result["mesh"] = mesh
            elif rep_mode == "volume":
                volume = self.deprojector.deproject_volume(
                    rgb,
                    grid_size=self.config.morphology.grid_size,
                )
                result["volume"] = volume
            else:
                raise ValueError(f"Unsupported representation: {rep_mode}")
            return result

        if use_spectral:
            channels = self.spectral_decomposer.decompose(rgb, mode=self.config.spectral.mode)
            ion_depths = self.ionization_model.compute_ionization_depths(depth, channels)
            result["channels"] = channels
            result["ion_depths"] = ion_depths

            if rep_mode == "point_cloud":
                points, colors, species_ids = self.spectral_reconstructor.reconstruct_point_cloud(
                    channels, ion_depths
                )
                result["points"] = points
                result["colors"] = colors
                result["species_ids"] = species_ids
            elif rep_mode == "mesh":
                shells = self.spectral_reconstructor.reconstruct_concentric_shells(
                    channels,
                    ion_depths,
                    resolution_scale=self.config.geometry.mesh.resolution_scale,
                )
                result["shells"] = shells
            elif rep_mode == "volume":
                volume = self.spectral_reconstructor.reconstruct_spectral_volume(
                    channels,
                    ion_depths,
                    depth_layers=self.config.geometry.volume.depth_layers,
                )
                result["volume"] = volume
            else:
                raise ValueError(f"Unsupported representation: {rep_mode}")
        else:
            if rep_mode == "point_cloud":
                points, colors = self.pc_reconstructor.reconstruct(rgb, depth)
                result["points"] = points
                result["colors"] = colors
            elif rep_mode == "mesh":
                mesh = self.mesh_reconstructor.reconstruct(rgb, depth)
                result["mesh"] = mesh
            elif rep_mode == "volume":
                volume = self.volume_reconstructor.reconstruct(rgb, depth)
                result["volume"] = volume
            else:
                raise ValueError(f"Unsupported representation: {rep_mode}")

        return result

    def render(
        self,
        result: dict[str, Any],
        show: bool = True,
        save_path: str | Path | None = None,
    ) -> pv.Plotter:
        rep_mode = result["representation"]
        stars = self.config.rendering.starfield.enabled
        is_spectral = result.get("spectral", False)
        is_deprojected = result.get("deprojected", False)

        if is_spectral and not is_deprojected and rep_mode == "mesh":
            return self.visualizer.plot_concentric_shells(
                result["shells"],
                add_stars=stars,
                show=show,
                save_path=save_path,
            )

        if rep_mode == "point_cloud":
            return self.visualizer.plot_point_cloud(
                result["points"],
                result["colors"],
                point_size=self.config.rendering.point_size,
                add_stars=stars,
                show=show,
                save_path=save_path,
            )
        elif rep_mode == "mesh":
            return self.visualizer.plot_mesh(
                result["mesh"],
                add_stars=stars,
                show=show,
                save_path=save_path,
            )
        elif rep_mode == "volume":
            return self.visualizer.plot_volume(
                result["volume"],
                cmap=self.config.rendering.colormap,
                opacity_name=self.config.rendering.opacity_curve,
                add_stars=stars,
                show=show,
                save_path=save_path,
            )
        else:
            raise ValueError(f"Cannot render unknown representation: {rep_mode}")

    def render_cinematic(
        self,
        result: dict[str, Any],
        output_path: str | Path,
        mode: str = "orbit",
        num_frames: int = 48,
    ) -> None:
        plotter = self.render(result, show=False)
        if mode == "fly_through":
            trajectory = CameraPathGenerator.generate_fly_through(num_frames=num_frames)
        else:
            trajectory = CameraPathGenerator.generate_orbit(num_frames=num_frames)
        CameraPathGenerator.render_animation(plotter, trajectory, output_path)

    def export(self, result: dict[str, Any], output_path: str | Path) -> None:
        path = Path(output_path)
        rep_mode = result["representation"]
        is_spectral = result.get("spectral", False)
        is_deprojected = result.get("deprojected", False)

        if is_spectral and not is_deprojected and rep_mode == "mesh":
            stem = path.stem
            suffix = path.suffix or ".ply"
            parent = path.parent
            parent.mkdir(parents=True, exist_ok=True)
            for spec_name, shell_mesh in result["shells"].items():
                dest = parent / f"{stem}_{spec_name}{suffix}"
                export_mesh(dest, shell_mesh)
            return

        if rep_mode == "point_cloud":
            export_point_cloud(path, result["points"], result["colors"])
        elif rep_mode == "mesh":
            export_mesh(path, result["mesh"])
        elif rep_mode == "volume":
            export_volume(path, result["volume"])

    def run(
        self,
        image_input: str | Path | np.ndarray,
        representation: str | None = None,
        spectral: bool | None = None,
        deproject: bool | None = None,
        show: bool = True,
        export_path: str | Path | None = None,
    ) -> dict[str, Any]:
        result = self.process(
            image_input,
            representation=representation,
            spectral=spectral,
            deproject=deproject,
        )
        if export_path:
            self.export(result, export_path)
        self.render(result, show=show)
        return result
