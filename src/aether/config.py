from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

@dataclass
class DepthConfig:
    model_type: str = "hybrid"
    invert: bool = False
    normalize: bool = True
    astronomical_radial_weight: float = 0.4
    astronomical_power: float = 1.5

@dataclass
class PointCloudConfig:
    depth_scale: float = 50.0
    subsample_step: int = 1
    threshold: float = 0.05
    remove_star_outliers: bool = True

@dataclass
class MeshConfig:
    depth_scale: float = 40.0
    resolution_scale: float = 0.5
    smooth_iterations: int = 15
    double_sided: bool = True
    threshold: float = 0.03
    remove_star_outliers: bool = True
    close_mesh: bool = True
    taper_edges: bool = True

@dataclass
class VolumeConfig:
    depth_layers: int = 48
    thickness_factor: float = 1.0
    gaussian_sigma: float = 1.8

@dataclass
class GeometryConfig:
    representation: str = "point_cloud"
    point_cloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)

@dataclass
class StarfieldConfig:
    enabled: bool = True
    num_stars: int = 1200
    spread: float = 300.0

@dataclass
class RenderingConfig:
    background_color: str = "black"
    colormap: str = "plasma"
    show_axes: bool = True
    point_size: float = 2.5
    opacity_curve: str = "sigmoid"
    starfield: StarfieldConfig = field(default_factory=StarfieldConfig)

@dataclass
class ExportConfig:
    output_dir: str = "data/outputs"
    format: str = "ply"

@dataclass
class SpectralConfig:
    enabled: bool = False
    mode: str = "natural_rgb"
    filter_sigma: float = 1.0
    oiii_radius: float = 0.65
    ha_radius: float = 1.00
    sii_radius: float = 1.25

@dataclass
class MorphologyConfig:
    enabled: bool = False
    model_type: str = "bipolar"
    inclination_deg: float = 45.0
    position_angle_deg: float = 0.0
    num_points: int = 40000
    grid_size: int = 64

@dataclass
class AetherConfig:
    device: str = "auto"
    depth: DepthConfig = field(default_factory=DepthConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base

def load_config(config_path: str | Path | None = None) -> AetherConfig:
    config = AetherConfig()
    path = Path(config_path) if config_path else Path("config/default.yaml")
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            raw_data = yaml.safe_load(file) or {}
            
        depth_data = raw_data.get("depth", {})
        config.depth = DepthConfig(**{k: v for k, v in depth_data.items() if hasattr(DepthConfig, k)})
        
        geom_data = raw_data.get("geometry", {})
        pc_data = geom_data.get("point_cloud", {})
        mesh_data = geom_data.get("mesh", {})
        vol_data = geom_data.get("volume", {})
        
        config.geometry = GeometryConfig(
            representation=geom_data.get("representation", "point_cloud"),
            point_cloud=PointCloudConfig(**{k: v for k, v in pc_data.items() if hasattr(PointCloudConfig, k)}),
            mesh=MeshConfig(**{k: v for k, v in mesh_data.items() if hasattr(MeshConfig, k)}),
            volume=VolumeConfig(**{k: v for k, v in vol_data.items() if hasattr(VolumeConfig, k)}),
        )
        
        rend_data = raw_data.get("rendering", {})
        sf_data = rend_data.get("starfield", {})
        config.rendering = RenderingConfig(
            background_color=rend_data.get("background_color", "black"),
            colormap=rend_data.get("colormap", "plasma"),
            show_axes=rend_data.get("show_axes", True),
            point_size=rend_data.get("point_size", 2.5),
            opacity_curve=rend_data.get("opacity_curve", "sigmoid"),
            starfield=StarfieldConfig(**{k: v for k, v in sf_data.items() if hasattr(StarfieldConfig, k)}),
        )
        
        spec_data = raw_data.get("spectral", {})
        config.spectral = SpectralConfig(**{k: v for k, v in spec_data.items() if hasattr(SpectralConfig, k)})
        
        morph_data = raw_data.get("morphology", {})
        config.morphology = MorphologyConfig(**{k: v for k, v in morph_data.items() if hasattr(MorphologyConfig, k)})
        
        exp_data = raw_data.get("export", {})
        config.export = ExportConfig(**{k: v for k, v in exp_data.items() if hasattr(ExportConfig, k)})
        config.device = raw_data.get("device", "auto")

    return config
