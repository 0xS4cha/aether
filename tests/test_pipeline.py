from pathlib import Path
import numpy as np
from aether.config import AetherConfig
from aether.pipeline import NebulaPipeline
from aether.io.dataset import generate_synthetic_nebula

def test_pipeline_point_cloud_process() -> None:
    config = AetherConfig()
    config.depth.model_type = "astronomical"
    config.geometry.representation = "point_cloud"
    pipeline = NebulaPipeline(config)
    img = generate_synthetic_nebula(width=64, height=64)
    result = pipeline.process(img)
    assert "points" in result
    assert "colors" in result
    assert "depth" in result

def test_pipeline_mesh_process() -> None:
    config = AetherConfig()
    config.depth.model_type = "astronomical"
    config.geometry.representation = "mesh"
    pipeline = NebulaPipeline(config)
    img = generate_synthetic_nebula(width=64, height=64)
    result = pipeline.process(img)
    assert "mesh" in result

def test_pipeline_volume_process() -> None:
    config = AetherConfig()
    config.depth.model_type = "astronomical"
    config.geometry.representation = "volume"
    config.geometry.volume.depth_layers = 12
    pipeline = NebulaPipeline(config)
    img = generate_synthetic_nebula(width=64, height=64)
    result = pipeline.process(img)
    assert "volume" in result

def test_pipeline_render_screenshot(tmp_path: Path) -> None:
    config = AetherConfig()
    config.depth.model_type = "astronomical"
    config.geometry.representation = "point_cloud"
    pipeline = NebulaPipeline(config)
    pipeline.visualizer.off_screen = True
    img = generate_synthetic_nebula(width=64, height=64)
    result = pipeline.process(img)
    screenshot_path = tmp_path / "shot.png"
    pipeline.render(result, show=False, save_path=screenshot_path)
    assert screenshot_path.exists()
