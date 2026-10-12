from pathlib import Path
import numpy as np
import pyvista as pv
from aether.io.image import load_image, save_image, filter_stars, to_grayscale
from aether.io.dataset import generate_synthetic_nebula
from aether.io.exporters import export_point_cloud, export_volume, export_depth_map

def test_generate_synthetic_nebula() -> None:
    img = generate_synthetic_nebula(width=64, height=64, seed=123)
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.float32
    assert img.min() >= 0.0
    assert img.max() <= 1.0

def test_image_save_and_load(tmp_path: Path) -> None:
    img = np.random.uniform(0.0, 1.0, (48, 48, 3)).astype(np.float32)
    path = tmp_path / "test.png"
    save_image(path, img)
    loaded = load_image(path)
    assert loaded.shape == (48, 48, 3)

def test_filter_stars() -> None:
    img = np.zeros((50, 50, 3), dtype=np.float32)
    img[25, 25] = 1.0
    cleaned, mask = filter_stars(img, threshold=0.8)
    assert mask.shape == (50, 50)
    assert cleaned.shape == (50, 50, 3)

def test_export_point_cloud(tmp_path: Path) -> None:
    points = np.random.uniform(-10.0, 10.0, (100, 3)).astype(np.float32)
    colors = np.random.uniform(0.0, 1.0, (100, 3)).astype(np.float32)
    dest = tmp_path / "cloud.ply"
    export_point_cloud(dest, points, colors)
    assert dest.exists()
    assert dest.stat().st_size > 0

def test_export_volume(tmp_path: Path) -> None:
    grid = pv.ImageData()
    grid.dimensions = np.array([10, 10, 10])
    grid.point_data["density"] = np.ones(1000, dtype=np.float32)
    dest = tmp_path / "volume.vti"
    export_volume(dest, grid)
    assert dest.exists()
    assert dest.stat().st_size > 0

def test_export_depth_map(tmp_path: Path) -> None:
    depth = np.random.uniform(0.0, 1.0, (30, 30)).astype(np.float32)
    dest = tmp_path / "depth.png"
    export_depth_map(dest, depth)
    assert dest.exists()
    assert dest.stat().st_size > 0
