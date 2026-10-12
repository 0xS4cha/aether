import numpy as np
import pyvista as pv
from aether.geometry import PointCloudReconstructor, MeshReconstructor, VolumeReconstructor

def test_point_cloud_reconstruction() -> None:
    rgb = np.ones((40, 40, 3), dtype=np.float32) * 0.7
    depth = np.ones((40, 40), dtype=np.float32) * 0.5
    reconstructor = PointCloudReconstructor(depth_scale=30.0, threshold=0.1, subsample_step=1)
    points, colors = reconstructor.reconstruct(rgb, depth)
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert colors.ndim == 2
    assert colors.shape[1] == 3
    assert len(points) == len(colors)
    assert len(points) > 0

def test_mesh_reconstruction() -> None:
    rgb = np.ones((32, 32, 3), dtype=np.float32) * 0.5
    depth = np.random.uniform(0.2, 0.8, (32, 32)).astype(np.float32)
    reconstructor = MeshReconstructor(depth_scale=20.0, resolution_scale=0.5, double_sided=True)
    mesh = reconstructor.reconstruct(rgb, depth)
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_points > 0
    assert "RGB" in mesh.point_data

def test_volume_reconstruction() -> None:
    rgb = np.ones((32, 32, 3), dtype=np.float32) * 0.6
    depth = np.full((32, 32), 0.5, dtype=np.float32)
    reconstructor = VolumeReconstructor(depth_layers=16, resolution_scale=0.5)
    volume = reconstructor.reconstruct(rgb, depth)
    assert isinstance(volume, pv.ImageData)
    assert "density" in volume.point_data
    assert volume.dimensions[2] == 16
