from pathlib import Path
import numpy as np
import pyvista as pv
from aether.morphology import (
    BipolarModel,
    ToroidalModel,
    EllipsoidalModel,
    MorphologicalDeprojector,
    CameraPathGenerator,
    create_morphology_model,
)
from aether.config import AetherConfig
from aether.pipeline import NebulaPipeline
from aether.io.dataset import generate_synthetic_nebula

def test_morphology_models_density() -> None:
    bipolar = BipolarModel(inclination_deg=35.0, position_angle_deg=10.0)
    toroidal = ToroidalModel(inclination_deg=45.0)
    ellipsoidal = EllipsoidalModel(inclination_deg=20.0)

    grid = np.linspace(-1.0, 1.0, 15, dtype=np.float32)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")

    b_dens = bipolar.compute_density(x, y, z)
    t_dens = toroidal.compute_density(x, y, z)
    e_dens = ellipsoidal.compute_density(x, y, z)

    assert b_dens.shape == (15, 15, 15)
    assert t_dens.shape == (15, 15, 15)
    assert e_dens.shape == (15, 15, 15)

    assert b_dens.min() >= 0.0
    assert t_dens.min() >= 0.0
    assert e_dens.min() >= 0.0

def test_rotation_matrix_orthogonality() -> None:
    model = BipolarModel(inclination_deg=45.0, position_angle_deg=30.0)
    rot = model.get_rotation_matrix()
    identity = rot @ rot.T
    assert np.allclose(identity, np.eye(3), atol=1e-5)

def test_create_morphology_model_factory() -> None:
    m_bip = create_morphology_model("bipolar")
    m_tor = create_morphology_model("toroidal")
    m_ell = create_morphology_model("ellipsoidal")

    assert isinstance(m_bip, BipolarModel)
    assert isinstance(m_tor, ToroidalModel)
    assert isinstance(m_ell, EllipsoidalModel)

def test_deproject_volume() -> None:
    deprojector = MorphologicalDeprojector(BipolarModel(inclination_deg=45.0))
    img = np.random.uniform(0.0, 1.0, (32, 32, 3)).astype(np.float32)
    vol = deprojector.deproject_volume(img, grid_size=16)

    assert isinstance(vol, pv.ImageData)
    assert vol.dimensions == (16, 16, 16)
    assert "density" in vol.point_data
    assert "RGB" in vol.point_data

def test_deproject_point_cloud() -> None:
    deprojector = MorphologicalDeprojector(ToroidalModel(inclination_deg=30.0))
    img = np.ones((32, 32, 3), dtype=np.float32) * 0.8
    pts, clrs = deprojector.deproject_point_cloud(img, num_points=1000)

    assert len(pts) > 0
    assert len(clrs) == len(pts)
    assert pts.shape[1] == 3
    assert clrs.shape[1] == 3

def test_deproject_surface_mesh() -> None:
    deprojector = MorphologicalDeprojector(BipolarModel(inclination_deg=45.0))
    img = np.ones((32, 32, 3), dtype=np.float32) * 0.7
    mesh = deprojector.deproject_surface_mesh(img, grid_size=24, iso_value=0.15)

    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_points > 0
    assert "RGB" in mesh.point_data

def test_camera_trajectories() -> None:
    orbit = CameraPathGenerator.generate_orbit(num_frames=12)
    fly = CameraPathGenerator.generate_fly_through(num_frames=12)

    assert len(orbit) == 12
    assert len(fly) == 12
    assert len(orbit[0]) == 3

def test_pipeline_deproject_end_to_end() -> None:
    config = AetherConfig()
    config.morphology.enabled = True
    config.morphology.model_type = "bipolar"
    config.morphology.num_points = 500
    config.geometry.representation = "point_cloud"

    pipeline = NebulaPipeline(config)
    img = generate_synthetic_nebula(width=48, height=48)
    result = pipeline.process(img)

    assert result["deprojected"] is True
    assert "points" in result
    assert "colors" in result
    assert len(result["points"]) > 0

def test_cinematic_animation_render(tmp_path: Path) -> None:
    config = AetherConfig()
    config.morphology.enabled = True
    config.morphology.grid_size = 12
    config.geometry.representation = "volume"
    pipeline = NebulaPipeline(config)
    pipeline.visualizer.off_screen = True

    img = generate_synthetic_nebula(width=32, height=32)
    result = pipeline.process(img)

    dest = tmp_path / "cinematic.gif"
    pipeline.render_cinematic(result, dest, mode="orbit", num_frames=6)
    assert dest.exists()
    assert dest.stat().st_size > 0
