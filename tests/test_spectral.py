import numpy as np
import pyvista as pv
from aether.spectral import (
    SpectralDecomposer,
    SpectralChannels,
    IonizationModel,
    IonizationProfile,
    IonSpeciesConfig,
    SpectralReconstructor,
)
from aether.config import AetherConfig
from aether.pipeline import NebulaPipeline
from aether.io.dataset import generate_synthetic_nebula

def test_spectral_decomposer_natural() -> None:
    decomposer = SpectralDecomposer()
    img = np.random.uniform(0.0, 1.0, (40, 40, 3)).astype(np.float32)
    channels = decomposer.decompose(img, mode="natural_rgb")
    assert isinstance(channels, SpectralChannels)
    assert channels.oiii.shape == (40, 40)
    assert channels.ha.shape == (40, 40)
    assert channels.sii.shape == (40, 40)
    assert 0.0 <= channels.oiii.min() <= channels.oiii.max() <= 1.0

def test_spectral_decomposer_hubble_sho() -> None:
    decomposer = SpectralDecomposer()
    img = np.random.uniform(0.0, 1.0, (32, 32, 3)).astype(np.float32)
    channels = decomposer.decompose(img, mode="hubble_sho")
    assert channels.oiii.shape == (32, 32)
    sho_rgb = decomposer.to_sho_rgb(channels)
    assert sho_rgb.shape == (32, 32, 3)

def test_spectral_decomposer_hoo_bicolor() -> None:
    decomposer = SpectralDecomposer()
    img = np.random.uniform(0.0, 1.0, (32, 32, 3)).astype(np.float32)
    channels = decomposer.decompose(img, mode="hoo_bicolor")
    assert channels.ha.shape == (32, 32)
    assert channels.oiii.shape == (32, 32)

def test_ionization_model_depths() -> None:
    model = IonizationModel()
    channels = SpectralChannels(
        oiii=np.ones((30, 30), dtype=np.float32) * 0.5,
        ha=np.ones((30, 30), dtype=np.float32) * 0.7,
        sii=np.ones((30, 30), dtype=np.float32) * 0.3,
    )
    base_depth = np.random.uniform(0.2, 0.8, (30, 30)).astype(np.float32)
    depths = model.compute_ionization_depths(base_depth, channels)
    assert "oiii" in depths
    assert "ha" in depths
    assert "sii" in depths
    assert depths["oiii"].shape == (30, 30)

def test_spectral_reconstructor_point_cloud() -> None:
    model = IonizationModel()
    reconstructor = SpectralReconstructor(ionization_model=model, threshold=0.01)
    channels = SpectralChannels(
        oiii=np.ones((20, 20), dtype=np.float32) * 0.5,
        ha=np.ones((20, 20), dtype=np.float32) * 0.6,
        sii=np.ones((20, 20), dtype=np.float32) * 0.4,
    )
    depths = {
        "oiii": np.full((20, 20), 0.4, dtype=np.float32),
        "ha": np.full((20, 20), 0.5, dtype=np.float32),
        "sii": np.full((20, 20), 0.6, dtype=np.float32),
    }
    pts, clrs, ids = reconstructor.reconstruct_point_cloud(channels, depths)
    assert len(pts) > 0
    assert len(clrs) == len(pts)
    assert len(ids) == len(pts)

def test_spectral_reconstructor_shells() -> None:
    reconstructor = SpectralReconstructor()
    channels = SpectralChannels(
        oiii=np.ones((24, 24), dtype=np.float32) * 0.5,
        ha=np.ones((24, 24), dtype=np.float32) * 0.6,
        sii=np.ones((24, 24), dtype=np.float32) * 0.4,
    )
    depths = {
        "oiii": np.full((24, 24), 0.4, dtype=np.float32),
        "ha": np.full((24, 24), 0.5, dtype=np.float32),
        "sii": np.full((24, 24), 0.6, dtype=np.float32),
    }
    shells = reconstructor.reconstruct_concentric_shells(channels, depths, resolution_scale=0.5)
    assert "oiii" in shells
    assert "ha" in shells
    assert "sii" in shells
    assert isinstance(shells["oiii"], pv.PolyData)

def test_spectral_reconstructor_volume() -> None:
    reconstructor = SpectralReconstructor()
    channels = SpectralChannels(
        oiii=np.ones((20, 20), dtype=np.float32) * 0.5,
        ha=np.ones((20, 20), dtype=np.float32) * 0.6,
        sii=np.ones((20, 20), dtype=np.float32) * 0.4,
    )
    depths = {
        "oiii": np.full((20, 20), 0.4, dtype=np.float32),
        "ha": np.full((20, 20), 0.5, dtype=np.float32),
        "sii": np.full((20, 20), 0.6, dtype=np.float32),
    }
    vol = reconstructor.reconstruct_spectral_volume(channels, depths, depth_layers=12, resolution_scale=0.5)
    assert isinstance(vol, pv.ImageData)
    assert "density" in vol.point_data
    assert "RGB" in vol.point_data

def test_pipeline_spectral_end_to_end() -> None:
    config = AetherConfig()
    config.depth.model_type = "astronomical"
    config.spectral.enabled = True
    config.geometry.representation = "mesh"
    pipeline = NebulaPipeline(config)
    img = generate_synthetic_nebula(width=48, height=48)
    res = pipeline.process(img)
    assert res["spectral"] is True
    assert "channels" in res
    assert "shells" in res
