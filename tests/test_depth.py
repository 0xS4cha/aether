import numpy as np
import pytest
from aether.depth import create_depth_estimator, AstronomicalDepthEstimator, HybridDepthEstimator
from aether.depth.base import BaseDepthEstimator

def test_base_depth_normalize() -> None:
    arr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    norm = BaseDepthEstimator.normalize_depth(arr)
    assert norm.min() == pytest.approx(0.0)
    assert norm.max() == pytest.approx(1.0)
    
    inverted = BaseDepthEstimator.normalize_depth(arr, invert=True)
    assert inverted.min() == pytest.approx(0.0)
    assert inverted.max() == pytest.approx(1.0)
    assert inverted[0, 0] == pytest.approx(1.0)

def test_astronomical_depth_estimator() -> None:
    estimator = AstronomicalDepthEstimator(radial_weight=0.5, power=1.2, device="cpu")
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[20:44, 20:44] = 0.8
    depth = estimator.estimate(img)
    assert depth.shape == (64, 64)
    assert depth.dtype == np.float32
    assert 0.0 <= depth.min() <= depth.max() <= 1.0

def test_create_depth_estimator_factory() -> None:
    astro = create_depth_estimator("astronomical", device="cpu")
    assert isinstance(astro, AstronomicalDepthEstimator)
    
    hybrid = create_depth_estimator("hybrid", device="cpu")
    assert isinstance(hybrid, HybridDepthEstimator)

def test_hybrid_fallback_execution() -> None:
    hybrid = HybridDepthEstimator(model_type="invalid_model_type", device="cpu")
    img = np.ones((32, 32, 3), dtype=np.float32) * 0.5
    depth = hybrid.estimate(img)
    assert depth.shape == (32, 32)
    assert depth.dtype == np.float32
