from abc import ABC, abstractmethod
import numpy as np
import torch

class BaseDepthEstimator(ABC):
    def __init__(self, device: str = "auto") -> None:
        self.device = self.resolve_device(device)

    @staticmethod
    def resolve_device(device_pref: str) -> torch.device:
        if device_pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if device_pref == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if device_pref == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def normalize_depth(depth: np.ndarray, invert: bool = False) -> np.ndarray:
        d_min = float(np.min(depth))
        d_max = float(np.max(depth))
        if d_max - d_min > 1e-8:
            norm = (depth - d_min) / (d_max - d_min)
        else:
            norm = np.zeros_like(depth)
        if invert:
            norm = 1.0 - norm
        return norm.astype(np.float32)

    @abstractmethod
    def estimate(self, image: np.ndarray) -> np.ndarray:
        pass
