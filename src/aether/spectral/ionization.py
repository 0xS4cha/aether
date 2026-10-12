from dataclasses import dataclass, field
import numpy as np
from aether.spectral.decomposer import SpectralChannels

@dataclass
class IonSpeciesConfig:
    radius_factor: float
    color: tuple[float, float, float]
    depth_multiplier: float

@dataclass
class IonizationProfile:
    oiii: IonSpeciesConfig = field(default_factory=lambda: IonSpeciesConfig(0.65, (0.08, 0.88, 0.95), 0.75))
    ha: IonSpeciesConfig = field(default_factory=lambda: IonSpeciesConfig(1.00, (0.95, 0.22, 0.32), 1.00))
    sii: IonSpeciesConfig = field(default_factory=lambda: IonSpeciesConfig(1.25, (0.96, 0.68, 0.12), 1.25))

class IonizationModel:
    def __init__(self, profile: IonizationProfile | None = None) -> None:
        self.profile = profile or IonizationProfile()

    def compute_ionization_depths(
        self,
        base_depth: np.ndarray,
        channels: SpectralChannels,
    ) -> dict[str, np.ndarray]:
        h, w = base_depth.shape
        oiii_flux = channels.oiii
        
        threshold = np.percentile(oiii_flux, 80)
        mask = oiii_flux > threshold
        if np.any(mask):
            y_indices, x_indices = np.indices((h, w))
            total_flux = np.sum(oiii_flux[mask])
            cy = float(np.sum(y_indices[mask] * oiii_flux[mask]) / total_flux)
            cx = float(np.sum(x_indices[mask] * oiii_flux[mask]) / total_flux)
        else:
            cy, cx = h / 2.0, w / 2.0

        y_coords = (np.arange(h) - cy) / (h / 2.0)
        x_coords = (np.arange(w) - cx) / (w / 2.0)
        xx, yy = np.meshgrid(x_coords, y_coords)
        radius = np.sqrt(xx**2 + yy**2)

        oiii_cfg = self.profile.oiii
        ha_cfg = self.profile.ha
        sii_cfg = self.profile.sii

        oiii_shell = np.sqrt(np.maximum(0.0, oiii_cfg.radius_factor**2 - radius**2))
        max_oiii = float(np.max(oiii_shell)) if float(np.max(oiii_shell)) > 0 else 1.0
        oiii_depth = (0.5 * base_depth + 0.5 * (oiii_shell / max_oiii)) * oiii_cfg.depth_multiplier

        ha_shell = np.sqrt(np.maximum(0.0, ha_cfg.radius_factor**2 - radius**2))
        max_ha = float(np.max(ha_shell)) if float(np.max(ha_shell)) > 0 else 1.0
        ha_depth = (0.6 * base_depth + 0.4 * (ha_shell / max_ha)) * ha_cfg.depth_multiplier

        sii_shell = np.sqrt(np.maximum(0.0, sii_cfg.radius_factor**2 - radius**2))
        max_sii = float(np.max(sii_shell)) if float(np.max(sii_shell)) > 0 else 1.0
        sii_depth = (0.4 * base_depth + 0.6 * (sii_shell / max_sii)) * sii_cfg.depth_multiplier

        return {
            "oiii": self._normalize_depth(oiii_depth),
            "ha": self._normalize_depth(ha_depth),
            "sii": self._normalize_depth(sii_depth),
        }

    @staticmethod
    def _normalize_depth(depth: np.ndarray) -> np.ndarray:
        d_min = float(np.min(depth))
        d_max = float(np.max(depth))
        if d_max - d_min > 1e-7:
            return ((depth - d_min) / (d_max - d_min)).astype(np.float32)
        return np.zeros_like(depth, dtype=np.float32)
