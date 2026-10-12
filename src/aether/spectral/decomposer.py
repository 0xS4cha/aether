from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter

@dataclass
class SpectralChannels:
    oiii: np.ndarray
    ha: np.ndarray
    sii: np.ndarray

class SpectralDecomposer:
    def __init__(self, filter_sigma: float = 1.0) -> None:
        self.filter_sigma = filter_sigma

    def decompose(self, rgb: np.ndarray, mode: str = "natural_rgb") -> SpectralChannels:
        norm_rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        r = norm_rgb[..., 0]
        g = norm_rgb[..., 1]
        b = norm_rgb[..., 2]

        normalized_mode = mode.lower()

        if normalized_mode == "hubble_sho":
            sii_raw = r
            ha_raw = g
            oiii_raw = b
        elif normalized_mode == "hoo_bicolor":
            ha_raw = r
            oiii_raw = 0.5 * (g + b)
            diff = np.maximum(0.0, r - oiii_raw)
            smooth_diff = gaussian_filter(diff, sigma=max(2.0, self.filter_sigma * 3.0))
            sii_raw = np.maximum(0.0, diff - 0.7 * smooth_diff)
        else:
            oiii_raw = np.maximum(0.0, 1.1 * b + 0.9 * g - 0.55 * r)
            ha_raw = np.maximum(0.0, 1.15 * r - 0.3 * b - 0.2 * g)
            outer_shock = np.maximum(0.0, r - 0.85 * ha_raw)
            smooth_r = gaussian_filter(r, sigma=max(2.0, self.filter_sigma * 2.0))
            sii_raw = np.maximum(0.0, outer_shock + 0.3 * np.maximum(0.0, r - smooth_r))

        oiii = self._normalize_channel(oiii_raw)
        ha = self._normalize_channel(ha_raw)
        sii = self._normalize_channel(sii_raw)

        return SpectralChannels(oiii=oiii, ha=ha, sii=sii)

    def to_sho_rgb(self, channels: SpectralChannels) -> np.ndarray:
        stacked = np.stack([channels.sii, channels.ha, channels.oiii], axis=-1)
        return np.clip(stacked, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _normalize_channel(channel: np.ndarray) -> np.ndarray:
        c_min = float(np.min(channel))
        c_max = float(np.max(channel))
        if c_max - c_min > 1e-7:
            return ((channel - c_min) / (c_max - c_min)).astype(np.float32)
        return np.zeros_like(channel, dtype=np.float32)
