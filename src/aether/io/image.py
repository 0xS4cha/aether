from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

def load_image(path: str | Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def save_image(path: str | Path, image: np.ndarray) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    arr = image[..., 0] if (image.ndim == 3 and image.shape[2] == 1) else image
    if arr.dtype != np.uint8:
        clipped = np.clip(arr, 0.0, 1.0) * 255.0
        uint_arr = clipped.astype(np.uint8)
    else:
        uint_arr = arr
    img = Image.fromarray(uint_arr, mode="L" if uint_arr.ndim == 2 else "RGB")
    img.save(dest)

def resize_image(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    arr = image[..., 0] if (image.ndim == 3 and image.shape[2] == 1) else image
    if arr.dtype != np.uint8:
        clipped = np.clip(arr, 0.0, 1.0) * 255.0
        uint_arr = clipped.astype(np.uint8)
    else:
        uint_arr = arr
    img = Image.fromarray(uint_arr, mode="L" if uint_arr.ndim == 2 else "RGB")
    resized = img.resize(target_size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0

def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb
    return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]

def filter_stars(rgb: np.ndarray, threshold: float = 0.90, filter_size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    gray = to_grayscale(rgb)
    smooth = median_filter(gray, size=filter_size)
    star_diff = np.maximum(0.0, gray - smooth)
    star_mask = (star_diff > (1.0 - threshold)) & (gray > 0.5)
    
    cleaned_rgb = rgb.copy()
    for c in range(3):
        channel_smooth = median_filter(rgb[..., c], size=filter_size)
        cleaned_rgb[star_mask, c] = channel_smooth[star_mask]
        
    return cleaned_rgb, star_mask
