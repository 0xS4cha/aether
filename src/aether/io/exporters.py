from pathlib import Path
import numpy as np
from PIL import Image
import pyvista as pv

def export_point_cloud(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    poly = pv.PolyData(points)
    poly.point_data["RGB"] = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
    poly.save(str(dest))

def export_mesh(path: str | Path, mesh: pv.PolyData) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(dest))

def export_volume(path: str | Path, volume: pv.ImageData) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    volume.save(str(dest))

def export_depth_map(path: str | Path, depth: np.ndarray) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix.lower() == ".npy":
        np.save(dest, depth)
    else:
        norm = np.clip(depth, 0.0, 1.0)
        uint16_depth = (norm * 65535.0).astype(np.uint16)
        img = Image.fromarray(uint16_depth)
        img.save(dest)
