from pathlib import Path
import numpy as np
from PIL import Image
import pyvista as pv

class CameraPathGenerator:
    @staticmethod
    def generate_orbit(
        radius: float = 260.0,
        elevation_deg: float = 25.0,
        num_frames: int = 48,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        elev = np.radians(elevation_deg)
        angles = np.linspace(0.0, 2.0 * np.pi, num_frames, endpoint=False)
        trajectory = []

        focal_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        view_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        for a in angles:
            pos_x = radius * np.cos(a) * np.cos(elev)
            pos_y = radius * np.sin(a) * np.cos(elev)
            pos_z = radius * np.sin(elev)
            position = np.array([pos_x, pos_y, pos_z], dtype=np.float32)
            trajectory.append((position, focal_point, view_up))

        return trajectory

    @staticmethod
    def generate_fly_through(
        start_dist: float = 320.0,
        end_dist: float = -280.0,
        num_frames: int = 48,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        trajectory = []
        z_vals = np.linspace(start_dist, end_dist, num_frames, dtype=np.float32)
        t_norm = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)

        view_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        for z, t in zip(z_vals, t_norm):
            x = 35.0 * np.sin(2.0 * np.pi * t)
            y = 20.0 * np.cos(2.0 * np.pi * t)
            position = np.array([x, y, z], dtype=np.float32)
            focal = np.array([0.0, 0.0, z - 80.0], dtype=np.float32)
            trajectory.append((position, focal, view_up))

        return trajectory

    @staticmethod
    def render_animation(
        plotter: pv.Plotter,
        trajectory: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        output_path: str | Path,
        fps: int = 24,
    ) -> None:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        frames = []
        for pos, focal, up in trajectory:
            plotter.camera_position = [pos, focal, up]
            frame_arr = plotter.screenshot(return_img=True)
            frames.append(Image.fromarray(frame_arr))

        if frames:
            duration_ms = int(1000.0 / fps)
            if dest.suffix.lower() == ".gif":
                frames[0].save(
                    dest,
                    save_all=True,
                    append_images=frames[1:],
                    duration=duration_ms,
                    loop=0,
                )
            else:
                frames[0].save(dest)
