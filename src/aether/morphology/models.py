from abc import ABC, abstractmethod
import numpy as np

class BaseMorphologyModel(ABC):
    def __init__(self, inclination_deg: float = 45.0, position_angle_deg: float = 0.0) -> None:
        self.inclination_deg = inclination_deg
        self.position_angle_deg = position_angle_deg

    def get_rotation_matrix(self) -> np.ndarray:
        inc = np.radians(self.inclination_deg)
        pa = np.radians(self.position_angle_deg)
        
        r_z = np.array([
            [np.cos(pa), -np.sin(pa), 0.0],
            [np.sin(pa), np.cos(pa), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        
        r_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(inc), -np.sin(inc)],
            [0.0, np.sin(inc), np.cos(inc)],
        ], dtype=np.float32)
        
        return r_z @ r_x

    def transform_to_intrinsic(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rot = self.get_rotation_matrix()
        xp = x * rot[0, 0] + y * rot[1, 0] + z * rot[2, 0]
        yp = x * rot[0, 1] + y * rot[1, 1] + z * rot[2, 1]
        zp = x * rot[0, 2] + y * rot[1, 2] + z * rot[2, 2]
        return xp, yp, zp

    @abstractmethod
    def evaluate_envelope_radius(self, z_prime: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def compute_density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        pass

class BipolarModel(BaseMorphologyModel):
    def __init__(
        self,
        inclination_deg: float = 45.0,
        position_angle_deg: float = 0.0,
        waist_radius: float = 0.35,
        expansion_factor: float = 1.3,
        lobe_power: float = 1.4,
    ) -> None:
        super().__init__(inclination_deg, position_angle_deg)
        self.waist_radius = waist_radius
        self.expansion_factor = expansion_factor
        self.lobe_power = lobe_power

    def evaluate_envelope_radius(self, z_prime: np.ndarray) -> np.ndarray:
        abs_z = np.abs(z_prime)
        return self.waist_radius * (1.0 + self.expansion_factor * np.power(abs_z, self.lobe_power))

    def compute_density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        xp, yp, zp = self.transform_to_intrinsic(x, y, z)

        r_perp = np.sqrt(xp**2 + yp**2)
        r_env = self.evaluate_envelope_radius(zp)

        shell_sigma = 0.22
        shell_dist = (r_perp - r_env) ** 2
        density_shell = np.exp(-shell_dist / (2.0 * (shell_sigma**2)))

        z_falloff = np.exp(- (zp**2) / 2.5)
        core_boost = np.exp(- (r_perp**2 + zp**2) / (2.0 * (self.waist_radius**2))) * 0.8

        return (density_shell * z_falloff + core_boost).astype(np.float32)

class ToroidalModel(BaseMorphologyModel):
    def __init__(
        self,
        inclination_deg: float = 30.0,
        position_angle_deg: float = 0.0,
        major_radius: float = 0.65,
        minor_radius: float = 0.28,
        axial_scale: float = 0.75,
    ) -> None:
        super().__init__(inclination_deg, position_angle_deg)
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.axial_scale = axial_scale

    def evaluate_envelope_radius(self, z_prime: np.ndarray) -> np.ndarray:
        scaled_z = np.abs(z_prime) / self.axial_scale
        rad_sq = np.maximum(0.0, self.minor_radius**2 - scaled_z**2)
        return self.major_radius + np.sqrt(rad_sq)

    def compute_density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        xp, yp, zp = self.transform_to_intrinsic(x, y, z)

        r_plane = np.sqrt(xp**2 + yp**2)
        dist_ring = np.sqrt((r_plane - self.major_radius)**2 + (zp / self.axial_scale)**2)

        density = np.exp(- (dist_ring**2) / (2.0 * (self.minor_radius**2)))
        central_cavity = 1.0 - np.exp(- (r_plane**2) / 0.05)

        return (density * central_cavity).astype(np.float32)

class EllipsoidalModel(BaseMorphologyModel):
    def __init__(
        self,
        inclination_deg: float = 40.0,
        position_angle_deg: float = 0.0,
        semi_axes: tuple[float, float, float] = (0.9, 0.7, 1.1),
        shell_thickness: float = 0.25,
    ) -> None:
        super().__init__(inclination_deg, position_angle_deg)
        self.semi_axes = semi_axes
        self.shell_thickness = shell_thickness

    def evaluate_envelope_radius(self, z_prime: np.ndarray) -> np.ndarray:
        a, b, c = self.semi_axes
        z_norm = np.clip(np.abs(z_prime) / c, 0.0, 1.0)
        return np.sqrt(a * b) * np.sqrt(1.0 - z_norm**2)

    def compute_density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        xp, yp, zp = self.transform_to_intrinsic(x, y, z)

        a, b, c = self.semi_axes
        ellip_dist = np.sqrt((xp / a)**2 + (yp / b)**2 + (zp / c)**2)
        shell = np.exp(- ((ellip_dist - 0.85)**2) / (2.0 * (self.shell_thickness**2)))

        return shell.astype(np.float32)
