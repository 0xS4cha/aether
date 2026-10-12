from aether.morphology.models import (
    BaseMorphologyModel,
    BipolarModel,
    ToroidalModel,
    EllipsoidalModel,
)
from aether.morphology.deprojector import MorphologicalDeprojector
from aether.morphology.cinematics import CameraPathGenerator

def create_morphology_model(
    model_type: str = "bipolar",
    inclination_deg: float = 45.0,
    position_angle_deg: float = 0.0,
) -> BaseMorphologyModel:
    norm = model_type.lower()
    if norm == "bipolar":
        return BipolarModel(inclination_deg=inclination_deg, position_angle_deg=position_angle_deg)
    elif norm == "toroidal":
        return ToroidalModel(inclination_deg=inclination_deg, position_angle_deg=position_angle_deg)
    elif norm == "ellipsoidal":
        return EllipsoidalModel(inclination_deg=inclination_deg, position_angle_deg=position_angle_deg)
    else:
        raise ValueError(f"Unknown morphology model type: {model_type}")

__all__ = [
    "BaseMorphologyModel",
    "BipolarModel",
    "ToroidalModel",
    "EllipsoidalModel",
    "MorphologicalDeprojector",
    "CameraPathGenerator",
    "create_morphology_model",
]
