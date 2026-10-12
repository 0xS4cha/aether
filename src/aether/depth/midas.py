import numpy as np
import torch
import torch.nn.functional as F
from aether.depth.base import BaseDepthEstimator

class MiDaSDepthEstimator(BaseDepthEstimator):
    def __init__(self, model_type: str = "MiDaS_small", device: str = "auto") -> None:
        super().__init__(device=device)
        self.model_type = model_type
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type in ("DPT_Large", "DPT_Hybrid"):
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def estimate(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        uint_img = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        
        batch = self.transform(uint_img).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(batch)
            if prediction.ndim == 3:
                prediction = prediction.unsqueeze(1)
            
            resized = F.interpolate(
                prediction,
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            depth_map = resized.cpu().numpy()
            
        return self.normalize_depth(depth_map)
