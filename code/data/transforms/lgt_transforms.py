import torch
from dataclasses import dataclass

class BinaryEmbedding(object):
    """
    Transfrom to 2 bit precision inputs using 3 equidistant thresholds; encoded with 3 binary values
    """
    def __init__(self):
        pass
    def __call__(self, X):
        # input x in [0,255] -> normalized in [0,1] and thresholded in with thresholds {0.25, 0.5, 0.75}
        return torch.concatenate([X/255 > torch.ones_like(X)*i*0.25 for i in range(1,4)]).reshape(9,32,32).float()


@dataclass(frozen=True)
class ActiveSpatialCrop:
    margin: int = 0
    min_active_pixels: int = 1

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.ndim != 3:
            raise ValueError(f"Expected frame with shape [C, H, W], got {tuple(frame.shape)}")

        active_mask = torch.any(frame > 0, dim=0)
        if int(active_mask.sum().item()) < int(self.min_active_pixels):
            return frame

        active_rows = torch.where(torch.any(active_mask, dim=1))[0]
        active_cols = torch.where(torch.any(active_mask, dim=0))[0]
        if len(active_rows) == 0 or len(active_cols) == 0:
            return frame

        top = max(int(active_rows[0].item()) - int(self.margin), 0)
        bottom = min(int(active_rows[-1].item()) + int(self.margin), frame.shape[-2] - 1)
        left = max(int(active_cols[0].item()) - int(self.margin), 0)
        right = min(int(active_cols[-1].item()) + int(self.margin), frame.shape[-1] - 1)
        return frame[:, top:bottom + 1, left:right + 1]
