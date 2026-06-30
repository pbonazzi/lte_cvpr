from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

TONIC_EVENT_DTYPE = np.dtype(
    [
        ("x", np.int16),
        ("y", np.int16),
        ("t", np.int64),
        ("p", np.int8),
    ]
)
SECONDS_TO_MICROSECONDS = 1_000_000.0


def _require_tonic():
    try:
        import tonic
    except ImportError as exc:
        raise ImportError(
            "tonic is required for event augmentations. Install it first, for example via "
            "`pip install tonic` or by recreating the project environment."
        ) from exc
    return tonic


def infer_tonic_timestamp_scale(events: np.ndarray) -> float:
    array = np.asarray(events)
    if array.ndim != 2 or array.shape[1] != 4 or len(array) == 0:
        return 1.0

    timestamps = array[:, 2]
    # Most tonic transforms expect integer timestamps. Our training scripts use
    # *_us hyperparameters, while some datasets store timestamps in seconds as
    # small floats (for example N-Cars). In that case we convert seconds to
    # integer microseconds before calling tonic, then scale back afterwards.
    if np.allclose(timestamps, np.rint(timestamps), atol=1e-6, rtol=0.0):
        return 1.0
    return SECONDS_TO_MICROSECONDS


def to_tonic_structured(events: np.ndarray, timestamp_scale: float = 1.0) -> np.ndarray:
    array = np.asarray(events)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"Expected event array with shape [N, 4], got {array.shape}")

    structured = np.empty(len(array), dtype=TONIC_EVENT_DTYPE)
    if len(array) == 0:
        return structured

    structured["x"] = np.rint(array[:, 0]).astype(np.int16, copy=False)
    structured["y"] = np.rint(array[:, 1]).astype(np.int16, copy=False)
    structured["t"] = np.rint(array[:, 2] * float(timestamp_scale)).astype(np.int64, copy=False)
    structured["p"] = np.rint(array[:, 3]).astype(np.int8, copy=False)
    return structured


def from_tonic_structured(events: np.ndarray, timestamp_scale: float = 1.0) -> np.ndarray:
    array = np.asarray(events)
    if array.dtype.names is None:
        if array.ndim != 2 or array.shape[1] != 4:
            raise ValueError(f"Expected event array with shape [N, 4], got {array.shape}")
        output = array.astype(np.float32, copy=False)
        if timestamp_scale != 1.0 and len(output) > 0:
            output = output.copy()
            output[:, 2] /= float(timestamp_scale)
        return output

    return np.stack(
        [
            array["x"].astype(np.float32, copy=False),
            array["y"].astype(np.float32, copy=False),
            (array["t"].astype(np.float32, copy=False) / float(timestamp_scale)),
            array["p"].astype(np.float32, copy=False),
        ],
        axis=1,
    )


def _to_tonic_sensor_size(sensor_size: Optional[Sequence[int]]) -> Optional[Tuple[int, int, int]]:
    if sensor_size is None:
        return None
    if len(sensor_size) != 2:
        raise ValueError(f"Expected sensor_size=(height, width), got {sensor_size}")
    height, width = int(sensor_size[0]), int(sensor_size[1])
    return (width, height, 2)


class TonicEventTransformAdapter:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, events: np.ndarray) -> np.ndarray:
        timestamp_scale = infer_tonic_timestamp_scale(events)
        structured = to_tonic_structured(events, timestamp_scale=timestamp_scale)
        transformed = self.transform(structured)
        return from_tonic_structured(transformed, timestamp_scale=timestamp_scale)


class EventTransformCompose:
    def __init__(self, transforms: Iterable):
        self.transforms = [transform for transform in transforms if transform is not None]

    def __call__(self, events: np.ndarray) -> np.ndarray:
        output = np.asarray(events, dtype=np.float32)
        for transform in self.transforms:
            output = transform(output)
        return output


@dataclass(frozen=True)
class SpatialJitter:
    max_shift: int = 1
    sensor_size: Optional[Sequence[int]] = None

    def __call__(self, events: np.ndarray) -> np.ndarray:
        array = np.asarray(events, dtype=np.float32)
        if len(array) == 0 or self.max_shift <= 0:
            return array.copy()

        tonic = _require_tonic()
        tonic_sensor_size = _to_tonic_sensor_size(self.sensor_size)
        if tonic_sensor_size is None:
            raise ValueError("tonic SpatialJitter requires sensor_size=(height, width)")

        variance = float(self.max_shift * (self.max_shift + 1) / 3.0)
        transform = TonicEventTransformAdapter(
            tonic.transforms.SpatialJitter(
                sensor_size=tonic_sensor_size,
                var_x=variance,
                var_y=variance,
                sigma_xy=0.0,
                clip_outliers=True,
            )
        )
        return transform(array)


@dataclass(frozen=True)
class RandomFlipLR:
    p: float = 0.5
    sensor_width: int = 240
    sensor_size: Optional[Sequence[int]] = None

    def __call__(self, events: np.ndarray) -> np.ndarray:
        array = np.asarray(events, dtype=np.float32)
        if len(array) == 0 or self.p <= 0.0:
            return array.copy()

        tonic = _require_tonic()
        tonic_sensor_size = _to_tonic_sensor_size(self.sensor_size)
        if tonic_sensor_size is None:
            tonic_sensor_size = (int(self.sensor_width), 1, 2)

        transform = TonicEventTransformAdapter(
            tonic.transforms.RandomFlipLR(sensor_size=tonic_sensor_size, p=float(self.p))
        )
        return transform(array)


@dataclass(frozen=True)
class Denoise:
    filter_time: float

    def __call__(
        self,
        events: np.ndarray,
        sensor_size: Optional[Sequence[int]] = None,
        return_mask: bool = False,
    ) -> np.ndarray:
        array = np.asarray(events, dtype=np.float32)
        if len(array) == 0:
            empty = array.copy()
            if return_mask:
                return empty, np.zeros(0, dtype=bool)
            return empty

        tonic = _require_tonic()
        timestamp_scale = infer_tonic_timestamp_scale(array)
        transform = TonicEventTransformAdapter(
            tonic.transforms.Denoise(filter_time=float(self.filter_time))
        )
        del sensor_size

        filtered = transform(array)
        if not return_mask:
            return filtered

        keep_mask = np.zeros(len(array), dtype=bool)
        if len(filtered) == 0:
            return filtered, keep_mask

        filtered_index = 0
        time_tolerance = 0.5 / float(timestamp_scale)
        for original_index, event in enumerate(array):
            if filtered_index >= len(filtered):
                break
            candidate = filtered[filtered_index]
            if (
                event[0] == candidate[0]
                and event[1] == candidate[1]
                and event[3] == candidate[3]
                and abs(float(event[2]) - float(candidate[2])) <= time_tolerance
            ):
                keep_mask[original_index] = True
                filtered_index += 1

        return filtered, keep_mask


# Tensor-level augmentations (for cached spike tensors)
import torch
import random


class TensorEventTransformCompose:
    """Compose multiple tensor-level augmentations."""
    def __init__(self, transforms: Iterable):
        self.transforms = [transform for transform in transforms if transform is not None]

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        output = tensor
        for transform in self.transforms:
            output = transform(output)
        return output


@dataclass(frozen=True)
class TensorRandomFlipLR:
    """Horizontal flip augmentation for spike tensor (B, H, W)."""
    p: float = 0.5
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or random.random() >= self.p:
            return tensor
        return torch.flip(tensor, dims=[-1])


@dataclass(frozen=True)
class TensorSpatialJitter:
    """Spatial jitter augmentation for spike tensor by random shifting.
    Out-of-bounds pixels are set to 0."""
    max_shift: int = 1
    sensor_size: Optional[Sequence[int]] = None
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.max_shift <= 0 or tensor.shape[0] == 0:
            return tensor
        
        shift_x = random.randint(-self.max_shift, self.max_shift)
        shift_y = random.randint(-self.max_shift, self.max_shift)
        
        if shift_x == 0 and shift_y == 0:
            return tensor
        
        result = tensor
        
        # Shift in x (width dimension)
        if shift_x != 0:
            if shift_x > 0:
                result = torch.cat([torch.zeros_like(result[:, :, :shift_x]), result[:, :, :-shift_x]], dim=-1)
            else:
                result = torch.cat([result[:, :, -shift_x:], torch.zeros_like(result[:, :, :(-shift_x)])], dim=-1)
        
        # Shift in y (height dimension)
        if shift_y != 0:
            if shift_y > 0:
                result = torch.cat([torch.zeros_like(result[:, :shift_y, :]), result[:, :-shift_y, :]], dim=-2)
            else:
                result = torch.cat([result[:, -shift_y:, :], torch.zeros_like(result[:, :(-shift_y), :])], dim=-2)
        
        return result

