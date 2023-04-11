import torch
from torch.nn.functional import interpolate, pad
from typing import List, Optional


def rescale(x: torch.Tensor, scale: float, mode="area"):
    """
    Rescale 3D tensor using torch.nn.functional.interpolate

    Args:
        x: Tensor (3-6 dimensional)
        scale: Scale factor > 0
        mode: Interpolation mode (see torch.nn.functional.interpolate)

    Returns: scaled tensor
    """
    if x.ndim == 7:
        return torch.stack(
            [torch.stack([interpolate(i, scale_factor=scale, mode=mode, recompute_scale_factor=False) for i in y])
             for y in x])
    elif x.ndim == 6:
        return torch.stack([interpolate(i, scale_factor=scale, mode=mode, recompute_scale_factor=False) for i in x])
    elif x.ndim == 5:
        return interpolate(x, scale_factor=scale, mode=mode, recompute_scale_factor=False)
    elif x.ndim == 4:
        return interpolate(x[None, :, :, :, :], scale_factor=scale, mode=mode, recompute_scale_factor=False)[0]
    elif x.ndim == 3:
        return interpolate(x[None, None, :, :, :], scale_factor=scale, mode=mode, recompute_scale_factor=False)[0, 0]


def crop(x, crop_dist: Optional[List[int]] = None, frac: Optional[float] = None):
    """
    Crop a tensor (either by giving an explicit crop distance or a fraction

    :param x: Tensor
    :param crop_dist: basically the same as negative arguments of torch's pad function
    :param frac: fraction to be cut off at each side
    :return: Cropped tensor
    """
    assert (crop_dist is None) != (frac is None), "Either crop_dist or frac have to hold a value."
    if crop_dist is not None:
        assert len(crop_dist) == 6, f"Need 6 crop values for 3D Scan but got {len(crop_dist)}."
        return x[..., crop_dist[0]:-crop_dist[1] if crop_dist[1] != 0 else None,
               crop_dist[2]:-crop_dist[3] if crop_dist[3] != 0 else None,
               crop_dist[4]:-crop_dist[5] if crop_dist[5] != 0 else None]
    elif frac is not None:
        size = x.shape[-1]
        pixel = int(frac * size)
        return x[..., pixel:size - pixel, pixel:size - pixel, pixel:size - pixel]


def replicate_pad_scale_dim(x, padding):
    l = x.shape[2]
    x = pad(x, padding)
    for i in range(padding[-1]):
        x[:, :, l + i, :, :] = x[:, :, l - 1, :, :, :]
    return x

