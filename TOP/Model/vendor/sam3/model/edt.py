# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""Windows EDT adapter; keeps the original function name for existing call sites."""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def edt_triton(data: torch.Tensor):
    """Exact Euclidean distance to an in-image zero pixel, returned as float32.

    Original Triton uses 1e18 for infinite squared distance. For a completely
    foreground image, return 1e9 rather than SciPy's implicit exterior point.
    """
    assert data.dim() == 3
    arrays = data.detach().to(device="cpu", dtype=torch.bool).numpy()
    output = np.empty(arrays.shape, dtype=np.float32)
    for i, array in enumerate(arrays):
        if array.all():
            output[i].fill(1e9)
        else:
            output[i] = distance_transform_edt(array)
    return torch.from_numpy(output).to(data.device)
