# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""Windows adapter: 8-connected binary components, without compiled CUDA extensions."""
import cv2
import numpy as np
import torch


def connected_components(input_tensor: torch.Tensor):
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(1)
    assert input_tensor.dim() == 4 and input_tensor.shape[1] == 1
    arrays = input_tensor.detach().to(device="cpu", dtype=torch.bool).numpy()[:, 0]
    labels = np.empty(arrays.shape, dtype=np.int32)
    counts = np.empty(arrays.shape, dtype=np.int32)
    for i, array in enumerate(arrays):
        _, lab, stats, _ = cv2.connectedComponentsWithStats(
            np.ascontiguousarray(array, dtype=np.uint8), connectivity=8
        )
        areas = stats[:, cv2.CC_STAT_AREA].copy()
        areas[0] = 0
        labels[i] = lab
        counts[i] = areas[lab]
    return (torch.from_numpy(labels[:, None]).to(input_tensor.device),
            torch.from_numpy(counts[:, None]).to(input_tensor.device))
