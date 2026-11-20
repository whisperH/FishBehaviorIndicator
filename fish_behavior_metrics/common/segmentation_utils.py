from __future__ import annotations

"""SAM 分割掩膜工具。

标准输入中的 segmentation 支持两种 counts：
1. COCO 压缩 RLE 字符串；
2. 普通 run-length 数组，例如 [background_count, foreground_count, ...]。

本模块不改动输入记录，只负责把 RLE 还原为二值掩膜并计算几何中心。
"""

import numpy as np

try:
    from pycocotools import mask as mask_utils
except ImportError:  # 普通游程数组和轮廓路径不依赖 pycocotools。
    mask_utils = None


def decode_compressed_rle(text: str) -> list[int]:
    """解码 COCO 压缩 RLE 的变长整数编码。"""
    counts: list[int] = []
    position = 0
    while position < len(text):
        value = 0
        shift = 0
        more = 1
        while more:
            # 低 5 位保存数据，第 6 位表示后面是否还有字节。
            character = ord(text[position]) - 48
            value |= (character & 0x1F) << (5 * shift)
            more = character & 0x20
            position += 1
            shift += 1
            if not more and (character & 0x10):
                value |= -1 << (5 * shift)
        if counts:
            value += counts[-1]
        counts.append(value)
    return counts


def cumulative_counts(segmentation: dict) -> tuple[int, int, np.ndarray]:
    """将 run-length 数组转换为累计边界，并验证覆盖完整图像。"""
    height, width = [int(value) for value in segmentation["size"]]
    total = height * width
    counts = segmentation["counts"]
    if isinstance(counts, str):
        cumulative = np.asarray(decode_compressed_rle(counts), dtype=np.int64)
    else:
        raw = np.asarray(counts, dtype=np.int64)
        # 有些上游直接给累计边界，有些给每段长度，需要自动区分。
        is_cumulative = (
            len(raw) > 0
            and np.all(raw >= 0)
            and np.all(np.diff(raw) >= 0)
            and int(raw[-1]) == total
        )
        cumulative = raw if is_cumulative else np.cumsum(raw)
    if len(cumulative) == 0 or int(cumulative[-1]) != total:
        raise ValueError("Invalid RLE segmentation")
    return height, width, cumulative


def rle_to_mask(segmentation: dict) -> np.ndarray:
    """把标准 segmentation 转成 HxW 的二值掩膜。"""
    if isinstance(segmentation["counts"], str):
        # pycocotools 负责处理 COCO 压缩 RLE 的符号扩展和游程还原。
        if mask_utils is None:
            raise ImportError(
                "pycocotools is required to decode compressed RLE segmentation"
            )
        decoded = mask_utils.decode(
            {
                "size": [int(value) for value in segmentation["size"]],
                "counts": segmentation["counts"].encode("ascii"),
            }
        )
        return np.asarray(decoded, dtype=np.uint8)
    height, width, cumulative = cumulative_counts(segmentation)
    flat = np.zeros(height * width, dtype=np.uint8)
    start = 0
    for run_index, end in enumerate(cumulative):
        end = int(end)
        # RLE 从背景段开始，奇数段为前景。
        if run_index % 2 == 1 and end > start:
            flat[start:end] = 1
        start = end
    return flat.reshape(height, width)


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """返回掩膜全部前景像素的算术中心 (x_mean, y_mean)。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Segmentation mask contains no foreground pixels")
    return float(xs.mean()), float(ys.mean())
