"""鱼体分割与角度指标。

指标定义：对鱼体分割掩膜的所有前景像素执行 PCA，以最大特征值对应的特征
向量作为鱼体主轴。
- 沿主轴统计宽度轮廓，作为鱼头方向的备用判断；
- 可读取 SAM3 鱼头掩膜，选择与鱼体端点匹配的鱼头并修正头尾方向；
- fish_body_angle / angle_horizontal_deg 为鱼体主轴与水平线的 0–90° 夹角。
- orientation_deg / directed_angle_deg 保留头尾方向，用于箭头显示。

主要数据流：
标准 data 记录 -> SAM RLE 掩膜 -> 连通域清理 -> PCA 主轴 ->
角度/伸长率 -> 标准行为结果。

公开入口 calculate_fish_body_angle 只依赖标准 data 中的 SAM RLE，不运行
视频、SAM3 模型或页面代码。文件中的 process_video 另外提供 MOT 框提示
SAM3 的离线分割流程，并在同一次输出中整合可选的 SAM3 鱼头掩膜。
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import io
import json
import math
import os
import re
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT / "datasets" / "angle.mp4"
DEFAULT_MOT = ROOT / "datasets" / "angle"
DEFAULT_CHECKPOINT = ROOT / "models" / "sam3.pt"
DEFAULT_BPE = ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"
DEFAULT_OUTPUT = ROOT / "results" / "angle"
DEFAULT_HEAD_MASKS = DEFAULT_OUTPUT / "head_mask_shapes.json"
DEFAULT_VIEWER = ROOT / "visualization" / "angle_viewer.html"


def preload_windows_runtime() -> None:
    """在导入 torch 前预加载系统 VC 运行库，避免优先选中 Anaconda 的旧 DLL。"""
    if os.name != "nt":
        return
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        path = system_root / "System32" / name
        if path.is_file():
            try:
                ctypes.WinDLL(str(path))
            except OSError:
                pass


preload_windows_runtime()

import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

from common.segmentation_utils import rle_to_mask


def resolve_device(requested: str) -> str:
    """解析 auto/cpu/cuda 设备选项，并在 CUDA 不可用时给出明确错误。"""
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is false."
            )
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def autocast_context(device: str, enabled: bool):
    """按设备和开关创建 CUDA FP16 autocast 上下文，CPU 下返回空上下文。"""
    if device == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def resolve_mot_file(path: Path) -> Path:
    """定位 MOT 跟踪文件，兼容目录、gt/gt.txt 和 gt.txt 三种形式。"""
    if path.is_file():
        return path
    candidates = (path / "gt" / "gt.txt", path / "gt.txt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"MOT annotation file not found under: {path}")


def load_mot(path: Path) -> pd.DataFrame:
    """读取并清洗 MOT 跟踪表，只保留有效框和正置信度记录。"""
    mot_path = resolve_mot_file(path)
    columns = [
        "frame",
        "track_id",
        "x",
        "y",
        "w",
        "h",
        "mark",
        "class_id",
        "confidence",
    ]
    if mot_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(mot_path) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name.lower().endswith("gt.txt") and not name.endswith("/")
            ]
            if not candidates:
                raise FileNotFoundError(f"gt.txt not found in archive: {mot_path}")
            data = archive.read(candidates[0])
        frame = pd.read_csv(io.BytesIO(data), header=None, names=columns)
    else:
        frame = pd.read_csv(mot_path, header=None, names=columns)

    for name in ("frame", "track_id"):
        frame[name] = frame[name].astype(int)
    for name in ("x", "y", "w", "h", "confidence"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna(subset=["x", "y", "w", "h"])
    # 小于 2 像素的框无法提供稳定的 SAM 提示区域。
    frame = frame[(frame["w"] > 1) & (frame["h"] > 1)]
    frame["confidence"] = frame["confidence"].fillna(1.0)
    frame = frame[frame["confidence"] > 0.0]
    return frame.sort_values(["frame", "track_id"]).reset_index(drop=True)


def read_video_info(video_path: Path) -> dict[str, float | int]:
    """读取视频宽高、帧率和总帧数，并校验视频可打开。"""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    info: dict[str, float | int] = {
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
    }
    capture.release()
    if info["fps"] <= 0:
        # 少数容器不写 FPS，后续采样按 30 FPS 的保守默认值处理。
        info["fps"] = 30.0
    if info["width"] <= 0 or info["height"] <= 0:
        raise RuntimeError(f"Invalid video dimensions: {video_path}")
    return info


def clip_box(
    row: object, width: int, height: int
) -> tuple[float, float, float, float] | None:
    """把 MOT 框裁剪到视频边界，返回有效的 xywh 框或 None。"""
    x1 = max(0.0, min(float(width), float(row.x)))
    y1 = max(0.0, min(float(height), float(row.y)))
    x2 = max(x1 + 1.0, min(float(width), float(row.x + row.w)))
    y2 = max(y1 + 1.0, min(float(height), float(row.y + row.h)))
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < 2.0 or box_height < 2.0:
        return None
    return x1, y1, box_width, box_height


def normalized_box(
    box_xywh: tuple[float, float, float, float], width: int, height: int
) -> list[float]:
    """将像素 xywh 框转换为 SAM3 期望的归一化 cx/cy/w/h。"""
    x, y, box_width, box_height = box_xywh
    return [
        (x + box_width / 2.0) / width,
        (y + box_height / 2.0) / height,
        box_width / width,
        box_height / height,
    ]


def box_iou_xyxy(first: np.ndarray, second: np.ndarray) -> float:
    """计算两个 xyxy 框的交并比 IoU。"""
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def clean_mask(mask: np.ndarray, minimum_area: int) -> tuple[np.ndarray | None, int]:
    """清理 SAM 二值掩膜，保留与最大连通域同量级的前景区域。

    先做闭运算填补小孔，再删除面积过小的孤立连通域。返回清理后的掩膜和
    前景面积；面积不足时返回 (None, area)。
    """
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim > 2:
        binary = np.squeeze(binary)
    if binary.shape[0] == 0 or binary.shape[1] == 0:
        return None, 0
    binary = (binary > 0).astype(np.uint8)
    # 闭运算只连接相邻小缺口，不改变鱼体的整体朝向。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return None, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    area = int(areas[largest - 1])
    if area < minimum_area:
        return None, area
    # 保留最大连通域以及至少达到其面积 8% 的显著连通域。
    keep_threshold = max(minimum_area, int(round(area * 0.08)))
    keep_labels = np.flatnonzero(areas >= keep_threshold) + 1
    merged = np.isin(labels, keep_labels)
    merged_area = int(merged.sum())
    if merged_area < minimum_area:
        return None, merged_area
    return merged.astype(np.uint8), merged_area


def mask_polygons(mask: np.ndarray) -> list[list[list[int]]]:
    """把二值掩膜转换为适合 JSON 存储的外轮廓多边形。"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    total_area = max(1, int(mask.sum()))
    polygons: list[list[list[int]]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        # 过滤分割噪点，并控制轮廓点数量以免输出文件过大。
        if area < max(12.0, total_area * 0.008):
            continue
        epsilon = max(1.2, 0.003 * cv2.arcLength(contour, True))
        approximated = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approximated) >= 3:
            polygons.append(approximated.astype(int).tolist())
    return polygons


def select_best_mask(
    state: dict,
    prompt_box_xywh: tuple[float, float, float, float],
    minimum_area: int,
) -> tuple[np.ndarray | None, float, float]:
    """从 SAM3 的多个候选掩膜中选择与 MOT 提示框最匹配的一个。

    评分由提示框 IoU、模型置信度和面积异常惩罚组成。mask=None 表示没有
    候选通过面积、重叠或清理检查。
    """
    masks = state.get("masks")
    scores = state.get("scores")
    if masks is None or scores is None or len(scores) == 0:
        return None, 0.0, 0.0

    prompt_x, prompt_y, prompt_w, prompt_h = prompt_box_xywh
    prompt_xyxy = np.array(
        [prompt_x, prompt_y, prompt_x + prompt_w, prompt_y + prompt_h],
        dtype=np.float32,
    )
    prompt_area = prompt_w * prompt_h
    best: tuple[float, np.ndarray | None, float, float] | None = None

    masks_cpu = masks.detach().float().cpu().numpy()
    scores_cpu = scores.detach().float().cpu().numpy()
    for index in range(masks_cpu.shape[0]):
        mask, area = clean_mask(masks_cpu[index], minimum_area)
        if mask is None:
            continue
        ys, xs = np.nonzero(mask)
        mask_box = np.array(
            [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32
        )
        overlap = box_iou_xyxy(mask_box, prompt_xyxy)
        # 与提示框几乎不重叠的掩膜通常来自其他鱼或背景。
        if overlap < 0.05:
            continue
        area_ratio = area / max(prompt_area, 1.0)
        area_penalty = 0.0
        # 掩膜远大于提示框时更不可信；过小的掩膜保留较小惩罚。
        if area_ratio > 2.5:
            area_penalty = 1.0
        elif area_ratio < 0.03:
            area_penalty = 0.5
        quality = overlap + 0.25 * float(scores_cpu[index]) - area_penalty
        if best is None or quality > best[0]:
            best = (quality, mask, float(scores_cpu[index]), overlap)

    if best is None:
        return None, 0.0, 0.0
    return best[1], best[2], best[3]


def normalize_angle_deg(value: float) -> float:
    """把角度规范化到 [0, 360)，并把浮点误差产生的 360° 转为 0°。"""
    normalized = float(value) % 360.0
    if math.isclose(normalized, 0.0, abs_tol=1e-9) or math.isclose(
        normalized, 360.0, abs_tol=1e-9
    ):
        return 0.0
    return normalized


def horizontal_angle(directed: float) -> float:
    """把带方向角折叠为鱼体主轴与水平线的 0–90° 无符号夹角。"""
    half_turn = directed % 180.0
    return min(half_turn, 180.0 - half_turn)


def fill_polygons(
    polygons: list[list[list[int]]],
    width: int,
    height: int,
) -> np.ndarray:
    """把 JSON 多边形恢复为指定视频尺寸的二值掩膜。"""
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
    return mask


def body_mask_axis(body_mask: np.ndarray) -> dict[str, object] | None:
    """计算头尾匹配所需的 PCA 主轴、端点和两端宽度。"""
    ys, xs = np.nonzero(body_mask)
    if len(xs) < 30:
        return None
    points = np.column_stack((xs, ys)).astype(np.float64)
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major_vector = eigenvectors[:, order[0]]
    minor_vector = eigenvectors[:, order[1]]
    projected = centered @ major_vector
    perpendicular = centered @ minor_vector
    low = float(np.percentile(projected, 2.0))
    high = float(np.percentile(projected, 98.0))
    span = high - low
    if span < 2.0:
        return None
    normalized = np.clip((projected - low) / span, 0.0, 0.999999)
    bins = 20
    bin_index = (normalized * bins).astype(int)
    widths = np.full(bins, np.nan, dtype=np.float64)
    for index in range(bins):
        selected = perpendicular[bin_index == index]
        if selected.size >= 3:
            low_width, high_width = np.percentile(selected, [10.0, 90.0])
            widths[index] = float(high_width - low_width)
    left_width = (
        float(np.nanmean(widths[:6]))
        if np.isfinite(widths[:6]).any()
        else 0.0
    )
    right_width = (
        float(np.nanmean(widths[-6:]))
        if np.isfinite(widths[-6:]).any()
        else 0.0
    )
    return {
        "center": center,
        "major": major_vector,
        "low": low,
        "high": high,
        "span": span,
        "low_point": center + major_vector * low,
        "high_point": center + major_vector * high,
        "left_width": left_width,
        "right_width": right_width,
    }


def choose_head_candidate(
    body_mask: np.ndarray,
    axis: dict[str, object],
    head_samples: list[dict],
    width: int,
    height: int,
) -> dict[str, object] | None:
    """从同帧鱼头掩膜中选择与鱼体一个端点最匹配的候选。"""
    center = np.asarray(axis["center"], dtype=np.float64)
    major = np.asarray(axis["major"], dtype=np.float64)
    span = float(axis["span"])
    best: tuple[float, dict[str, object]] | None = None
    for sample in head_samples:
        head_mask = fill_polygons(sample.get("polygons", []), width, height)
        head_area = int(head_mask.sum())
        if head_area < 30:
            continue
        intersection = int(np.logical_and(body_mask, head_mask).sum())
        overlap = intersection / head_area
        if overlap < 0.25:
            continue
        ys, xs = np.nonzero(head_mask)
        head_center = np.asarray([xs.mean(), ys.mean()], dtype=np.float64)
        projection = float((head_center - center) @ major)
        endpoint_distance = min(
            abs(projection - float(axis["low"])),
            abs(projection - float(axis["high"])),
        )
        endpoint_score = float(
            np.clip(1.0 - endpoint_distance / max(1.0, span * 0.45), 0.0, 1.0)
        )
        score = (
            0.50 * overlap
            + 0.30 * endpoint_score
            + 0.20 * float(sample.get("score", 0.0))
        )
        candidate = {
            "sample": sample,
            "score": score,
            "overlap": overlap,
            "head_is_high": projection >= 0.0,
            "head_center_x": float(head_center[0]),
            "head_center_y": float(head_center[1]),
        }
        if best is None or score > best[0]:
            best = (score, candidate)
    return best[1] if best else None


def apply_head_direction(
    rows: list[dict[str, object]],
    mask_samples: list[dict[str, object]],
    head_payload: dict | None,
    width: int,
    height: int,
) -> dict[str, int]:
    """用预计算的 SAM3 鱼头掩膜修正头尾方向，失败时保留宽度轮廓结果。"""
    row_by_key = {
        (int(row["frame"]), int(row["track_id"])): row
        for row in rows
        if str(row.get("valid", "")).lower() == "true"
    }
    for row in row_by_key.values():
        row["direction_source"] = "width_profile_fallback"
    for sample in mask_samples:
        sample["direction_source"] = "width_profile_fallback"

    valid_observations = len(row_by_key)
    matched = 0
    changed = 0
    if head_payload is None:
        return {
            "valid_observations": valid_observations,
            "matched": 0,
            "fallback": valid_observations,
            "changed": 0,
        }

    head_samples_by_frame: dict[int, list[dict]] = {}
    for sample in head_payload.get("samples", []):
        head_samples_by_frame.setdefault(int(sample["frame"]), []).append(sample)

    for sample in mask_samples:
        key = (int(sample["frame"]), int(sample["track_id"]))
        row = row_by_key.get(key)
        if row is None:
            continue
        body_mask = fill_polygons(sample.get("polygons", []), width, height)
        if not body_mask.any():
            continue
        axis = body_mask_axis(body_mask)
        if axis is None:
            continue
        match = choose_head_candidate(
            body_mask,
            axis,
            head_samples_by_frame.get(key[0], []),
            width,
            height,
        )
        if match is None:
            continue

        if bool(match["head_is_high"]):
            head_point = np.asarray(axis["high_point"], dtype=np.float64)
            selected_tail_point = np.asarray(axis["low_point"], dtype=np.float64)
            head_vector = np.asarray(axis["major"], dtype=np.float64)
            head_width = float(axis["right_width"])
            tail_width = float(axis["left_width"])
        else:
            head_point = np.asarray(axis["low_point"], dtype=np.float64)
            selected_tail_point = np.asarray(axis["high_point"], dtype=np.float64)
            head_vector = -np.asarray(axis["major"], dtype=np.float64)
            head_width = float(axis["left_width"])
            tail_width = float(axis["right_width"])

        directed = normalize_angle_deg(
            math.degrees(math.atan2(float(head_vector[1]), float(head_vector[0])))
        )
        previous = float(
            row.get("directed_angle_deg")
            or row.get("orientation_deg")
            or 0.0
        )
        if abs(((directed - previous + 180.0) % 360.0) - 180.0) > 90.0:
            changed += 1
        matched += 1
        updates: dict[str, object] = {
            "angle_horizontal_deg": horizontal_angle(directed),
            "orientation_deg": directed,
            "directed_angle_deg": directed,
            "head_x": float(head_point[0]),
            "head_y": float(head_point[1]),
            "tail_x": float(selected_tail_point[0]),
            "tail_y": float(selected_tail_point[1]),
            "head_width": head_width,
            "tail_width": tail_width,
            "direction_source": "sam3_head_mask",
            "head_match_score": float(match["score"]),
            "head_match_overlap": float(match["overlap"]),
        }
        row.update(updates)
        sample.update(updates)

    return {
        "valid_observations": valid_observations,
        "matched": matched,
        "fallback": valid_observations - matched,
        "changed": changed,
    }


def replace_script_data(html: str, element_id: str, text: str) -> str:
    """替换角度页面中指定 id 的 Base64 数据块。"""
    encoded = base64.b64encode(text.encode("utf-8-sig")).decode("ascii")
    pattern = re.compile(
        rf'(<script id="{re.escape(element_id)}" type="text/plain">).*?(</script>)',
        re.DOTALL,
    )
    replaced, count = pattern.subn(rf"\g<1>{encoded}\2", html)
    if count != 1:
        raise RuntimeError(f"Expected one {element_id} script block, found {count}")
    return replaced


def update_viewer_data(viewer_path: Path, angle_path: Path, mask_path: Path) -> None:
    """把最终角度和掩膜数据嵌入角度可视化页面。"""
    if not viewer_path.is_file():
        return
    angle_text = angle_path.read_text(encoding="utf-8-sig")
    mask_text = mask_path.read_text(encoding="utf-8")
    html = viewer_path.read_text(encoding="utf-8")
    html = replace_script_data(html, "angle-data", angle_text)
    html = replace_script_data(html, "mask-data", mask_text)
    viewer_path.write_text(html, encoding="utf-8")


def mask_orientation(
    mask: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
] | None:
    """计算鱼体中心、头尾方向角、伸长率和头尾端点。

    先通过 PCA 得到鱼体主轴，再把掩膜投影到主轴上并按位置分箱统计
    鱼体宽度。鱼头一侧通常更宽，因此宽度均值较大的端点判为头端。
    指标角度为鱼体主轴与水平线的无符号夹角；头尾方向另用于箭头显示。

    Returns:
    - center_x, center_y: 掩膜质心。
    - horizontal_angle: 鱼体主轴与水平线的无符号夹角，范围 0–90°。
    - directed_angle: 从尾端指向头端的方向角，范围 0–360°。
    - elongation: 主次特征值之比的平方根，越大表示鱼体越细长。
    - tail_x, tail_y, head_x, head_y: 尾端和头端坐标。
    - head_width, tail_width: 两端附近宽度轮廓的均值。
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return None
    points = np.column_stack((xs, ys)).astype(np.float64)
    center = points.mean(axis=0)
    centered = points - center
    # PCA：最大特征值对应的特征向量就是鱼体像素分布的主方向。
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # eigh 返回升序特征值，反转后第一个特征向量即鱼体主轴。
    order = np.argsort(eigenvalues)[::-1]
    major_value = max(float(eigenvalues[order[0]]), 1e-9)
    minor_value = max(float(eigenvalues[order[1]]), 1e-9)
    major_vector = eigenvectors[:, order[0]]
    minor_vector = eigenvectors[:, order[1]]
    elongation = math.sqrt(major_value / minor_value)

    # 将掩膜点投影到主轴和副轴，用于定位两端并计算局部宽度轮廓。
    projected = centered @ major_vector
    perpendicular = centered @ minor_vector
    low = float(np.percentile(projected, 2.0))
    high = float(np.percentile(projected, 98.0))
    span = high - low
    if span < 2.0:
        return None

    normalized = (projected - low) / span
    profile_bins = 20
    bin_index = np.clip(
        (normalized * profile_bins).astype(int),
        0,
        profile_bins - 1,
    )
    widths = np.full(profile_bins, np.nan, dtype=np.float64)
    counts = np.zeros(profile_bins, dtype=np.int64)
    for index in range(profile_bins):
        selected = perpendicular[bin_index == index]
        counts[index] = selected.size
        if selected.size >= 3:
            low_width, high_width = np.percentile(selected, [10.0, 90.0])
            widths[index] = float(high_width - low_width)

    end_bins = max(2, int(round(profile_bins * 0.30)))
    left_widths = widths[:end_bins]
    right_widths = widths[-end_bins:]
    left_width = (
        float(np.nanmean(left_widths))
        if np.isfinite(left_widths).any()
        else 0.0
    )
    right_width = (
        float(np.nanmean(right_widths))
        if np.isfinite(right_widths).any()
        else 0.0
    )
    left_count = int(counts[:end_bins].sum())
    right_count = int(counts[-end_bins:].sum())
    head_is_right = (right_width, right_count) >= (left_width, left_count)

    low_point = center + major_vector * low
    high_point = center + major_vector * high
    if head_is_right:
        head_x, head_y = high_point
        tail_x, tail_y = low_point
        head_vector = major_vector
        head_width = right_width
        tail_width = left_width
    else:
        head_x, head_y = low_point
        tail_x, tail_y = high_point
        head_vector = -major_vector
        head_width = left_width
        tail_width = right_width

    # 图像坐标 y 轴向下：0° 向右、90° 向下、180° 向左、270° 向上。
    directed_angle = normalize_angle_deg(
        math.degrees(math.atan2(float(head_vector[1]), float(head_vector[0])))
    )
    half_turn_angle = directed_angle % 180.0
    horizontal_angle = min(half_turn_angle, 180.0 - half_turn_angle)
    return (
        float(center[0]),
        float(center[1]),
        horizontal_angle,
        directed_angle,
        elongation,
        float(tail_x),
        float(tail_y),
        float(head_x),
        float(head_y),
        float(head_width),
        float(tail_width),
    )


def calculate_fish_body_angle(data: list[dict]) -> dict[str, list[dict[str, object]]]:
    """从标准 SAM 记录计算每一帧的鱼体角度。

    输入记录必需包含 track_id、frame_id 和 segmentation；model_info、
    video_info、view_info、category_id、area、bbox 等元数据不会参与计算。
    segmentation 是 COCO 风格 RLE，size=[height, width]。

    每帧返回一条标准行为记录：
    - behavior_indicator 固定为 fish_body_angle。
    - behavior_value 为鱼体主轴与水平线的无符号夹角，范围 0–90°。
      图像坐标中 0° 向右、90° 向下、180° 向左、270° 向上。
    - frame_range 只包含当前帧。

    鱼头端通过主轴方向的宽度轮廓判断：将掩膜投影到主轴上分成 20 段，
    比较两端各 30% 范围的平均宽度，较宽的一端作为鱼头。

    掩膜缺失、为空、像素不足或解码失败时，behavior_value 返回 0。
    """
    if not isinstance(data, list) or not data:
        raise ValueError("data must be a non-empty standard record array")
    output: list[dict[str, object]] = []
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise TypeError(f"data[{index}] must be an object")
        if "track_id" not in record or "frame_id" not in record:
            raise ValueError(f"data[{index}] requires track_id and frame_id")
        frame_id = int(record["frame_id"])
        angle = 0.0
        segmentation = record.get("segmentation")
        if isinstance(segmentation, dict):
            try:
                orientation = mask_orientation(rle_to_mask(segmentation))
                if orientation is not None:
                    directed_angle = orientation[2]
                    angle = float(directed_angle)
            except (ValueError, KeyError, TypeError):
                # 单帧分割异常不阻断整批数据，按无效观测返回 0。
                angle = 0.0
        result = {
            "track_id": int(record["track_id"]),
            "frame_id": frame_id,
            "behavior_indicator": "fish_body_angle",
            "behavior_value": round(angle, 6),
            "unit": "degree",
            "frame_range": [frame_id],
        }
        output.append(result)
    return {"data": output}


def build_model(
    checkpoint: Path,
    bpe: Path,
    device: str,
    use_fp16: bool,
    resolution: int,
) -> tuple[torch.nn.Module, Sam3Processor]:
    """加载 SAM3 权重和处理器，并按设备配置 FP32/FP16 精度。"""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")
    if not bpe.is_file():
        raise FileNotFoundError(f"SAM3 BPE vocabulary not found: {bpe}")

    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        bpe_path=str(bpe),
        device="cpu",
        enable_inst_interactivity=False,
        load_from_HF=False,
    )
    model.eval()
    if device == "cuda":
        if use_fp16:
            model = model.half()
        model = model.cuda()
    else:
        model = model.float()

    processor = Sam3Processor(
        model,
        resolution=resolution,
        device=device,
        confidence_threshold=0.25,
    )
    if device == "cuda" and use_fp16:
        original_transform = processor.transform

        def half_transform(image: object) -> torch.Tensor:
            # 权重为 FP16 时，图像预处理结果也必须保持相同 dtype。
            return original_transform(image).half()

        processor.transform = half_transform
    return model, processor


def clear_prompt(state: dict) -> None:
    """清除当前图像的文本/几何提示和候选掩膜，避免跨鱼复用。"""
    for key in ("geometric_prompt", "boxes", "masks", "masks_logits", "scores"):
        state.pop(key, None)


def add_box_prompts(
    processor: Sam3Processor,
    boxes_xywh: list[list[float]],
    state: dict,
    text_prompt: str = "fish",
) -> dict:
    """向 SAM3 状态添加一个或多个框提示，并执行 grounding 前向计算。"""
    if "language_features" not in state["backbone_out"]:
        prompt = text_prompt.strip() or "visual"
        state["backbone_out"].update(
            processor.model.backbone.forward_text([prompt], device=processor.device)
        )
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()

    dtype = next(
        parameter.dtype
        for parameter in processor.model.parameters()
        if parameter.is_floating_point()
    )
    # SAM3 几何提示使用归一化 cx/cy/w/h，并在末尾增加提示维度。
    boxes = torch.tensor(
        boxes_xywh, device=processor.device, dtype=dtype
    ).view(-1, 1, 4)
    labels = torch.ones(
        (len(boxes_xywh), 1), device=processor.device, dtype=torch.bool
    )
    state["geometric_prompt"].append_boxes(boxes, labels)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=processor.device == "cuda" and dtype == torch.float16,
    ):
        return processor._forward_grounding(state)


def add_box_prompt(
    processor: Sam3Processor,
    box: list[float],
    state: dict,
    text_prompt: str = "fish",
) -> dict:
    """单框提示的便捷包装。"""
    return add_box_prompts(processor, [box], state, text_prompt)


def circular_mean_deg(values: np.ndarray) -> float:
    """计算 0–360° 方向角的圆周均值。"""
    radians = np.deg2rad(values)
    return normalize_angle_deg(
        np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()))
    )


def circular_median_deg(values: np.ndarray) -> float:
    """选择到所有方向角圆周距离之和最小的样本作为圆周中位数。"""
    differences = (
        (values[:, None] - values[None, :] + 180.0) % 360.0
    ) - 180.0
    return float(values[np.abs(differences).sum(axis=1).argmin()])


def circular_std_deg(values: np.ndarray) -> float:
    """按平均合向量长度计算方向角的圆周标准差。"""
    radians = np.deg2rad(values)
    resultant = math.hypot(
        float(np.sin(radians).mean()),
        float(np.cos(radians).mean()),
    )
    return float(
        np.rad2deg(math.sqrt(-2.0 * math.log(max(resultant, 1e-12))))
    )


def summarize_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    """按鱼汇总有效帧的方向角均值、中位数、标准差和极值。"""
    valid = dataframe[dataframe["valid"]].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "track_id",
                "valid_frames",
                "mean_angle_deg",
                "median_angle_deg",
                "std_angle_deg",
                "min_angle_deg",
                "max_angle_deg",
            ]
        )
    rows: list[dict[str, object]] = []
    for track_id, group in valid.groupby("track_id", sort=True):
        angles = group["angle_horizontal_deg"].to_numpy(dtype=float)
        rows.append(
            {
                "track_id": int(track_id),
                "valid_frames": int(len(angles)),
                "mean_angle_deg": float(np.mean(angles)),
                "median_angle_deg": float(np.median(angles)),
                "std_angle_deg": float(np.std(angles)),
                "min_angle_deg": float(np.min(angles)),
                "max_angle_deg": float(np.max(angles)),
            }
        )
    return pd.DataFrame(rows)


def process_video(args: argparse.Namespace) -> None:
    """运行 SAM3 分割与角度提取，并保存逐帧、汇总和掩膜数据。

    MOT 框只作为 SAM3 的空间提示；实际角度由清理后的 SAM 掩膜 PCA 主轴
    计算。--stride 控制参与推理的帧间隔，--prompt-mode 控制提示方式。
    """
    video_path = Path(args.video).resolve()
    mot_path = Path(args.mot).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    bpe = Path(args.bpe).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    head_mask_output = output_dir / "head_mask_shapes.json"
    head_masks_arg = getattr(args, "head_masks", None)
    if head_masks_arg:
        head_mask_path = Path(head_masks_arg).resolve()
        if not head_mask_path.is_file():
            raise FileNotFoundError(f"SAM3 head masks not found: {head_mask_path}")
    else:
        candidates = (head_mask_output, DEFAULT_HEAD_MASKS)
        head_mask_path = next(
            (
                path.resolve()
                for path in candidates
                if path.is_file()
            ),
            None,
        )
    head_payload: dict | None = None
    if head_mask_path is not None:
        head_payload = json.loads(head_mask_path.read_text(encoding="utf-8"))

    tracking = load_mot(mot_path)
    video_info = read_video_info(video_path)
    width = int(video_info["width"])
    height = int(video_info["height"])
    fps = float(video_info["fps"])
    if head_payload is not None:
        head_meta = head_payload.get("meta", {})
        head_width = int(head_meta.get("width", 0) or 0)
        head_height = int(head_meta.get("height", 0) or 0)
        if head_width and head_height and (
            head_width != width or head_height != height
        ):
            raise ValueError(
                "SAM3 head masks use a different frame size: "
                f"{head_width}x{head_height}, video is {width}x{height}"
            )
    # MOT 标注可能基于不同分辨率，需要把框同步缩放到视频像素坐标。
    scale_x = width / args.mot_width
    scale_y = height / args.mot_height
    tracking["x"] *= scale_x
    tracking["w"] *= scale_x
    tracking["y"] *= scale_y
    tracking["h"] *= scale_y
    device = resolve_device(args.device)
    if device == "cuda":
        # 开启 TF32 以提升矩阵运算吞吐，不影响角度定义。
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print(f"Video: {video_path}")
    print(f"MOT: {mot_path}")
    print(
        f"Frames: {video_info['frame_count']}  FPS: {fps:.3f}  "
        f"Size: {width}x{height}"
    )
    print(
        f"MOT source size: {args.mot_width}x{args.mot_height}  "
        f"Scale: {scale_x:.3f}x{scale_y:.3f}"
    )
    print(f"Tracks: {sorted(tracking['track_id'].unique().tolist())}")
    print(f"Device: {device}  FP16: {args.fp16 and device == 'cuda'}")
    if head_mask_path is None:
        print("SAM3 head masks: not found; using width-profile fallback")
    else:
        print(f"SAM3 head masks: {head_mask_path}")

    model, processor = build_model(
        checkpoint=checkpoint,
        bpe=bpe,
        device=device,
        use_fp16=args.fp16,
        resolution=args.resolution,
    )
    processor.set_confidence_threshold(args.confidence)
    # processor 已持有模型引用，删除局部变量可减少显存重复占用。
    del model

    by_frame = {
        int(frame_number): group.sort_values("track_id")
        for frame_number, group in tracking.groupby("frame", sort=False)
    }
    first_frame = int(tracking["frame"].min())
    last_frame = int(tracking["frame"].max())
    max_frames = args.max_frames if args.max_frames is not None else last_frame

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    rows: list[dict[str, object]] = []
    mask_samples: list[dict[str, object]] = []
    started = time.perf_counter()
    processed_frames = 0
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index < first_frame or frame_index > max_frames:
                continue
            if (frame_index - first_frame) % max(1, args.stride) != 0:
                # 只推理抽样帧；输出中的时间仍对应原始视频帧号。
                continue

            tracks = by_frame.get(frame_index)
            if tracks is None or tracks.empty:
                continue

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with autocast_context(device, args.amp):
                state = processor.set_image(image)
            if args.prompt_mode == "text":
                with autocast_context(device, args.amp):
                    state = processor.set_text_prompt(args.text_prompt, state)

            track_items: list[tuple[int, tuple[float, float, float, float]]] = []
            for track in tracks.itertuples(index=False):
                prompt_box = clip_box(track, width, height)
                if prompt_box is None:
                    continue
                track_items.append((int(track.track_id), prompt_box))

            if args.prompt_mode == "multi" and track_items:
                clear_prompt(state)
                # 同帧所有框一次性送入，减少重复的图像编码和模型调用。
                with autocast_context(device, args.amp):
                    state = add_box_prompts(
                        processor,
                        [
                            normalized_box(prompt_box, width, height)
                            for _, prompt_box in track_items
                        ],
                        state,
                        args.text_prompt,
                    )

            for track_id, prompt_box in track_items:
                if args.prompt_mode == "box":
                    # 单框模式下逐鱼重建提示并重新执行 grounding。
                    clear_prompt(state)
                    with autocast_context(device, args.amp):
                        state = add_box_prompt(
                            processor,
                            normalized_box(prompt_box, width, height),
                            state,
                            args.text_prompt,
                        )
                mask, score, overlap = select_best_mask(
                    state, prompt_box, args.minimum_mask_area
                )
                base_row: dict[str, object] = {
                    "frame": frame_index,
                    "time_s": (frame_index - 1) / fps,
                    "track_id": track_id,
                    "score": score,
                    "box_iou": overlap,
                    "valid": False,
                }
                if mask is None:
                    # 保留一条 valid=False 记录，方便下游统计失败帧。
                    rows.append(base_row)
                    continue
                orientation = mask_orientation(mask)
                if orientation is None:
                    rows.append(base_row)
                    continue
                (
                    center_x,
                    center_y,
                    horizontal_angle,
                    directed_angle,
                    elongation,
                    tail_x,
                    tail_y,
                    head_x,
                    head_y,
                    head_width,
                    tail_width,
                ) = orientation
                base_row.update(
                    {
                        "valid": True,
                        "mask_area": int(mask.sum()),
                        "center_x": center_x,
                        "center_y": center_y,
                        "angle_horizontal_deg": horizontal_angle,
                        "orientation_deg": directed_angle,
                        "directed_angle_deg": directed_angle,
                        "elongation": elongation,
                        "head_x": head_x,
                        "head_y": head_y,
                        "tail_x": tail_x,
                        "tail_y": tail_y,
                        "head_width": head_width,
                        "tail_width": tail_width,
                    }
                )
                rows.append(base_row)
                polygons = mask_polygons(mask)
                if polygons:
                    mask_samples.append(
                        {
                            "frame": frame_index,
                            "track_id": track_id,
                            "box": [float(value) for value in prompt_box],
                            "score": score,
                            "box_iou": overlap,
                            "angle_horizontal_deg": horizontal_angle,
                            "orientation_deg": directed_angle,
                            "directed_angle_deg": directed_angle,
                            "elongation": elongation,
                            "head_x": head_x,
                            "head_y": head_y,
                            "tail_x": tail_x,
                            "tail_y": tail_y,
                            "head_width": head_width,
                            "tail_width": tail_width,
                            "polygons": polygons,
                        }
                    )
            processed_frames += 1
            if processed_frames % 20 == 0:
                elapsed = time.perf_counter() - started
                rate = processed_frames / max(elapsed, 1e-6)
                remaining_samples = max(
                    0, (max_frames - frame_index + max(1, args.stride) - 1) // max(1, args.stride)
                )
                eta_minutes = remaining_samples / max(rate, 1e-6) / 60.0
                print(
                    f"frame {frame_index}/{max_frames}  "
                    f"{rate:.2f} frames/s  ETA {eta_minutes:.1f} min",
                    flush=True,
                )
            if device == "cuda" and processed_frames % 50 == 0:
                torch.cuda.empty_cache()
    finally:
        capture.release()

    row_by_key = {
        (int(row["frame"]), int(row["track_id"])): row
        for row in rows
        if bool(row.get("valid", False))
    }
    synchronized_keys = (
        "center_x",
        "center_y",
        "angle_horizontal_deg",
        "orientation_deg",
        "directed_angle_deg",
        "elongation",
        "head_x",
        "head_y",
        "tail_x",
        "tail_y",
        "head_width",
        "tail_width",
    )
    for sample in mask_samples:
        row = row_by_key.get((int(sample["frame"]), int(sample["track_id"])))
        if row is None:
            continue
        for key in synchronized_keys:
            sample[key] = row[key]
    head_stats = apply_head_direction(
        rows,
        mask_samples,
        head_payload,
        width,
        height,
    )
    if head_payload is not None:
        head_mask_output.write_text(
            json.dumps(
                head_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

    dataframe = pd.DataFrame(rows)
    csv_path = output_dir / "frame_angles.csv"
    dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = summarize_results(dataframe)
    summary_path = output_dir / "track_angle_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    mask_path = output_dir / "mask_shapes.json"
    # 掩膜只保存简化轮廓，避免把完整逐帧二值掩膜写入 JSON。
    mask_path.write_text(
        json.dumps(
            {
                "meta": {
                    "video": str(video_path),
                    "fps": fps,
                    "frame_stride": args.stride,
                    "width": width,
                    "height": height,
                    "head_mask_source": (
                        str(head_mask_path)
                        if head_mask_path is not None
                        else None
                    ),
                    "head_matched": head_stats["matched"],
                    "head_fallback": head_stats["fallback"],
                },
                "samples": mask_samples,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    elapsed_seconds = time.perf_counter() - started
    result = {
        "meta": {
            "video": str(video_path),
            "mot": str(mot_path),
            "checkpoint": str(checkpoint),
            "bpe": str(bpe),
            "device": device,
            "fp16": bool(args.fp16 and device == "cuda"),
            "amp": bool(args.amp and device == "cuda"),
            "width": width,
            "height": height,
            "sam3_resolution": args.resolution,
            "fps": fps,
            "mot_source_width": args.mot_width,
            "mot_source_height": args.mot_height,
            "processed_frames": processed_frames,
            "frame_stride": args.stride,
            "prompt_mode": args.prompt_mode,
            "text_prompt": args.text_prompt,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "angle_definition": (
                "Unsigned angle between the fish-body principal axis and the "
                "horizontal image direction, in degrees from 0 to 90. The "
                "head endpoint is retained separately for direction arrows."
            ),
            "angle_method": (
                "pca_sam3_head_with_width_profile_fallback"
                if head_payload is not None
                else "pca_width_profile"
            ),
            "head_detection": (
                "SAM3 fish-head mask with width-profile fallback"
                if head_payload is not None
                else "width profile"
            ),
            "head_mask_source": (
                str(head_mask_path) if head_mask_path is not None else None
            ),
            "sam3_head_matched_observations": head_stats["matched"],
            "sam3_head_fallback_observations": head_stats["fallback"],
            "sam3_head_valid_observations": head_stats["valid_observations"],
            "sam3_head_direction_changed": head_stats["changed"],
        },
        "tracks": summary.to_dict(orient="records"),
    }
    json_path = output_dir / "angle_summary.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    viewer_path = Path(getattr(args, "viewer", str(DEFAULT_VIEWER))).resolve()
    update_viewer_data(viewer_path, csv_path, mask_path)

    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {mask_path}")
    print(f"Saved: {json_path}")
    print(
        "SAM3 head matched "
        f"{head_stats['matched']}/{head_stats['valid_observations']} "
        f"observations; direction changed in {head_stats['changed']}"
    )


def build_parser() -> argparse.ArgumentParser:
    """构建鱼体角度离线分析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Segment fish bodies with SAM3 and measure their angle."
    )
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--mot", default=str(DEFAULT_MOT))
    parser.add_argument("--mot-width", type=int, default=1280)
    parser.add_argument("--mot-height", type=int, default=720)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--bpe", default=str(DEFAULT_BPE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--head-masks",
        default=None,
        help=(
            "optional SAM3 fish-head mask_shapes.json; when omitted, "
            "output/head_mask_shapes.json and DEFAULT_HEAD_MASKS are tried"
        ),
    )
    parser.add_argument("--viewer", default=str(DEFAULT_VIEWER))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="process every Nth frame; default 1 processes all frames",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--minimum-mask-area", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--text-prompt", default="fish")
    parser.add_argument(
        "--prompt-mode", choices=("multi", "box", "text"), default="box"
    )
    parser.set_defaults(fp16=False, amp=False)
    return parser


def main() -> None:
    """校验命令行参数后启动 SAM3 视频分析。"""
    # Python 直接调用示例（不经过命令行）：
    # from angle_analysis import calculate_fish_body_angle
    # result = calculate_fish_body_angle(data)
    args = build_parser().parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1")
    if args.mot_width < 1 or args.mot_height < 1:
        raise ValueError("--mot-width and --mot-height must be at least 1")
    process_video(args)


if __name__ == "__main__":
    main()
