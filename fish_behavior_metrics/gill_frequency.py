"""鳃动频率计算模块。

指标定义：在每条鱼的分割掩膜内计算逐帧灰度均值，再在固定时间窗内检测
周期性峰值，将峰数换算为每分钟鳃动次数（times/min）。

当前阶段 segmentation 暂由人工标注鳃动区域，后续可直接替换为检测算法输出
的同格式掩膜，指标接口无需改变。

主要数据流：
segmentation 掩膜 -> 逐帧掩膜灰度均值 -> 窗口覆盖率检查 ->
0.60-1.50 Hz 带通滤波 -> 平滑/归一化 -> 峰值检测 -> 频率输出。

公开入口 calculate_gill_frequency 只接收标准行为记录和视频路径，不依赖
命令行或可视化页面。多帧指标使用固定时间窗；窗口不足或覆盖率过低时返回
0，并保留实际累计的 frame_range。
"""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, savgol_filter, sosfiltfilt


ROOT = Path(__file__).resolve().parent
RESULTS_DIRNAME = "results"
GILL_RESULTS_DIRNAME = "gill"
GILL_DATA_FILENAME = "gill_data.json"
GILL_ANALYSIS_DIRNAME = "analysis"
_LOCAL_VIDEO = ROOT / "datasets" / "video.mp4"
_LOCAL_ZIP = ROOT / "datasets" / "annotations_mot.zip"
_LOCAL_ANNOTATIONS = ROOT / "datasets" / "annotations_mot"
DEFAULT_VIDEO = _LOCAL_VIDEO
DEFAULT_ZIP = _LOCAL_ZIP if _LOCAL_ZIP.exists() else _LOCAL_ANNOTATIONS
# 鳃动信号的有效频带和峰值筛选阈值；离线分析 API 共用这组默认参数。
BAND_LOW_HZ = 0.60
BAND_HIGH_HZ = 1.50
PEAK_PROMINENCE = 0.05
PEAK_DISTANCE_S = 0.25


def load_mot(zip_path: Path) -> pd.DataFrame:
    """从 MOT 目录或 zip 包读取 gt.txt，并返回标准列名的跟踪表。"""
    if zip_path.is_dir():
        frame = pd.read_csv(
            zip_path / "gt" / "gt.txt",
            header=None,
            names=["frame", "fish_id", "x", "y", "w", "h", "mark", "class", "conf"],
        )
    else:
        with zipfile.ZipFile(zip_path) as archive:
            frame = pd.read_csv(
                io.BytesIO(archive.read("gt/gt.txt")),
                header=None,
                names=["frame", "fish_id", "x", "y", "w", "h", "mark", "class", "conf"],
            )
    for column in ("frame", "fish_id", "mark", "class"):
        frame[column] = frame[column].astype(int)
    return frame


def interpolate(values: np.ndarray) -> np.ndarray:
    """用相邻有效值线性插值补齐 NaN/Inf；有效点少于两个时返回全零。"""
    y = np.asarray(values, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return np.zeros_like(y)
    index = np.arange(len(y))
    return np.interp(index, index[finite], y[finite])


def bandpass(values: np.ndarray, fps: float) -> np.ndarray:
    """先补齐缺失值，再对信号做零相位四阶巴特沃斯带通滤波。"""
    y = interpolate(values)
    sos = butter(
        4,
        [BAND_LOW_HZ, BAND_HIGH_HZ],
        btype="bandpass",
        fs=fps,
        output="sos",
    )
    return sosfiltfilt(sos, y)


def robust_z(values: np.ndarray) -> np.ndarray:
    """使用中位数和 IQR 做稳健标准化，降低异常振幅对归一化的影响。"""
    y = np.asarray(values, dtype=float)
    median = float(np.median(y))
    q25, q75 = np.percentile(y, [25, 75])
    scale = float(q75 - q25) or float(np.std(y)) or 1.0
    return (y - median) / scale


def peak_candidates(values: np.ndarray, fps: float, raw: bool) -> list[dict[str, float | int]]:
    """检测峰值候选，并返回峰值帧索引和显著度。

    raw=True 时直接在插值后的原始曲线上寻找峰；raw=False 时先做带通滤波。
    两种模式都先稳健归一化，再用 Savitzky-Golay 平滑以抑制单帧噪声。
    """
    base = interpolate(values) if raw else bandpass(values, fps)
    z = robust_z(base)
    smooth = savgol_filter(z, 11, 3)
    peaks, properties = find_peaks(
        smooth,
        distance=max(1, int(round(0.10 * fps))),
        prominence=0.001,
    )
    result: list[dict[str, float | int]] = []
    for peak, prominence in zip(peaks, properties["prominences"]):
        index = int(peak)
        if raw:
            # 平滑会轻微移动峰位，因此在原始曲线附近重新定位局部最大值。
            lo = max(0, index - 6)
            hi = min(len(z), index + 7)
            index = lo + int(np.argmax(z[lo:hi]))
        result.append(
            {
                "index": index,
                "prominence": round(float(prominence), 6),
            }
        )
    return result


def selected_peaks(candidates: list[dict[str, float | int]], fps: float) -> list[dict[str, float | int]]:
    """按显著度和最小时间间隔筛选峰值，最终按帧顺序返回。"""
    minimum = max(1, int(round(PEAK_DISTANCE_S * fps)))
    eligible = sorted(
        [item for item in candidates if item["prominence"] >= PEAK_PROMINENCE],
        key=lambda item: item["prominence"],
        reverse=True,
    )
    chosen: list[dict[str, float | int]] = []
    for item in eligible:
        # 从强到弱贪心保留峰值，避免同一鳃动周期被重复计数。
        if all(abs(int(other["index"]) - int(item["index"])) >= minimum for other in chosen):
            chosen.append(item)
    return sorted(chosen, key=lambda item: int(item["index"]))


def nullable(values: np.ndarray, digits: int = 3) -> list[float | None]:
    """把 NumPy 曲线转换为可直接写入 JSON 的列表，非有限值转为 None。"""
    return [
        round(float(value), digits) if np.isfinite(value) else None
        for value in values
    ]


def extract_metrics(video_path: Path, tracking: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """按 MOT 跟踪框提取逐帧灰度均值和方差，供离线可视化分析使用。

    跟踪框会先裁剪到视频边界；只有置信度达标且框几乎完整位于画面内时，
    该帧才标记为有效，避免越界目标污染曲线。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    by_frame = {
        int(frame): group.sort_values("fish_id")
        for frame, group in tracking.groupby("frame", sort=False)
    }
    rows: list[dict[str, float | int | bool]] = []
    for frame_index in range(min(frame_count, int(tracking.frame.max()))):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Video ended at frame {frame_index + 1}")
        frame_no = frame_index + 1
        tracks = by_frame.get(frame_no)
        if tracks is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for row in tracks.itertuples(index=False):
            # MOT 的 x/y/w/h 是浮点框，这里转换成左闭右开的像素边界。
            x1 = max(0, min(width - 1, int(math.floor(row.x))))
            y1 = max(0, min(height - 1, int(math.floor(row.y))))
            x2 = max(x1 + 1, min(width, int(math.ceil(row.x + row.w))))
            y2 = max(y1 + 1, min(height, int(math.ceil(row.y + row.h))))
            roi = gray[y1:y2, x1:x2].astype(np.float64)
            # 裁剪后面积与原框面积的比值用于识别严重出界的跟踪框。
            visible = roi.size / max(1.0, float(row.w) * float(row.h))
            valid = bool(float(row.conf) >= 0.5 and visible >= 0.98 and roi.size)
            rows.append(
                {
                    "frame": frame_no,
                    "time_s": frame_index / fps,
                    "fish_id": int(row.fish_id),
                    "valid_tracking": valid,
                    "confidence": float(row.conf),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "gray_mean": float(roi.mean()) if valid else np.nan,
                    "gray_variance": float(roi.var()) if valid else np.nan,
                }
            )
    cap.release()
    metrics = pd.DataFrame(rows).sort_values(["fish_id", "frame"]).reset_index(drop=True)
    return metrics, {
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "duration": frame_count / fps,
    }


def build_visualization_data(metrics: pd.DataFrame, video_info: dict[str, float | int]) -> dict[str, object]:
    """把逐帧灰度统计整理成可视化页面使用的鱼级曲线和峰值数据。"""
    fps = float(video_info["fps"])
    fish_records: list[dict[str, object]] = []
    for fish_id, group in metrics.groupby("fish_id", sort=True):
        group = group.sort_values("frame").reset_index(drop=True)
        first_frame = int(group.frame.min())
        last_frame = int(group.frame.max())
        mean = np.full(last_frame, np.nan)
        variance = np.full(last_frame, np.nan)
        valid = np.zeros(last_frame, dtype=bool)
        boxes: list[list[float] | None] = [None] * last_frame
        for row in group.itertuples(index=False):
            index = int(row.frame) - 1
            mean[index] = float(row.gray_mean)
            variance[index] = float(row.gray_variance)
            valid[index] = bool(row.valid_tracking)
            boxes[index] = [int(row.x1), int(row.y1), int(row.x2), int(row.y2), bool(row.valid_tracking)]

        # 同时保留原始曲线、带通曲线和平滑曲线，便于页面排查峰值来源。
        filtered_mean = robust_z(bandpass(mean, fps))
        filtered_variance = robust_z(bandpass(variance, fps))
        smoothed_mean = savgol_filter(filtered_mean, 11, 3)
        smoothed_variance = savgol_filter(filtered_variance, 11, 3)
        mean_candidates = peak_candidates(mean, fps, raw=False)
        variance_candidates = peak_candidates(variance, fps, raw=False)
        mean_raw = peak_candidates(mean, fps, raw=True)
        variance_raw = peak_candidates(variance, fps, raw=True)
        mean_peaks = selected_peaks(mean_candidates, fps)
        variance_peaks = selected_peaks(variance_candidates, fps)
        mean_valid = mean[np.isfinite(mean)]
        variance_valid = variance[np.isfinite(variance)]
        fish_records.append(
            {
                "id": int(fish_id),
                "sourceTrackId": int(fish_id),
                "firstFrame": first_frame,
                "lastFrame": last_frame,
                "validFrames": int(valid.sum()),
                "baselineMean": float(np.median(mean_valid)),
                "baselineVariance": float(np.median(variance_valid)),
                "minMean": float(np.min(mean_valid)),
                "maxMean": float(np.max(mean_valid)),
                "minVariance": float(np.min(variance_valid)),
                "maxVariance": float(np.max(variance_valid)),
                "mean": nullable(mean),
                "variance": nullable(variance),
                "filteredMean": nullable(filtered_mean),
                "filteredVariance": nullable(filtered_variance),
                "smoothedMean": nullable(smoothed_mean),
                "smoothedVariance": nullable(smoothed_variance),
                "meanPeakCandidates": mean_candidates,
                "variancePeakCandidates": variance_candidates,
                "meanRawPeakCandidates": mean_raw,
                "varianceRawPeakCandidates": variance_raw,
                "meanPeakIndices": [item["index"] for item in mean_peaks],
                "variancePeakIndices": [item["index"] for item in variance_peaks],
                "meanPeakCount": len(mean_peaks),
                "variancePeakCount": len(variance_peaks),
                "boxes": boxes,
                "valid": valid.tolist(),
            }
        )
    return {
        "meta": {
            **video_info,
            "frameCount": int(video_info["frame_count"]),
            "fps": fps,
            "bandLowHz": BAND_LOW_HZ,
            "bandHighHz": BAND_HIGH_HZ,
            "regionType": "box",
            "source": str(DEFAULT_VIDEO) + " + " + str(DEFAULT_ZIP),
        },
        "fish": fish_records,
    }


def export_peak_tables(data: dict[str, object], output_dir: Path) -> pd.DataFrame:
    """导出每条鱼、每条曲线的峰值数量和峰值发生位置。"""
    fps = float(data["meta"]["fps"])
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for fish in data["fish"]:
        duration = (int(fish["lastFrame"]) - int(fish["firstFrame"]) + 1) / fps
        curves = (
            ("raw_mean", "均值", "meanRawPeakCandidates"),
            ("raw_variance", "方差", "varianceRawPeakCandidates"),
            ("filtered_mean", "带通均值", "meanPeakCandidates"),
        )
        for curve, label, key in curves:
            peaks = selected_peaks(fish[key], fps)
            # 汇总表按“鱼 + 曲线”聚合，明细表保留每一个峰的时间与显著度。
            summary_rows.append(
                {
                    "fish_id": int(fish["id"]),
                    "curve": curve,
                    "metric_label": label,
                    "duration_s": duration,
                    "peak_count": len(peaks),
                    "peak_rate_per_min": len(peaks) * 60.0 / duration,
                    "peak_prominence_min": PEAK_PROMINENCE,
                    "peak_distance_min_s": PEAK_DISTANCE_S,
                }
            )
            for number, peak in enumerate(peaks, start=1):
                index = int(peak["index"])
                detail_rows.append(
                    {
                        "fish_id": int(fish["id"]),
                        "curve": curve,
                        "metric_label": label,
                        "peak_number": number,
                        "frame": index + 1,
                        "time_s": index / fps,
                        "prominence": float(peak["prominence"]),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)
    summary.to_csv(
        output_dir / "peak_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    detail.to_csv(
        output_dir / "peak_details.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def decode_compressed_rle(text: str) -> list[int]:
    """解码 COCO 压缩 RLE 字符串，返回各游程的累计边界。"""
    counts: list[int] = []
    position = 0
    while position < len(text):
        value = 0
        shift = 0
        more = 1
        while more:
            # 每个字符低 5 位承载数据，bit 5 表示是否继续读取。
            character = ord(text[position]) - 48
            value |= (character & 0x1F) << (5 * shift)
            more = character & 0x20
            position += 1
            shift += 1
            if not more and (character & 0x10):
                # bit 4 为符号位，补 1 以还原负数。
                value |= -1 << (5 * shift)
        if counts:
            value += counts[-1]
        counts.append(value)
    return counts


def cumulative_counts(segmentation: dict) -> tuple[int, int, np.ndarray]:
    """把压缩字符串或游程数组统一转换为累计边界，并校验覆盖整幅图像。"""
    height, width = [int(value) for value in segmentation["size"]]
    counts = segmentation["counts"]
    total = height * width
    if isinstance(counts, str):
        cumulative = np.asarray(decode_compressed_rle(counts), dtype=np.int64)
    else:
        raw = np.asarray(counts, dtype=np.int64)
        # 兼容上游提供累计边界或每段长度两种非压缩 RLE 形式。
        is_cumulative = (
            len(raw) > 0
            and np.all(raw >= 0)
            and np.all(np.diff(raw) >= 0)
            and int(raw[-1]) == total
        )
        cumulative = raw if is_cumulative else np.cumsum(raw)
    if len(cumulative) == 0 or int(cumulative[-1]) != total:
        raise ValueError("Invalid RLE segmentation: counts do not cover the image")
    return height, width, cumulative


def rle_to_mask(segmentation: dict) -> np.ndarray:
    """将标准 RLE 解码为 HxW 的二值掩膜。"""
    height, width, cumulative = cumulative_counts(segmentation)
    flat = np.zeros(height * width, dtype=np.uint8)
    start = 0
    for run_index, end in enumerate(cumulative):
        end = int(end)
        # RLE 从背景段开始，因此奇数编号的游程是鱼体前景。
        if run_index % 2 == 1 and end > start:
            flat[start:end] = 1
        start = end
    return flat.reshape(height, width)


def segmentation_gray_stats(gray: np.ndarray, segmentation: dict) -> tuple[float, int]:
    """计算掩膜内灰度均值与前景像素数，不把掩膜外像素计入统计。"""
    height, width, cumulative = cumulative_counts(segmentation)
    if gray.shape[0] != height or gray.shape[1] != width:
        # 掩膜与视频分辨率不同时，用最近邻缩放保持二值边界。
        mask = cv2.resize(
            rle_to_mask(segmentation),
            (gray.shape[1], gray.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        count = int(mask.sum())
        if count <= 0:
            raise ValueError("Segmentation mask contains no foreground pixels")
        return float(gray[mask.astype(bool)].mean()), count

    total = 0.0
    count = 0
    start = 0
    for run_index, end in enumerate(cumulative):
        end = int(end)
        if end < start:
            raise ValueError("Invalid RLE segmentation: negative run length")
        if run_index % 2 == 1 and end > start:
            # 直接按游程读取像素，避免为每一帧创建完整布尔掩膜。
            first_row, first_column = divmod(start, width)
            last_row, last_column = divmod(end - 1, width)
            if first_row == last_row:
                values = gray[first_row, first_column : last_column + 1]
                total += float(values.sum())
                count += int(values.size)
            else:
                first = gray[first_row, first_column:]
                last = gray[last_row, : last_column + 1]
                total += float(first.sum()) + float(last.sum())
                count += int(first.size + last.size)
                if last_row > first_row + 1:
                    middle = gray[first_row + 1 : last_row, :]
                    total += float(middle.sum())
                    count += int(middle.size)
        start = end
    if count <= 0:
        raise ValueError("Segmentation mask contains no foreground pixels")
    return total / count, count


def _validate_standard_data(data: list[dict]) -> list[dict]:
    """校验标准输入的基本结构，并返回原列表以保持调用链简洁。"""
    if not isinstance(data, list) or not data:
        raise ValueError("data must be a non-empty list of standard records")
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise TypeError(f"data[{index}] must be an object")
        for field in ("track_id", "frame_id", "segmentation"):
            if field not in record:
                raise ValueError(f"data[{index}] is missing required field: {field}")
        segmentation = record["segmentation"]
        if not isinstance(segmentation, dict):
            raise TypeError(f"data[{index}].segmentation must be an object")
        if "size" not in segmentation or "counts" not in segmentation:
            raise ValueError(
                f"data[{index}].segmentation requires size and counts"
            )
    return data


def extract_segmentation_series(
    data: list[dict],
    video_path: str | Path,
) -> tuple[dict[int, dict[int, float]], float]:
    """从 segmentation 掩膜提取每条鱼、每一帧的掩膜灰度均值。

    必需字段为 track_id、frame_id 和 segmentation。frame_id 从 0 开始时按
    零基解释，否则按一基解释。segmentation 当前暂由人工标注，后续可由检测
    算法生成；掩膜分辨率与视频不同时会先以最近邻方式缩放。

    Returns:
    - series: {track_id: {frame_id: gray_mean}}。
    - fps: 视频帧率。
    """
    records = _validate_standard_data(data)
    frame_ids = [int(record["frame_id"]) for record in records]
    # 混合数据中只要有 0 帧，就认为整批数据采用零基编号。
    zero_based = min(frame_ids) == 0
    by_video_index: dict[int, list[dict]] = {}
    for record in records:
        frame_id = int(record["frame_id"])
        video_index = frame_id if zero_based else frame_id - 1
        by_video_index.setdefault(video_index, []).append(record)

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video FPS: {video_path}")
    maximum_index = max(by_video_index)
    series: dict[int, dict[int, float]] = {}
    try:
        for video_index in range(maximum_index + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    "Video ended before the requested frame: "
                    f"{video_index + (1 if zero_based else 0)}"
                )
            frame_records = by_video_index.get(video_index)
            if not frame_records:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for record in frame_records:
                track_id = int(record["track_id"])
                frame_id = int(record["frame_id"])
                try:
                    # 灰度统计只读取 segmentation 前景；当前掩膜暂由人工标注。
                    mean_value, _ = segmentation_gray_stats(
                        gray, record["segmentation"]
                    )
                except (ValueError, KeyError, TypeError):
                    # 单帧掩膜损坏时跳过该观测，不中断其他鱼的计算。
                    continue
                series.setdefault(track_id, {})[frame_id] = mean_value
    finally:
        cap.release()
    return series, fps


def _frequency_from_window(
    values_by_frame: dict[int, float],
    start_frame: int,
    end_frame: int,
    fps: float,
    min_distance_seconds: float,
    prominence: float,
    minimum_coverage: float,
) -> tuple[float, int, list[int]]:
    """在指定帧窗口内计算鳃动频率，并返回覆盖率与实际使用的帧号。"""
    # 先检查窗口覆盖率和最小样本数，避免稀疏数据产生伪峰值。
    window_frames = end_frame - start_frame + 1
    frame_ids = sorted(
        frame_id
        for frame_id in values_by_frame
        if start_frame <= frame_id <= end_frame
    )
    coverage = len(frame_ids) / max(1, window_frames)
    if coverage < minimum_coverage or len(frame_ids) < 3:
        return 0.0, int(round(coverage * 100)), frame_ids

    array = np.full(window_frames, np.nan, dtype=float)
    for frame_id in frame_ids:
        array[frame_id - start_frame] = values_by_frame[frame_id]
    finite = np.isfinite(array)
    if finite.sum() < 3:
        return 0.0, int(round(coverage * 100)), frame_ids
    indexes = np.arange(window_frames)
    # 缺失帧用线性插值补齐，只在已有帧覆盖率达标后执行。
    filled = np.interp(indexes, indexes[finite], array[finite])
    # 鳃动有效频段固定在 0.60–1.50 Hz，滤除慢漂移和高频噪声。
    sos = butter(
        4,
        [BAND_LOW_HZ, BAND_HIGH_HZ],
        btype="bandpass",
        fs=fps,
        output="sos",
    )
    try:
        filtered = sosfiltfilt(sos, filled)
    except ValueError:
        # 窗口过短时滤波器不满足稳定条件，按“结果不可用”处理。
        return 0.0, int(round(coverage * 100)), frame_ids
    # Savitzky-Golay 窗口必须是奇数，且不能超过当前分析窗口长度。
    smooth_window = min(
        11,
        window_frames if window_frames % 2 == 1 else window_frames - 1,
    )
    smoothed = (
        savgol_filter(filtered, smooth_window, 3)
        if smooth_window >= 5
        else filtered
    )
    q25, q75 = np.percentile(smoothed, [25, 75])
    scale = float(q75 - q25) or float(np.std(smoothed)) or 1.0
    # 使用中位数和 IQR 做稳健归一化，降低个别帧振幅变化的影响。
    normalized = (smoothed - float(np.median(smoothed))) / scale
    peaks, _ = find_peaks(
        normalized,
        distance=max(1, int(round(min_distance_seconds * fps))),
        prominence=prominence,
    )
    # 峰数换算为每分钟次数。
    frequency = len(peaks) * 60.0 / (window_frames / fps)
    return frequency, int(round(coverage * 100)), frame_ids


def calculate_gill_frequency(
    data: list[dict],
    video_path: str | Path,
    window_seconds: float = 10.0,
    min_distance_seconds: float = 0.25,
    prominence: float = 0.05,
    minimum_coverage: float = 0.80,
) -> dict[str, list[dict[str, object]]]:
    """从鳃动区域掩膜计算每条鱼最新的鳃动频率。

    segmentation 当前暂由人工标注，后续可替换为检测算法输出；本函数只依赖
    RLE 掩膜，不使用 bbox 推断鳃动区域。

    Args:
    - data: 标准记录数组。
      - track_id: 鱼的稳定轨迹编号，整数。
      - frame_id: 原始视频帧编号，支持 0 基或 1 基编号。
      - segmentation: 鳃动区域 RLE 掩膜。
      - size: [height, width]，掩膜高和宽。
      - counts: 压缩 RLE 字符串或游程数组。
      - model_info、video_info、view_info、category_id、area、bbox: 可选，
        当前计算不使用。
    - video_path: 输入视频路径，用于读取掩膜内灰度。
    - window_seconds: 最新分析窗口长度，默认 10 秒。
    - min_distance_seconds: 相邻峰最小间隔，默认 0.25 秒。
    - prominence: 最小峰值显著度，默认 0.05。
    - minimum_coverage: 窗口最低有效帧覆盖率，默认 0.80。

    Returns:
    - {"data": [...]}，每条鱼一条最新结果。
    - track_id: 鱼的轨迹编号。
    - frame_id: 最新原始帧号。
    - behavior_indicator: 固定为 "gill_frequency"。
    - behavior_value: 鳃动频率，单位 times/min；窗口不足或不可用时为 0。
    - unit: 固定为 "times/min"。
    - frame_range: 实际参与计算的原始帧号。
    """
    series, fps = extract_segmentation_series(data, video_path)
    # 至少 15 帧可避免低帧率视频上窗口过短导致滤波器不稳定。
    window_frames = max(15, int(round(window_seconds * fps)))
    output_rows: list[dict[str, object]] = []
    for track_id, values_by_frame in sorted(series.items()):
        frame_ids = sorted(values_by_frame)
        first_frame = frame_ids[0]
        last_frame = frame_ids[-1]
        if last_frame - first_frame + 1 < window_frames:
            # 多帧窗口尚未累计完成：按约定返回 0。
            used_frames = frame_ids
            value = 0.0
            coverage = len(used_frames) / max(1, last_frame - first_frame + 1)
        else:
            # 使用最新一段完整窗口计算频率。
            start_frame = last_frame - window_frames + 1
            value, _, used_frames = _frequency_from_window(
                values_by_frame,
                start_frame,
                last_frame,
                fps,
                min_distance_seconds,
                prominence,
                minimum_coverage,
            )
            coverage = len(used_frames) / window_frames
        output_rows.append(
            {
                # 该时间序列对应的鱼/鳃动区域轨迹编号。
                "track_id": int(track_id),
                # 结果对应的最新原始视频帧号。
                "frame_id": int(last_frame),
                # 统一行为输出中的指标名称。
                "behavior_indicator": "gill_frequency",
                # 每分钟鳃动次数；窗口不足或数据不可用时为 0.0。
                "behavior_value": round(float(value), 3),
                # 鳃动频率单位：次/分钟。
                "unit": "times/min",
                # 实际参与计算的原始帧号，而不是连续数组下标。
                "frame_range": [int(frame_id) for frame_id in used_frames],
            }
        )
    return {"data": output_rows}


def build_sliding_frequency(
    payload: dict,
    video_path: Path,
    window_seconds: float = 10.0,
    min_distance_seconds: float = 0.25,
    prominence: float = 0.05,
    minimum_coverage: float = 0.80,
) -> dict:
    """兼容旧版 {"data": [...]} 输入，参数含义同 calculate_gill_frequency。"""
    records = payload.get("data")
    if not isinstance(records, list):
        raise ValueError("Input JSON must contain a data array")
    return calculate_gill_frequency(
        records,
        video_path,
        window_seconds=window_seconds,
        min_distance_seconds=min_distance_seconds,
        prominence=prominence,
        minimum_coverage=minimum_coverage,
    )


def run_analysis(
    video_path: Path,
    annotation_path: Path,
    output: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    """运行 MOT + 视频的离线分析，并保存可视化 JSON 和峰值表格。"""
    video_path = video_path.resolve()
    annotation_path = annotation_path.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    analysis_dir = output / GILL_ANALYSIS_DIRNAME
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tracking = load_mot(annotation_path)
    metrics, video_info = extract_metrics(video_path, tracking)
    # 先保存逐帧原始指标，后续曲线和峰值都可以从该表复查。
    metrics.to_csv(
        analysis_dir / "frame_gray_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    data = build_visualization_data(metrics, video_info)
    data["meta"]["source"] = f"{video_path} + {annotation_path}"
    (output / GILL_DATA_FILENAME).write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    summary = export_peak_tables(data, analysis_dir)
    return data, summary


def analysis_command(argv: list[str]) -> None:
    """命令行入口：分析 MOT 标注与视频，生成鳃动可视化数据。"""
    parser = argparse.ArgumentParser(prog="gill_frequency.py analyze")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--zip", default=str(DEFAULT_ZIP))
    parser.add_argument(
        "--output",
        default=str(ROOT / RESULTS_DIRNAME / GILL_RESULTS_DIRNAME),
    )
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    data, summary = run_analysis(
        Path(args.video),
        Path(args.zip),
        output,
    )
    print(f"Fish regions: {len(data['fish'])}")
    for row in summary.itertuples(index=False):
        print(
            f"fish={row.fish_id} curve={row.curve} "
            f"peaks={row.peak_count} rate_per_min={row.peak_rate_per_min:.2f}"
        )
    print(output / GILL_DATA_FILENAME)


def behavior_command(argv: list[str]) -> None:
    """命令行入口：读取标准 data JSON，输出鳃动频率 JSON。"""
    parser = argparse.ArgumentParser(prog="gill_frequency.py api")
    parser.add_argument("--input", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--min-distance-seconds", type=float, default=0.25)
    parser.add_argument("--prominence", type=float, default=0.05)
    parser.add_argument("--minimum-coverage", type=float, default=0.80)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    # 同时接受裸 data 数组和 {"data": [...]} 两种 JSON 顶层形式。
    records = payload if isinstance(payload, list) else payload.get("data")
    if not isinstance(records, list):
        raise ValueError("Input JSON must be a data array or contain a data array")
    result = calculate_gill_frequency(
        records,
        Path(args.video),
        window_seconds=args.window_seconds,
        min_distance_seconds=args.min_distance_seconds,
        prominence=args.prominence,
        minimum_coverage=args.minimum_coverage,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Records: {len(result['data'])}")
    print(output.resolve())


def main() -> None:
    # Python 直接调用示例（不经过命令行）：
    # from gill_frequency import calculate_gill_frequency
    # result = calculate_gill_frequency(data, video_path="datasets/video.mp4")
    commands = {
        "analyze": analysis_command,
        "api": behavior_command,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage:")
        print("  python gill_frequency.py analyze")
        print("  python gill_frequency.py api --input input.json --video video.mp4 --output output.json")
        return
    command = sys.argv[1]
    commands[command](sys.argv[2:])


if __name__ == "__main__":
    main()
