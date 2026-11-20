"""鱼体几何中心速度指标。

指标定义：先对 SAM segmentation 的全部前景像素求算术中心，再对同一
track_id 最近 10 个有效帧的几何中心与帧号做线性拟合，用拟合斜率计算速度：

    velocity = linear_fit(mask_center, frame_index) * fps
    speed = hypot(velocity_x, velocity_y)

窗口未满 10 帧时速度返回 0。速度方向与图像坐标系一致：0° 向右、
90° 向下、180° 向左、270° 向上。

主要数据流：
标准 data 记录 -> SAM RLE 掩膜 -> 掩膜几何中心 -> 10 帧滑动窗口拟合 ->
速度/速度分量/方向 -> 逐帧结果、轨迹汇总和可视化数据。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common.segmentation_utils import mask_centroid, rle_to_mask


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT / "datasets" / "angle.mp4"
DEFAULT_MOT = ROOT / "datasets" / "angle"
DEFAULT_MASK_SHAPES = ROOT / "results" / "angle" / "mask_shapes.json"
DEFAULT_OUTPUT = ROOT / "results" / "speed"
DEFAULT_FPS = 30.0


def _validate_fps(value: float, source: str) -> float:
    fps = float(value)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{source} must be a positive finite number")
    return fps


def _record_fps(record: dict) -> float | None:
    """从标准记录的 video_info 中读取常见 FPS 字段。"""
    video_info = record.get("video_info")
    if not isinstance(video_info, dict):
        return None
    for key in ("fps", "frame_rate", "frameRate", "video_fps"):
        value = video_info.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            fps = float(value)
            if fps > 0:
                return fps
    return None


def _resolve_fps(data: list[dict], fps: float | None) -> float:
    if fps is not None:
        return _validate_fps(fps, "fps")
    for record in data:
        if isinstance(record, dict):
            record_fps = _record_fps(record)
            if record_fps is not None:
                return record_fps
    return DEFAULT_FPS


def _deduplicate_observations(
    observations: list[dict[str, object]],
) -> list[dict[str, object]]:
    """同一轨迹同一帧只保留最后一条有效观测。"""
    by_frame = {
        int(observation["frame_id"]): observation
        for observation in observations
    }
    return [by_frame[frame_id] for frame_id in sorted(by_frame)]


def extract_center_series(
    data: list[dict],
    fps: float | None = None,
) -> tuple[dict[int, list[dict[str, object]]], float]:
    """从标准记录计算每条鱼的掩膜几何中心时间序列。

    输入记录必需包含 track_id、frame_id 和 segmentation。几何中心是掩膜
    全部前景像素坐标的算术均值，不是 bbox 中心。

    Returns:
        ({track_id: observations}, resolved_fps)
    """
    if not isinstance(data, list) or not data:
        raise ValueError("data must be a non-empty standard record array")
    resolved_fps = _resolve_fps(data, fps)
    series: dict[int, list[dict[str, object]]] = {}

    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise TypeError(f"data[{index}] must be an object")
        if "track_id" not in record or "frame_id" not in record:
            raise ValueError(f"data[{index}] requires track_id and frame_id")
        segmentation = record.get("segmentation")
        if not isinstance(segmentation, dict):
            raise ValueError(f"data[{index}] requires segmentation")

        mask = rle_to_mask(segmentation)
        center_x, center_y = mask_centroid(mask)
        track_id = int(record["track_id"])
        frame_id = int(record["frame_id"])
        bbox = record.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            bbox = [center_x, center_y, 0.0, 0.0]
        series.setdefault(track_id, []).append(
            {
                "frame_id": frame_id,
                "center_x": center_x,
                "center_y": center_y,
                "bbox": [float(value) for value in bbox[:4]],
                "area": float(mask.sum()),
            }
        )

    for track_id, observations in series.items():
        series[track_id] = _deduplicate_observations(observations)
    return series, resolved_fps


def calculate_speed_series(
    center_series: dict[int, list[dict[str, object]]],
    fps: float,
    window_frames: int = 10,
) -> dict[int, list[dict[str, object]]]:
    """用固定长度滑动窗口拟合几何中心轨迹并计算速度。"""
    resolved_fps = _validate_fps(fps, "fps")
    if window_frames < 2:
        raise ValueError("window_frames must be at least 2")
    result: dict[int, list[dict[str, object]]] = {}

    for track_id, observations in sorted(center_series.items()):
        ordered = sorted(observations, key=lambda item: int(item["frame_id"]))
        speed_rows: list[dict[str, object]] = []
        previous: dict[str, object] | None = None
        window: list[dict[str, object]] = []
        for observation in ordered:
            frame_id = int(observation["frame_id"])
            center_x = float(observation["center_x"])
            center_y = float(observation["center_y"])
            window.append(observation)
            if len(window) > window_frames:
                window.pop(0)

            if previous is None:
                dx = 0.0
                dy = 0.0
                displacement = 0.0
                previous_frame = frame_id
            else:
                previous_frame = int(previous["frame_id"])
                dx = center_x - float(previous["center_x"])
                dy = center_y - float(previous["center_y"])
                displacement = math.hypot(dx, dy)

            window_ready = len(window) >= window_frames
            first_frame = int(window[0]["frame_id"])
            dt_seconds = (frame_id - first_frame) / resolved_fps
            window_span_frames = frame_id - first_frame
            window_dx = center_x - float(window[0]["center_x"])
            window_dy = center_y - float(window[0]["center_y"])
            window_displacement = math.hypot(window_dx, window_dy)

            if window_ready and window_span_frames > 0:
                frames = np.asarray(
                    [int(item["frame_id"]) for item in window],
                    dtype=np.float64,
                )
                xs = np.asarray(
                    [float(item["center_x"]) for item in window],
                    dtype=np.float64,
                )
                ys = np.asarray(
                    [float(item["center_y"]) for item in window],
                    dtype=np.float64,
                )
                centered_frames = frames - frames.mean()
                denominator = float(np.dot(centered_frames, centered_frames))
                velocity_x = (
                    float(
                        np.dot(
                            centered_frames,
                            xs - xs.mean(),
                        )
                    )
                    / denominator
                    * resolved_fps
                )
                velocity_y = (
                    float(
                        np.dot(
                            centered_frames,
                            ys - ys.mean(),
                        )
                    )
                    / denominator
                    * resolved_fps
                )
                speed = math.hypot(velocity_x, velocity_y)
                direction = math.degrees(
                    math.atan2(velocity_y, velocity_x)
                ) % 360.0
            else:
                velocity_x = 0.0
                velocity_y = 0.0
                speed = 0.0
                direction = 0.0

            speed_rows.append(
                {
                    "track_id": int(track_id),
                    "frame_id": frame_id,
                    "previous_frame": previous_frame,
                    "frame_range": [
                        int(item["frame_id"]) for item in window
                    ],
                    "center_x": center_x,
                    "center_y": center_y,
                    "dx_px": dx,
                    "dy_px": dy,
                    "displacement_px": displacement,
                    "window_dx_px": window_dx,
                    "window_dy_px": window_dy,
                    "window_displacement_px": window_displacement,
                    "dt_s": dt_seconds,
                    "window_ready": window_ready,
                    "window_samples": len(window),
                    "window_span_frames": window_span_frames,
                    "velocity_x_px_s": velocity_x,
                    "velocity_y_px_s": velocity_y,
                    "speed_px_s": speed,
                    "direction_deg": direction,
                    "bbox": observation.get("bbox", [center_x, center_y, 0.0, 0.0]),
                    "area": float(observation.get("area", 0.0)),
                }
            )
            previous = observation

        result[int(track_id)] = speed_rows
    return result


def summarize_speed_series(
    speed_series: dict[int, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """汇总每条轨迹的速度统计量和累计中心位移。"""
    rows: list[dict[str, object]] = []
    for track_id, observations in sorted(speed_series.items()):
        speeds = np.asarray(
            [
                float(item["speed_px_s"])
                for item in observations
                if bool(item["window_ready"])
            ],
            dtype=float,
        )
        displacements = np.asarray(
            [float(item["displacement_px"]) for item in observations],
            dtype=float,
        )
        latest = observations[-1]
        rows.append(
            {
                "track_id": int(track_id),
                "samples": len(observations),
                "speed_samples": int(speeds.size),
                "mean_speed_px_s": (
                    round(float(speeds.mean()), 6) if speeds.size else 0.0
                ),
                "median_speed_px_s": (
                    round(float(np.median(speeds)), 6) if speeds.size else 0.0
                ),
                "std_speed_px_s": (
                    round(float(speeds.std()), 6) if speeds.size else 0.0
                ),
                "max_speed_px_s": (
                    round(float(speeds.max()), 6) if speeds.size else 0.0
                ),
                "latest_speed_px_s": round(float(latest["speed_px_s"]), 6),
                "total_distance_px": round(float(displacements.sum()), 6),
                "mean_center_x": round(
                    float(
                        np.mean([float(item["center_x"]) for item in observations])
                    ),
                    3,
                ),
                "mean_center_y": round(
                    float(
                        np.mean([float(item["center_y"]) for item in observations])
                    ),
                    3,
                ),
            }
        )
    return rows


def _standard_metric_rows(
    speed_series: dict[int, list[dict[str, object]]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for track_id, observations in sorted(speed_series.items()):
        for item in observations:
            base = {
                "track_id": int(track_id),
                "frame_id": int(item["frame_id"]),
                "frame_range": list(item["frame_range"]),
            }
            output.extend(
                [
                    {
                        **base,
                        "behavior_indicator": "segmentation_center_x",
                        "behavior_value": round(float(item["center_x"]), 6),
                        "unit": "pixel",
                    },
                    {
                        **base,
                        "behavior_indicator": "segmentation_center_y",
                        "behavior_value": round(float(item["center_y"]), 6),
                        "unit": "pixel",
                    },
                    {
                        **base,
                        "behavior_indicator": "fish_velocity_x",
                        "behavior_value": round(
                            float(item["velocity_x_px_s"]),
                            6,
                        ),
                        "unit": "pixels/s",
                    },
                    {
                        **base,
                        "behavior_indicator": "fish_velocity_y",
                        "behavior_value": round(
                            float(item["velocity_y_px_s"]),
                            6,
                        ),
                        "unit": "pixels/s",
                    },
                    {
                        **base,
                        "behavior_indicator": "fish_speed",
                        "behavior_value": round(float(item["speed_px_s"]), 6),
                        "unit": "pixels/s",
                    },
                    {
                        **base,
                        "behavior_indicator": "fish_velocity_direction",
                        "behavior_value": round(float(item["direction_deg"]), 6),
                        "unit": "degree",
                    },
                ]
            )
    return output


def calculate_fish_speed(
    data: list[dict],
    fps: float | None = None,
    window_frames: int = 10,
) -> dict[str, list[dict[str, object]]]:
    """用滑动窗口计算每条鱼每帧的几何中心、速度分量和方向。

    `segmentation_center_x/y` 为掩膜全部前景像素的算术中心；`fish_speed`
    和 `fish_velocity_direction` 来自最近 `window_frames` 个有效中心坐标的
    线性拟合，单位分别为 `pixels/s` 和 `degree`。窗口未满时速度返回 0。
    """
    center_series, resolved_fps = extract_center_series(data, fps)
    speed_series = calculate_speed_series(
        center_series,
        resolved_fps,
        window_frames=window_frames,
    )
    return {"data": _standard_metric_rows(speed_series)}


def _group_center_observations(
    observations: list[dict[str, object]],
) -> dict[int, list[dict[str, object]]]:
    series: dict[int, list[dict[str, object]]] = {}
    for observation in observations:
        series.setdefault(int(observation["track_id"]), []).append(observation)
    for track_id, track_rows in series.items():
        series[track_id] = _deduplicate_observations(track_rows)
    return series


def load_mask_shape_centers(
    path: Path,
    width: int,
    height: int,
    max_frames: int | None = None,
) -> tuple[dict[int, list[dict[str, object]]], float | None]:
    """从角度分析保存的多边形轮廓重新计算逐帧掩膜几何中心。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("mask_shapes.json must contain a samples array")
    meta = payload.get("meta", {})
    meta_fps = meta.get("fps") if isinstance(meta, dict) else None
    resolved_fps = None
    if isinstance(meta_fps, (int, float)) and float(meta_fps) > 0:
        resolved_fps = float(meta_fps)

    observations: list[dict[str, object]] = []
    for sample in samples:
        frame_id = int(sample["frame"])
        if max_frames is not None and frame_id > max_frames:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in sample.get("polygons", []):
            points = np.asarray(polygon, dtype=np.int32)
            if len(points) >= 3:
                cv2.fillPoly(mask, [points], 1)
        if not mask.any():
            continue
        center_x, center_y = mask_centroid(mask)
        bbox = sample.get("box")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            bbox = [center_x, center_y, 0.0, 0.0]
        observations.append(
            {
                "track_id": int(sample["track_id"]),
                "frame_id": frame_id,
                "center_x": center_x,
                "center_y": center_y,
                "bbox": [float(value) for value in bbox[:4]],
                "area": float(mask.sum()),
            }
        )
    return _group_center_observations(observations), resolved_fps


def load_mot_centers(
    path: Path,
    max_frames: int | None = None,
) -> dict[int, list[dict[str, object]]]:
    """缺少掩膜轮廓时，用 MOT 框中心生成近似速度时间序列。"""
    mot_file = path / "gt" / "gt.txt" if path.is_dir() else path
    dataframe = pd.read_csv(
        mot_file,
        header=None,
        names=[
            "frame",
            "track_id",
            "x",
            "y",
            "w",
            "h",
            "mark",
            "class_id",
            "confidence",
        ],
    )
    dataframe = dataframe[dataframe["confidence"] > 0]
    if max_frames is not None:
        dataframe = dataframe[dataframe["frame"] <= max_frames]
    observations: list[dict[str, object]] = []
    for row in dataframe.itertuples(index=False):
        center_x = float(row.x + row.w / 2.0)
        center_y = float(row.y + row.h / 2.0)
        observations.append(
            {
                "track_id": int(row.track_id),
                "frame_id": int(row.frame),
                "center_x": center_x,
                "center_y": center_y,
                "bbox": [float(row.x), float(row.y), float(row.w), float(row.h)],
                "area": float(row.w * row.h),
            }
        )
    return _group_center_observations(observations)


def run_analysis(
    center_series: dict[int, list[dict[str, object]]],
    output: Path,
    fps: float,
    video_path: Path,
    center_source: str,
    window_frames: int = 10,
) -> None:
    """导出滑动窗口速度、轨迹汇总、标准行为和页面元数据。"""
    output.mkdir(parents=True, exist_ok=True)
    speed_series = calculate_speed_series(
        center_series,
        fps,
        window_frames=window_frames,
    )
    frame_rows: list[dict[str, object]] = []
    for observations in speed_series.values():
        for item in observations:
            frame_rows.append(
                {
                    "track_id": int(item["track_id"]),
                    "frame_id": int(item["frame_id"]),
                    "time_s": round((int(item["frame_id"]) - 1) / fps, 6),
                    "center_x": round(float(item["center_x"]), 3),
                    "center_y": round(float(item["center_y"]), 3),
                    "velocity_x_px_s": round(
                        float(item["velocity_x_px_s"]),
                        3,
                    ),
                    "velocity_y_px_s": round(
                        float(item["velocity_y_px_s"]),
                        3,
                    ),
                    "speed_px_s": round(float(item["speed_px_s"]), 3),
                    "direction_deg": round(float(item["direction_deg"]), 3),
                    "displacement_px": round(
                        float(item["displacement_px"]),
                        3,
                    ),
                    "dt_s": round(float(item["dt_s"]), 6),
                    "window_ready": bool(item["window_ready"]),
                    "window_samples": int(item["window_samples"]),
                    "window_span_frames": int(item["window_span_frames"]),
                    "window_dx_px": round(float(item["window_dx_px"]), 3),
                    "window_dy_px": round(float(item["window_dy_px"]), 3),
                    "window_displacement_px": round(
                        float(item["window_displacement_px"]),
                        3,
                    ),
                    "bbox_x": round(float(item["bbox"][0]), 3),
                    "bbox_y": round(float(item["bbox"][1]), 3),
                    "bbox_w": round(float(item["bbox"][2]), 3),
                    "bbox_h": round(float(item["bbox"][3]), 3),
                }
            )

    frame_dataframe = pd.DataFrame(frame_rows).sort_values(
        ["frame_id", "track_id"]
    )
    frame_path = output / "frame_speed.csv"
    frame_dataframe.to_csv(frame_path, index=False, encoding="utf-8-sig")

    summary_rows = summarize_speed_series(speed_series)
    summary_path = output / "track_speed_summary.csv"
    pd.DataFrame(summary_rows).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    distribution_path = output / "speed_distribution.json"
    distribution_path.write_text(
        json.dumps(
            {"data": _standard_metric_rows(speed_series)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_path_json = output / "speed_summary.json"
    summary_path_json.write_text(
        json.dumps(
            {
                "meta": {
                    "video": str(video_path),
                    "fps": fps,
                    "center_source": center_source,
                    "center_definition": (
                        "Arithmetic mean of all foreground mask pixel coordinates."
                    ),
                    "speed_definition": (
                        "Velocity from a least-squares linear fit of the most "
                        f"recent {window_frames} valid geometric centers "
                        "against frame index."
                    ),
                    "window_frames": window_frames,
                    "behavior_indicator": "fish_speed",
                    "unit": "pixels/s",
                },
                "tracks": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved: {frame_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {distribution_path}")
    print(f"Saved: {summary_path_json}")


def build_parser() -> argparse.ArgumentParser:
    """构建鱼体中心速度命令行参数。"""
    parser = argparse.ArgumentParser(
        prog="speed_analysis.py",
        description="Calculate fish-body center speed from SAM masks.",
    )
    parser.add_argument("--input", help="standard behavior data JSON")
    parser.add_argument("--mot", default=str(DEFAULT_MOT))
    parser.add_argument("--mask-shapes", default=str(DEFAULT_MASK_SHAPES))
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--fps", type=float)
    parser.add_argument("--window-frames", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main() -> None:
    """读取掩膜中心数据，计算速度并写出结果文件。"""
    args = build_parser().parse_args()
    video_path = Path(args.video).resolve()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()

    if args.input:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        data = payload if isinstance(payload, list) else payload.get("data")
        if not isinstance(data, list):
            raise ValueError("input must be a data array or contain data")
        center_series, resolved_fps = extract_center_series(data, args.fps)
        center_source = str(Path(args.input).resolve())
    elif Path(args.mask_shapes).is_file():
        mask_path = Path(args.mask_shapes).resolve()
        center_series, mask_fps = load_mask_shape_centers(
            mask_path,
            width=width,
            height=height,
            max_frames=args.max_frames,
        )
        resolved_fps = (
            _validate_fps(args.fps, "fps")
            if args.fps is not None
            else mask_fps or (video_fps if video_fps > 0 else DEFAULT_FPS)
        )
        center_source = str(mask_path)
    else:
        center_series = load_mot_centers(
            Path(args.mot),
            max_frames=args.max_frames,
        )
        resolved_fps = (
            _validate_fps(args.fps, "fps")
            if args.fps is not None
            else (video_fps if video_fps > 0 else DEFAULT_FPS)
        )
        center_source = f"MOT boxes: {Path(args.mot).resolve()}"

    run_analysis(
        center_series,
        Path(args.output).resolve(),
        resolved_fps,
        video_path,
        center_source,
        window_frames=args.window_frames,
    )


if __name__ == "__main__":
    main()
