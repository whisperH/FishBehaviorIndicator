"""侧面视角模块，维护负责人：chenxi。
所有路径默认指向合并项目；可用命令行参数覆盖。只操作自身 Side 目录。
"""
from __future__ import annotations
from pathlib import Path
import sys, argparse, csv, json, math, os, time, io, zipfile
import numpy as np
import pandas as pd
import cv2
SIDE_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SIDE_ROOT))
from common.segmentation_utils import *
INPUT=Path(r"E:\Phd\FishBehavior\Side\Tracker\Output\body\result.csv")
OUTPUT=Path(r"E:\Phd\FishBehavior\Side\Behavior\Output\speed")
WINDOW_FRAMES=10  # 最近10个有效观测，不要求帧号连续；保留原算法的缺帧处理。
DEFAULT_FPS=30.0
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


def run(input_csv=INPUT,output=OUTPUT,window_frames=WINDOW_FRAMES):
    """只读取body/result.csv，不读取视频和掩膜。速度=最近N观测的线性拟合斜率×fps。
    保留px/s、图像右正下正、无符号速率、窗口未满零占位和window_ready标记。
    """
    rows,meta=read_body(input_csv);series={}
    for r in observations(rows):
        if not flag(r['position_valid']):continue
        series.setdefault(int(r['track_id']),[]).append(dict(track_id=int(r['track_id']),frame_id=int(r['frame_id']),center_x=float(r['center_x']),center_y=float(r['center_y']),bbox=[float(r[k]) for k in ['bbox_x','bbox_y','bbox_w','bbox_h']],area=float(r['area'] or 0)))
    if not series:raise ValueError('No valid body positions')
    result=run_analysis(series,Path(output),float(meta['fps']),Path(meta['video_path']),str(Path(input_csv).resolve()),window_frames)
    # 保存基础结果的批次、来源和原视频指纹。页面据此识别旧分析，不改变任何CSV或计算公式。
    summary_path=Path(output)/'speed_summary.json'
    payload=json.loads(summary_path.read_text('utf-8'))
    payload['meta'].update(meta)
    payload['meta']['input_csv']=str(Path(input_csv).resolve())
    payload['meta']['center_definition']='Arithmetic mean of foreground pixels after filling simplified body-contour polygons, preserving the original side-view speed/water-layer measurement convention.'
    summary_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default=str(INPUT));p.add_argument('--output',default=str(OUTPUT));p.add_argument('--window-frames',type=int,default=WINDOW_FRAMES);a=p.parse_args();run(a.input,a.output,a.window_frames)
