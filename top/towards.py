"""鱼头朝向分析：读取 geometry.py 共享中轴线和头尾质量判断。"""
from pathlib import Path

# ==================== 只需要修改这里 ====================
SEGMENTATION_DIR = Path(r"E:\Phd\temperature\sam3\top\outputs\20261114_095054_184431")
OUTPUT_DIR = Path(r"E:\Phd\temperature\sam3\top\analysis\towards")
HEAD_HINTS = {}  # 与 geometry.py 相同；有歧义时指定该ID初次头端方向。
USE_FILENAME_TIMESTAMPS = True
HORIZONTAL_TOLERANCE_DEG = 10.0  # 距水平向右/向左均不超过此角度算水平。
SAVE_VIDEO = True
# ========================================================

import csv
import json
import math
import traceback
from collections import defaultdict
from datetime import datetime

import numpy as np


def heading_degrees(dx, dy):
    """图像坐标 x 向右、y 向下；头朝右=0，上=+90，下=-90。"""
    if not (math.isfinite(dx) and math.isfinite(dy)) or math.hypot(dx, dy) < 1e-9:
        return math.nan
    return math.degrees(math.atan2(-dy, dx))


def tilt_degrees(dx, dy):
    """相对水平线的倾角：左右水平均为 0，朝上正、朝下负，范围 [-90, 90]。"""
    if not (math.isfinite(dx) and math.isfinite(dy)) or math.hypot(dx, dy) < 1e-9:
        return math.nan
    return math.degrees(math.atan2(-dy, abs(dx)))


def circular_statistics(angles):
    """单位向量平均，正确处理 +179/-179；合向量为零时均值无定义。"""
    values = np.asarray(angles, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    radians = np.deg2rad(values)
    x, y = float(np.cos(radians).mean()), float(np.sin(radians).mean())
    strength = min(1.0, math.hypot(x, y))
    mean = math.degrees(math.atan2(y, x)) if strength > 1e-8 else math.nan
    return mean, strength


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def prepare_rows(records):
    """纯计算：从共享几何记录计算倾角、完整航向与连续段展开角，不改输入。

    输入包含frame/time_s/id/valid/head_dx/head_dy；方向采用图像坐标x右y下。
    tilt_deg=atan2(-dy,abs(dx))，左右水平均0，上正下负；raw_heading_deg
    保留左/右象限，unwrapped_heading_deg仅用于连续有效段曲线。invalid
    的角度始终NaN。帧号可从1开始；缺整数帧、无效帧和时间倒退都切段。
    不需要image_paths，不加载geometry/SAM3，不写文件；空输入返回[]。
    """
    rows, previous, segment_numbers = [], {}, defaultdict(int)
    for record in sorted(records, key=lambda r: (int(r["frame"]), int(r["id"]))):
        identity, frame = int(record["id"]), int(record["frame"])
        time_s = float(record["time_s"])
        dx, dy = record.get("head_dx"), record.get("head_dy")
        angle = heading_degrees(float(dx), float(dy)) if finite(dx) and finite(dy) else math.nan
        valid = bool(record.get("valid", False)) and finite(angle) and finite(time_s)
        reason = str(record.get("quality_reason", ""))
        if not valid and not reason:
            reason = "invalid_geometry_or_head_direction"
        row = dict(frame=frame, filename=record.get("filename", ""), time_seconds=time_s,
                   id=identity, cx=record.get("cx"), cy=record.get("cy"), area=record.get("area"),
                   tilt_deg=math.nan, raw_heading_deg=math.nan, unwrapped_heading_deg=math.nan,
                   segment_id=None, direction="invalid", valid=valid, quality_reason=reason,
                   head_confidence=record.get("head_confidence"))
        if valid:
            prev = previous.get(identity)
            continuous = prev is not None and frame == prev["frame"] + 1 and time_s > prev["time_seconds"]
            if continuous:
                # 选择相邻角度的最短弧；这不填补缺帧，也不能证明帧间没有快速整圈旋转。
                delta = math.degrees(math.atan2(math.sin(math.radians(angle - prev["raw_heading_deg"])),
                                                math.cos(math.radians(angle - prev["raw_heading_deg"]))))
                unwrapped = prev["unwrapped_heading_deg"] + delta
            else:
                segment_numbers[identity] += 1
                unwrapped = angle
            vertical = math.sin(math.radians(angle))
            horizontal = abs(vertical) <= math.sin(math.radians(HORIZONTAL_TOLERANCE_DEG)) + 1e-12
            row.update(tilt_deg=tilt_degrees(float(dx), float(dy)),
                       raw_heading_deg=angle, unwrapped_heading_deg=unwrapped,
                       segment_id=segment_numbers[identity],
                       direction="horizontal" if horizontal else ("up" if vertical > 0 else "down"))
            previous[identity] = row
        else:
            previous.pop(identity, None)
        rows.append(row)
    return rows


def summarize_rows(rows):
    """按ID汇总有效观测：倾角普通均值，完整航向圆均值及合向量长度。

    0/180度相反方向的圆均值无定义时保留NaN，不错误取90度；up/down/
    horizontal计数只统计有效记录。空输入返回[]；单帧合法并有明确定义。
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row["id"]].append(row)
    summaries = []
    for identity, group in sorted(groups.items()):
        valid = [r for r in group if r["valid"]]
        mean, strength = circular_statistics([r["raw_heading_deg"] for r in valid])
        tilts = [r["tilt_deg"] for r in valid if finite(r["tilt_deg"])]
        summaries.append(dict(id=identity, observed_frames=len(group), valid_frames=len(valid),
                              invalid_frames=len(group) - len(valid), valid_fraction=len(valid) / len(group),
                              continuous_segments=len({r["segment_id"] for r in valid}),
                              mean_tilt_deg=float(np.mean(tilts)) if tilts else math.nan,
                              circular_mean_heading_deg=mean, mean_resultant_length=strength,
                              up_frames=sum(r["direction"] == "up" for r in valid),
                              down_frames=sum(r["direction"] == "down" for r in valid),
                              horizontal_frames=sum(r["direction"] == "horizontal" for r in valid)))
    return summaries


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None or
                                  isinstance(row.get(k), (float, np.floating)) and not finite(row[k])
                                  else row.get(k)) for k in fields})


def plot_trends(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["id"]].append(row)
    for identity, group in sorted(groups.items()):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, constrained_layout=True)
        valid = [r for r in group if r["valid"]]
        invalid = [r for r in group if not r["valid"] and finite(r["time_seconds"])]
        axes[0].scatter([r["time_seconds"] for r in valid], [r["tilt_deg"] for r in valid],
                        s=13, color="#176B96", label="Valid tilt (+up / -down)")
        if invalid:
            axes[0].scatter([r["time_seconds"] for r in invalid], [-0.07] * len(invalid), marker="x",
                            color="#D04C41", s=18, transform=axes[0].get_xaxis_transform(),
                            clip_on=False, label="Invalid (no angle)")
        segments = defaultdict(list)
        for row in valid:
            segments[row["segment_id"]].append(row)
        for segment in segments.values():
            axes[1].plot([r["time_seconds"] for r in segment],
                         [r["unwrapped_heading_deg"] for r in segment], ".-", linewidth=1, markersize=3)
        if not valid:
            for ax in axes:
                ax.text(.5, .5, "No valid head direction", transform=ax.transAxes, ha="center", va="center")
        axes[0].set_title(f"Fish ID {identity}: tilt relative to the horizontal line")
        axes[0].set_ylabel("Tilt (deg; +up / -down)")
        axes[0].set_ylim(-95, 95)
        axes[0].set_yticks([-90, -45, 0, 45, 90])
        axes[1].set_ylabel("Unwrapped heading (deg)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_title("Unwrapped separately in each continuous valid segment", fontsize=10)
        for ax in axes:
            ax.axhline(0, color="gray", linewidth=.7, alpha=.7)
            ax.grid(alpha=.2)
        if valid or invalid:
            axes[0].legend(loc="best", fontsize=8)
        fig.savefig(plot_dir / f"fish_{identity:04d}.png", dpi=160)
        plt.close(fig)


def read_image(path):
    """仅导出视频时读取原图；数值计算无需任何图片路径。"""
    import cv2
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取原始图片：{path}")
    return image


def draw_fish(image, record, row):
    """有效鱼绘制前半中轴与指头箭头；无效鱼只显示INVALID。"""
    import cv2
    if not finite(record.get("cx")) or not finite(record.get("cy")):
        return
    center = (int(round(record["cx"])), int(round(record["cy"])))
    valid = row["valid"]
    color = (40, 220, 80) if valid else (50, 60, 240)
    cv2.circle(image, center, 4, color, -1, cv2.LINE_AA)
    if valid:
        line = record.get("centerline")
        line = np.asarray(line, dtype=float) if line is not None else np.empty((0, 2))
        if line.ndim == 2 and line.shape[1] == 2 and len(line) >= 2 and np.isfinite(line).all():
            front = line[:len(line) // 2 + 1]
            cv2.polylines(image, [np.rint(front).astype(np.int32)], False, (0, 210, 255), 3, cv2.LINE_AA)
            anchor = tuple(np.rint(front[-1]).astype(int))
        else:
            anchor = center
        ax, ay = anchor
        length = max(30, min(80, int(math.sqrt(max(float(record.get("area", 1)), 1)))))
        cv2.line(image, (ax - length, ay), (ax + length, ay), (200, 200, 200), 1, cv2.LINE_AA)
        dx, dy = float(record["head_dx"]), float(record["head_dy"])
        magnitude = math.hypot(dx, dy)
        endpoint = (int(round(ax + length * dx / magnitude)), int(round(ay + length * dy / magnitude)))
        cv2.arrowedLine(image, anchor, endpoint, color, 2, cv2.LINE_AA, tipLength=.22)
        label = f"ID {row['id']}  tilt {row['tilt_deg']:+.1f} deg"
    else:
        label = f"ID {row['id']}  INVALID"
    x = max(2, min(center[0] + 7, image.shape[1] - 220))
    y = max(18, min(center[1] - 10, image.shape[0] - 8))
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .46, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .46, color, 1, cv2.LINE_AA)


ROW_FIELDS = ["frame", "filename", "time_seconds", "id", "cx", "cy", "area", "tilt_deg",
              "raw_heading_deg", "unwrapped_heading_deg", "segment_id", "direction", "valid",
              "quality_reason", "head_confidence"]
SUMMARY_FIELDS = ["id", "observed_frames", "valid_frames", "invalid_frames", "valid_fraction",
                  "continuous_segments", "mean_tilt_deg", "circular_mean_heading_deg",
                  "mean_resultant_length", "up_frames", "down_frames", "horizontal_frames"]


def make_video(output, records, rows, meta):
    """按meta.frame_ids中的实际帧号导出视频；缺原图时返回跳过说明。

    不再假设帧号从0开始，允许[1,2,4]。保持列表顺序、一图一视频帧；缺帧
    不伪造补图。视频播放仍使用标称fps，真实时间保留在CSV。没有任何原图
    或部分原图缺失时跳过整个视频，不影响数值导出，也不产出截短视频。
    """
    paths = {int(k): v for k, v in meta.get("image_paths", {}).items()}
    frame_ids = [int(f) for f in meta.get("frame_ids", sorted(paths or {r["frame"] for r in records}))]
    result = dict(video_saved=False, video_skip_reason="", video_frames=0, frame_ids=frame_ids)
    if not frame_ids:
        result["video_skip_reason"] = "no_frames"
        return result
    missing = [f for f in frame_ids if not paths.get(f) or not Path(paths[f]).is_file()]
    if missing:
        result.update(video_skip_reason="missing_source_images", missing_image_frames=missing)
        return result
    width, height = int(meta.get("width") or 0), int(meta.get("height") or 0)
    # 空数据或纯数值调用可以不知道fps；不把未知值补成猜测帧率。
    fps = float(meta["fps"]) if meta.get("fps") is not None else math.nan
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
        result["video_skip_reason"] = "invalid_video_metadata"
        return result
    import cv2
    by_frame, by_key = defaultdict(list), {(r["frame"], r["id"]): r for r in rows}
    for record in records:
        by_frame[int(record["frame"])].append(record)
    writer = cv2.VideoWriter(str(output / "towards.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("无法创建towards.mp4，请检查输出路径和OpenCV。")
    try:
        for frame in frame_ids:
            image = read_image(paths[frame])
            if image.shape[:2] != (height, width):
                raise ValueError(f"第{frame}帧尺寸与分割记录不一致。")
            for record in by_frame[frame]:
                draw_fish(image, record, by_key[(frame, int(record["id"]))])
            valid_count = sum(by_key[(frame, int(r["id"]))]["valid"] for r in by_frame[frame])
            text = f"Head tilt | frame ID {frame} | valid {valid_count}/{len(by_frame[frame])}"
            cv2.putText(image, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .57, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .57, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Tilt: 0=horizontal  +90=up  -90=down", (12, 47),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(image)
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(output / "towards.mp4"))
    count = 0
    try:
        while True:
            ok, decoded = cap.read()
            if not ok:
                break
            if decoded.shape[:2] != (height, width):
                raise RuntimeError("导出视频尺寸不一致。")
            count += 1
    finally:
        cap.release()
    if count != len(frame_ids):
        raise RuntimeError(f"导出视频仅能读取{count}/{len(frame_ids)}帧。")
    result.update(video_saved=True, video_skip_reason="", video_frames=count)
    return result


def export_results(records, meta, output: Path, save_video=True, rows=None):
    """导出朝向两张CSV、每鱼趋势图、可选视频和run_info.json，返回(rows,summaries)。

    records/meta为geometry共享接口的输出；原图不是数值分析的必要输入。
    rows=None时调用prepare_rows，否则复用传入行，避免总入口重复计算。
    output是调用方给定的结果目录；同名文件会写入此目录，应为每批选新目录。
    空输入输出有表头的CSV和data_quality='no_records'，单帧照常输出。
    valid与quality_reason保留几何质量，视频跳过由独立video_skip_reason记录，
    不将无原图误判为无有效角度。仅main为独立运行创建时间戳目录。
    """
    if not 0 <= HORIZONTAL_TOLERANCE_DEG < 90:
        raise ValueError("水平角容差应在0至90度之间（不含90）。")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = prepare_rows(records)
    summaries = summarize_rows(rows)
    valid_count = sum(bool(r["valid"]) for r in rows)
    info = dict(status="running", segmentation_directory=str(meta.get("segmentation_dir", "")),
                frame_count=int(meta.get("frame_count", len({r["frame"] for r in records}))),
                fps=float(meta["fps"]) if meta.get("fps") is not None else None,
                time_source=meta.get("time_source"), observed_records=len(rows),
                head_hints=meta.get("head_hints"),
                use_filename_timestamps=meta.get("use_filename_timestamps"),
                requested_use_filename_timestamps=meta.get("requested_use_filename_timestamps"),
                used_filename_timestamps=meta.get("used_filename_timestamps"),
                save_video_requested=bool(save_video),
                valid_records=valid_count, invalid_records=len(rows) - valid_count,
                data_quality="no_records" if not rows else "ok" if valid_count else "no_valid_geometry",
                horizontal_tolerance_deg=HORIZONTAL_TOLERANCE_DEG,
                tilt_definition="Primary angle: atan2(-dy,abs(dx)) in degrees, [-90,90]; horizontal left/right=0; up positive; down negative",
                heading_definition="Anterior centerline fitted axis pointing toward the head; image-plane atan2(-dy,dx), degrees",
                convention="tilt is signed angle to horizontal line; full heading retains left/right: 0 right, +/-180 left; 2D only",
                unwrap="Shortest arc within consecutive valid frames of each ID only; resets after gaps/invalid/time reversal",
                summary="Equal weight per valid observation; mean tilt uses arithmetic mean; full heading uses circular mean and resultant length",
                video_timing="One listed source image per video frame at nominal FPS; actual source frame IDs retained")
    def save_info():
        (output / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    save_info()
    try:
        write_csv(output / "heading_per_frame.csv", rows, ROW_FIELDS)
        write_csv(output / "heading_summary.csv", summaries, SUMMARY_FIELDS)
        plot_trends(output, rows)
        if save_video:
            info.update(make_video(output, records, rows, meta))
        else:
            info.update(video_saved=False, video_skip_reason="disabled", video_frames=0)
        info.update(status="complete", fish_ids=len(summaries))
        save_info()
    except BaseException:
        info.update(status="failed", error=traceback.format_exc())
        save_info()
        raise
    return rows, summaries


def main():
    """独立运行入口：读取共享几何，导出到新的时间戳目录。"""
    from geometry import load_geometry
    records, meta = load_geometry(SEGMENTATION_DIR, head_hints=HEAD_HINTS,
                                  use_filename_timestamps=USE_FILENAME_TIMESTAMPS)
    output = OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    export_results(records, meta, output, save_video=SAVE_VIDEO)
    print(f"朝向分析完成：{output}", flush=True)
    return output


if __name__ == "__main__":
    main()


