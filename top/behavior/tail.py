"""摆尾数值分析和结果导出。几何提取统一由同目录 geometry.py 提供。

纯计算：analyze_tail(records, meta)，不读原图、不运行SAM3、不写文件。
导出：export_results(records, meta, output, ...)，原图缺失时仍导出全部数值。
直接运行：修改下方路径后运行本文件；与综合入口共用固定目录，重复运行覆盖摆尾结果。
"""
from pathlib import Path
# 支持绝对路径直接运行本文件；正常包导入时无需修改模块搜索路径。
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ==================== 日常修改区 ====================
SEGMENTATION_DIR = Path(r"E:\Phd\temperature\sam3\top\output\tracking")
OUTPUT_DIR = Path(r"E:\Phd\temperature\sam3\top\output\behavior\tail")
USE_FILENAME_TIMESTAMPS = True  # 目录输入优先读取文件名采集时间；具体来源保存在meta。
HEAD_HINTS = {}                # 例如{0: "left"}，含义和优先级由geometry.py统一实现。
MIN_FREQUENCY_HZ = 0.4         # 接受的最低摆尾频率，单位Hz。
MAX_FREQUENCY_HZ = 6.0         # 同时受每周期至少5个样本限制；不突破采样率上限。
MIN_AMPLITUDE_DEG = 3.0        # 去趋势后角度95/5百分位半差，低于此值不报告可信频率。
MIN_PEAK_PROMINENCE_DEG = 3.0  # 同相位峰的最小突出程度，避免噪声被计为摆尾。
MIN_COMPLETE_CYCLES = 2       # 至少两个完整同相位峰间隔；单个摆动不够估频。
MAX_PERIOD_CV = 0.35          # 周期间隔变异系数上限；超过则标记不规则周期。
SAVE_VIDEO = True             # False只禁用视频；有原图仍保存首帧预览。
# ====================================================

import csv
import json
from data_format import prepare_output_directory
import math
import traceback
from collections import defaultdict, Counter

import numpy as np
from scipy.signal import detrend, find_peaks, periodogram, savgol_filter


def read_image(path):
    """读取原图，支持Windows中文路径；只在导出预览/视频时调用。"""
    import cv2
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取原始图片：{path}")
    return image


def save_image(path, image):
    """按扩展名编码图片；保留Windows中文路径兼容。"""
    import cv2
    ok, encoded = cv2.imencode(Path(path).suffix, image)
    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")
    encoded.tofile(str(path))


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: "" if isinstance(value, (float, np.floating)) and not math.isfinite(value) else value
                          for key, value in row.items()} for row in rows)


def valid_segments(records, fps):
    """将同一ID分成连续有效段，不跨缺帧、无效几何或时间断点。

    records应来自geometry接口，至少含frame/time_s/valid/tail_angle_deg。
    fps是标称采样率；两帧时间差须在(0, 1.8/fps)内。帧号可从1开始，
    但缺少一个整数帧号会切段，绝不跨缺帧插值后计算频率。空输入返回[]。
    """
    segments, current = [], []
    for row in sorted(records, key=lambda r: r["frame"]):
        good = row["valid"] and math.isfinite(row["tail_angle_deg"]) and math.isfinite(row["time_s"])
        if current and (not good or row["frame"] != current[-1]["frame"] + 1
                        or not 0 < row["time_s"] - current[-1]["time_s"] < 1.8 / fps):
            segments.append(current); current = []
        if good:
            current.append(row)
    if current:
        segments.append(current)
    return segments


def estimate_frequency(segment, fps):
    """估计一个连续有效段的摆尾频率，不读文件、不修改输入。

    主频率取同相位正峰间隔倒数的中位周期；频谱仅作独立核对，避免
    对abs(angle)计算导致频率翻倍。smoothed_angles与输入段逐行对应。
    frequency_quality='ok'才表示frequency_hz可用；其余状态保留NaN，
    包括too_short、amplitude_too_small、insufficient_complete_cycles等。
    空段/单帧返回too_short。算法及默认阈值保留原项目的实现。
    """
    result = {"frequency_hz": float("nan"), "spectral_peak_hz": float("nan"),
              "amplitude_deg": float("nan"), "cycles": 0, "period_cv": float("nan"),
              "frequency_quality": "too_short", "peak_times_s": [], "smoothed_angles": []}
    if len(segment) < 7:
        return result
    t = np.asarray([r["time_s"] for r in segment])
    angle = np.degrees(np.unwrap(np.radians([r["tail_angle_deg"] for r in segment])))
    sample_rate = (len(t) - 1) / (t[-1] - t[0])
    # 统一到近等间距，处理采集时间戳33/34毫秒的正常抖动。
    grid = np.linspace(t[0], t[-1], len(t))
    smooth = savgol_filter(np.interp(grid, t, angle), 5, 2)
    wave = detrend(smooth)
    amplitude = (np.percentile(wave, 95) - np.percentile(wave, 5)) / 2
    result.update(amplitude_deg=float(amplitude), smoothed_angles=np.interp(t, grid, smooth).tolist())
    if amplitude < MIN_AMPLITUDE_DEG:
        result["frequency_quality"] = "amplitude_too_small"
        return result
    if t[-1] - t[0] < MIN_COMPLETE_CYCLES / MAX_FREQUENCY_HZ:
        return result
    # 至少每周期5个样本；30fps上限6Hz。频谱仍查看更高频段，以拒绝不稳定信号。
    upper = min(MAX_FREQUENCY_HZ, sample_rate / 5)
    frequency, power = periodogram(wave, fs=(len(grid)-1)/(grid[-1]-grid[0]), window="hann",
                                  nfft=max(256, 4 * len(wave)))
    band = (frequency >= MIN_FREQUENCY_HZ) & (frequency < sample_rate / 2)
    if band.any():
        spectral = float(frequency[band][np.argmax(power[band])])
        result["spectral_peak_hz"] = spectral
        if spectral > upper * 1.001:
            result["frequency_quality"] = "frequency_above_safe_limit"
            return result
    peaks, _ = find_peaks(wave, prominence=max(MIN_PEAK_PROMINENCE_DEG, amplitude * .25),
                         distance=max(1, int(math.floor(sample_rate / upper))))
    result["peak_times_s"] = grid[peaks].tolist()
    intervals = np.diff(grid[peaks])
    result["cycles"] = int(len(intervals))
    if len(intervals) < MIN_COMPLETE_CYCLES:
        result["frequency_quality"] = "insufficient_complete_cycles"
        return result
    estimates = 1 / intervals
    cv = float(np.std(intervals) / np.mean(intervals))
    result["period_cv"] = cv
    if np.any((estimates < MIN_FREQUENCY_HZ) | (estimates > upper * 1.001)):
        result["frequency_quality"] = "period_outside_frequency_range"
    elif cv > MAX_PERIOD_CV:
        result["frequency_quality"] = "irregular_cycles"
    else:
        value = float(1 / np.median(intervals))
        spectral = result["spectral_peak_hz"]
        if math.isfinite(spectral) and abs(spectral - value) > max(.35, .3 * value):
            result["frequency_quality"] = "spectrum_peak_disagrees"
        else:
            result.update(frequency_hz=value, frequency_quality="ok")
    return result


def draw_geometry(image, row):
    """绘制中轴、头尾端点和两段轴向；invalid只标记，不虚构角度。"""
    import cv2
    identity = row["id"]
    if not (math.isfinite(row.get("cx", math.nan)) and math.isfinite(row.get("cy", math.nan))):
        return
    color = tuple(int(v) for v in np.random.default_rng(identity).integers(90, 255, 3))
    cx, cy = int(round(row["cx"])), int(round(row["cy"]))
    if not row["valid"]:
        cv2.putText(image, f"ID {identity}: invalid", (cx - 30, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (80, 80, 255), 1, cv2.LINE_AA)
        return
    line = row["centerline"]
    cv2.polylines(image, [np.rint(line).astype(np.int32)], False, color, 2, cv2.LINE_AA)
    head_point, tail_point = tuple(np.rint(line[0]).astype(int)), tuple(np.rint(line[-1]).astype(int))
    cv2.circle(image, head_point, 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(image, tail_point, 5, (255, 150, 20), -1, cv2.LINE_AA)
    for position, axis, axis_color in ((line[14], (row["head_dx"], row["head_dy"]), (0, 0, 255)),
                                      (line[40], (row["tail_dx"], row["tail_dy"]), (255, 150, 20))):
        start = tuple(np.rint(position - np.asarray(axis) * 30).astype(int))
        end = tuple(np.rint(position + np.asarray(axis) * 30).astype(int))
        cv2.arrowedLine(image, start, end, axis_color, 2, cv2.LINE_AA, tipLength=.15)
    cv2.putText(image, f"ID {identity}  {row['tail_angle_deg']:+.1f} deg", (cx - 50, cy - 25),
                cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2, cv2.LINE_AA)


TAIL_FIELDS = ["frame", "filename", "time_s", "id", "cx", "cy", "area", "valid", "quality_reason",
               "head_confidence", "body_length_px", "head_dx", "head_dy", "heading_deg",
               "tail_dx", "tail_dy", "tail_angle_deg", "smoothed_angle_deg", "frequency_segment"]
FREQUENCY_FIELDS = ["id", "segment", "start_frame", "end_frame", "start_time_s", "end_time_s", "valid_frames",
                    "frequency_hz", "spectral_peak_hz", "amplitude_deg", "cycles", "period_cv", "frequency_quality"]


def analyze_tail(records, meta):
    """纯计算入口：返回每条鱼每个连续段的频率/质量字典列表。

    records为geometry产生的可修改字典列表，meta至少含正数fps。
    本函数只补写每行smoothed_angle_deg和frequency_segment，不改原角度、
    几何、时间或valid。无效行的平滑角为NaN、段号为-1。
    返回值保留原CSV字段，并附frame_range（实际帧号列表）、peak_times_s和
    smoothed_angles，便于总入口定位证据帧。不存在可靠频率不等于0Hz：
    geometry无效标no_valid_geometry，单帧等不足周期标too_short等。
    空records返回[]；不要求image_paths或图片，也不导入SAM3/torch。
    """
    if not records:
        return []
    fps = float(meta["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("meta['fps']必须为有限正数。")
    by_id = defaultdict(list)
    for row in records:
        row.update(smoothed_angle_deg=float("nan"), frequency_segment=-1)
        by_id[row["id"]].append(row)
    frequency_rows = []
    for identity, fish in sorted(by_id.items()):
        segments = valid_segments(fish, fps)
        for number, segment in enumerate(segments):
            result = estimate_frequency(segment, fps)
            frequency_rows.append(dict(id=identity, segment=number,
                                       start_frame=segment[0]["frame"], end_frame=segment[-1]["frame"],
                                       start_time_s=segment[0]["time_s"], end_time_s=segment[-1]["time_s"],
                                       valid_frames=len(segment), frame_range=[r["frame"] for r in segment], **result))
            for index, row in enumerate(segment):
                row["frequency_segment"] = number
                if result["smoothed_angles"]:
                    row["smoothed_angle_deg"] = result["smoothed_angles"][index]
        if not segments:
            frequency_rows.append(dict(id=identity, segment=-1, start_frame=None, end_frame=None,
                                       start_time_s=None, end_time_s=None, valid_frames=0, frame_range=[],
                                       frequency_hz=float("nan"), spectral_peak_hz=float("nan"),
                                       amplitude_deg=float("nan"), cycles=0, period_cv=float("nan"),
                                       frequency_quality="no_valid_geometry", peak_times_s=[], smoothed_angles=[]))
    return frequency_rows


def plot_results(records, frequency_rows, output):
    """每ID导出一张原角度/轻平滑/同相位峰图；不连接无效段，不重新估频。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_id, results = defaultdict(list), defaultdict(list)
    for row in records:
        by_id[row["id"]].append(row)
    for result in frequency_rows:
        results[result["id"]].append(result)
    for identity, fish in sorted(by_id.items()):
        fig, ax = plt.subplots(figsize=(10, 4))
        times = [r["time_s"] for r in fish]
        raw = [r["tail_angle_deg"] if r["valid"] else np.nan for r in fish]
        ax.scatter(times, raw, color="0.65", s=5, label="Signed tail angle")
        by_frame = {r["frame"]: r for r in fish}
        for result in results[identity]:
            smooth = result.get("smoothed_angles", [])
            frames = result.get("frame_range", [])
            if smooth:
                ax.plot([by_frame[f]["time_s"] for f in frames], smooth, color="#0077b6", linewidth=1.2,
                        label="Light smoothing" if result["segment"] == 0 else None)
            for peak in result.get("peak_times_s", []):
                ax.axvline(peak, color="#e76f51", alpha=.3, linewidth=.7)
        ax.axhline(0, color="0.3", linewidth=.6)
        ax.set(xlabel="Time (s)", ylabel="Signed head-tail axis angle (deg)",
               title=f"Fish ID {identity} | gaps = invalid geometry; orange = same-phase peaks")
        ax.grid(alpha=.2); ax.legend(loc="best"); fig.tight_layout()
        fig.savefig(output / f"tail_angle_id_{identity}.png", dpi=160); plt.close(fig)


def render_outputs(records, meta, output, save_video):
    """仅有完整原图路径时画视频；缺图返回跳过原因，绝不生成缩短的伪完整视频。

    meta.frame_ids指定实际帧号和顺序，可为[1,2,4]；视频按该列表一图一帧，
    不填造缺失帧，时长为图数/fps。首帧预览不依赖save_video开关。
    返回字典包括video_saved、video_skip_reason、video_frames、preview_saved。
    """
    paths = {int(k): v for k, v in meta.get("image_paths", {}).items()}
    frame_ids = [int(f) for f in meta.get("frame_ids", sorted(paths or {r["frame"] for r in records}))]
    result = dict(video_saved=False, video_skip_reason="disabled" if not save_video else "",
                  video_frames=0, preview_saved=False, frame_ids=frame_ids)
    if not frame_ids:
        result["video_skip_reason"] = "no_frames"
        return result
    missing = [f for f in frame_ids if not paths.get(f) or not Path(paths[f]).is_file()]
    if missing:
        result.update(video_skip_reason="missing_source_images", missing_image_frames=missing)
        return result
    width, height = int(meta.get("width") or 0), int(meta.get("height") or 0)
    # 未知fps保留未知；空帧已在上面返回，缺元数据则跳过视频。
    fps = float(meta["fps"]) if meta.get("fps") is not None else math.nan
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
        result["video_skip_reason"] = "invalid_video_metadata"
        return result
    import cv2
    by_frame = defaultdict(list)
    for row in records:
        by_frame[row["frame"]].append(row)
    writer = None
    try:
        if save_video:
            writer = cv2.VideoWriter(str(output / "tail.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("无法打开MP4编码器。")
        selected = frame_ids if save_video else frame_ids[:1]
        for index, frame in enumerate(selected):
            image = read_image(paths[frame])
            if image.shape[:2] != (height, width):
                raise ValueError(f"原图尺寸不匹配：frame={frame}")
            for row in by_frame[frame]:
                draw_geometry(image, row)
            cv2.putText(image, f"Frame ID {frame} | red: HEAD | blue: TAIL | axes both tail->head",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
            if index == 0:
                save_image(output / "preview.jpg", image)
                result["preview_saved"] = True
            if writer is not None:
                writer.write(image)
    finally:
        if writer is not None:
            writer.release()
    if save_video:
        cap = cv2.VideoCapture(str(output / "tail.mp4"))
        count = 0
        try:
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                count += 1
        finally:
            cap.release()
        if count != len(frame_ids):
            raise RuntimeError(f"视频帧数验证失败：{count}/{len(frame_ids)}")
        result.update(video_saved=True, video_skip_reason="", video_frames=count)
    return result


def export_results(records, meta, output: Path, save_video=True, frequency_rows=None):
    """导出尾角三张CSV、每鱼曲线、可选视频、首帧预览与analysis_info.json。

    output为调用方指定的绝对路径，直接使用该目录，不追加时间戳。
    frequency_rows=None时调用analyze_tail，否则复用已经计算的结果，避免重算。
    只清理并替换摆尾生成文件，旧鱼ID曲线及视频不会残留。空输入仍保存表头和JSON，
    data_quality='no_records'；无有效几何和有效角但周期不足另有状态区分。
    不提供原图仅跳过可视化，不影响CSV/曲线/数值返回。
    """
    output = Path(output)
    if frequency_rows is None:
        frequency_rows = analyze_tail(records, meta)
    output = prepare_output_directory(output, [
        "analysis_info.json", "tail_frames.csv", "tail_frequency.csv", "centerlines.csv",
        "tail_angle_id_*.png", "preview.jpg", "tail.mp4"], invalidate_combined=True)
    valid_count = sum(bool(r.get("valid")) for r in records)
    quality_counts = dict(Counter(r["frequency_quality"] for r in frequency_rows))
    quality = ("no_records" if not records else "no_valid_geometry" if not valid_count
               else "ok" if quality_counts.get("ok", 0) else "no_qualified_frequency")
    info = {k: v for k, v in meta.items() if k not in ("image_paths", "frame_times")}
    info.update(status="running", total_objects=len(records), valid_objects=valid_count,
                head_hints=meta.get("head_hints"),
                use_filename_timestamps=meta.get("use_filename_timestamps"),
                requested_use_filename_timestamps=meta.get("requested_use_filename_timestamps"),
                save_video_requested=bool(save_video),
                invalid_objects=len(records) - valid_count, data_quality=quality,
                frequency_quality_counts=quality_counts, valid_frequency_segments=quality_counts.get("ok", 0),
                method="mask skeleton longest path; head width + temporal continuity; signed tail angle",
                frequency_range_hz=[MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ],
                min_complete_cycles=MIN_COMPLETE_CYCLES, min_amplitude_deg=MIN_AMPLITUDE_DEG,
                note="2D silhouette estimates; missing frequency is not zero; video uses listed actual frame IDs at nominal FPS.")
    def save_info():
        (output / "analysis_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    save_info()
    try:
        write_csv(output / "tail_frames.csv", records, TAIL_FIELDS)
        write_csv(output / "tail_frequency.csv", frequency_rows, FREQUENCY_FIELDS)
        centerline_rows = []
        for row in records:
            if row["valid"] and row.get("centerline") is not None:
                centerline_rows.extend(dict(frame=row["frame"], id=row["id"], point=i, x=float(p[0]), y=float(p[1]))
                                       for i, p in enumerate(row["centerline"]))
        write_csv(output / "centerlines.csv", centerline_rows, ["frame", "id", "point", "x", "y"])
        plot_results(records, frequency_rows, output)
        info.update(render_outputs(records, meta, output, save_video))
        info["status"] = "complete"
        save_info()
    except BaseException:
        info.update(status="failed", error=traceback.format_exc())
        save_info()
        raise
    return frequency_rows


def main():
    """独立运行入口：与main共用output/behavior/tail，覆盖此模块旧结果。"""
    from behavior.geometry import load_geometry
    records, meta = load_geometry(SEGMENTATION_DIR, head_hints=HEAD_HINTS,
                                  use_filename_timestamps=USE_FILENAME_TIMESTAMPS)
    output = OUTPUT_DIR
    export_results(records, meta, output, save_video=SAVE_VIDEO)
    print(f"摆尾分析完成：{output}", flush=True)
    return output


if __name__ == "__main__":
    main()


