"""top 项目：按固定帧步长分析质心运动，带符号距离=完整欧氏位移×前后向符号。"""
from pathlib import Path

# ---------------------- 用户参数（均使用绝对路径） ----------------------
SEGMENTATION_DIR = Path(r"E:\Phd\temperature\sam3\top\outputs\20261114_095054_184431")
OUTPUT_DIR = Path(r"E:\Phd\temperature\sam3\top\analysis\speed")
FRAME_STEPS = [1, 5]            # 每鱼首次观测为锚：step=5 时 0→5→10，不是滑动窗口
HEAD_HINTS = {}                 # 首次头向提示，如 {0: "left"}，由公共 geometry 解释
USE_FILENAME_TIMESTAMPS = True  # 优先采用原图文件名的时间；geometry 会把起点归零
MAX_AREA_CHANGE_RATIO = 1.5     # 窗口内每一对相邻帧均检查，不仅检查首末帧
MAX_HEAD_TURN_DEG = 90.0        # 窗口内每一对相邻头轴最大转角；不限制窗口累计总转角
SIGN_DOT_TOLERANCE = 1e-9       # 点积容差=此系数×max(1,欧氏距离)，容差内侧移取正
MAX_EXPLICIT_MISSING_FRAMES = 1000  # 缺帧多于此值只写压缩段，避免展开巨量帧号
SAVE_VIDEO = True               # 是否额外输出视频；不影响 CSV 和数值计算
TRAIL_FRAMES = 30               # 仅控制视频显示，计算始终采用未经平滑的掩膜质心
# --------------------------------------------------------------------

import csv
import json
import math
from collections import defaultdict, deque
from bisect import bisect_left, bisect_right
from datetime import datetime

import cv2
import numpy as np


# 空输入时也按固定字段导出表头。帧号列表和缺帧闭区间段在CSV中保存为JSON数组。
# missing_frame_count始终是总缺帧数；missing_frame_ranges是[[起帧,末帧],...]；
# 缺帧数<=MAX_EXPLICIT_MISSING_FRAMES时missing_frames也逐帧展开，否则为[]。
# 不可单凭missing_frames是否为空判连续性，应检查missing_frame_count。
# 一行表示一个固定采样窗口，主键为 (frame_step,id,frame)。start_frame 和 frame 是
# 窗口计划起点/本行终点；终点缺失仍保留无效占位行。首帧 start_frame==frame；
# 尾部不足窗口的 frame 为最后实际观测帧，而 expected_end_frame 仍为计划终点。
# cx/cy/area/heading_deg/head_confidence 复制当前端点几何；端点缺失时为空。
# distance_valid 控制 dx/dy/step_distance/speed；direction_valid 控制正负量和两个
# 投影诊断。dt_s 可能在无效窗口仍有值，它本身不能代替有效标记。
# cumulative_* 是各自有效窗口累计，当前行无效时仍保留以前有效段的累计值；
# 无任何有效区间时算术和为0，必须结合 summary_valid，不能解释为有效的静止。
FRAME_FIELDS = [
    "frame_step", "anchor_frame", "start_frame", "expected_end_frame", "frame", "frame_range", "missing_frames",
    "missing_frame_count", "missing_frame_ranges",
    "window_complete", "start_time_s", "time_s", "id", "filename", "cx", "cy", "area",
    "heading_deg", "head_confidence", "dt_s", "dx_px", "dy_image_px",
    "step_distance_px", "speed_px_s", "signed_distance_px", "signed_speed_px_s",
    "forward_projection_px", "lateral_distance_px", "direction_sign", "sign_tolerance_px",
    "distance_valid", "direction_valid", "quality_reason", "quality_detail",
    "cumulative_path_distance_px", "cumulative_signed_distance_px",
]
# 汇总主键为 (frame_step,id)。observations 为实际观测帧数，sample_rows 含首行和
# 尾部不足行；candidate_intervals=sample_rows-1，含已输出的无效/不足窗口，
# 不包含完全没有观测的中间窗口（这些窗口不会大量展开）。track_span_s
# 是鱼首次至末次观测时间差。两种 valid_time 分别只加相应有效窗口的 dt_s；
# coverage=有效时间/track_span_s。path_distance 是无符号欧氏端点距离累计；
# signed_distance 是有符号累计；forward/backward 分别累加正值/负值绝对值。
# 注意新规则把纯侧移记为正，因此 forward_distance 也包含纯侧移，不能称纯纵向位移。
# 两种 mean 分别为相应总距离/相应有效时间；有效时间为0时均值留空。
SUMMARY_FIELDS = [
    "frame_step", "id", "anchor_frame", "start_frame", "end_frame", "frame_range", "observations",
    "sample_rows", "candidate_intervals", "incomplete_intervals", "track_span_s",
    "valid_distance_intervals", "valid_signed_intervals", "valid_distance_time_s", "valid_signed_time_s",
    "distance_time_coverage", "signed_time_coverage", "path_distance_px", "signed_distance_px",
    "forward_distance_px", "backward_distance_px", "mean_speed_px_s", "mean_signed_speed_px_s",
    "distance_summary_valid", "direction_summary_valid",
]
CURVES = [
    ("speed_px_s", "unsigned_speed.png", "Unsigned speed (px/s)", "distance_valid"),
    ("signed_speed_px_s", "signed_speed.png", "Signed speed (px/s)", "direction_valid"),
    ("step_distance_px", "unsigned_distance.png", "Endpoint distance (px)", "distance_valid"),
    ("signed_distance_px", "signed_distance.png", "Signed endpoint distance (px)", "direction_valid"),
    ("cumulative_path_distance_px", "cumulative_unsigned_distance.png", "Cumulative valid distance (px)", "distance_valid"),
    ("cumulative_signed_distance_px", "cumulative_signed_distance.png", "Cumulative signed distance (px)", "direction_valid"),
]


def _number(value, default=math.nan):
    """把内部记录中的数值转为有限浮点数；缺失/非法值保留为 NaN。"""
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _valid(value):
    """允许原生 bool、0/1 或 CSV 风格真假文本，避免 bool('False') 的陷阱。"""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1"}
    return _number(value, default=0) == 1


def _steps(frame_steps):
    """验证步长正整数列表，去重并保持调用者指定顺序。"""
    try:
        values = list(FRAME_STEPS if frame_steps is None else frame_steps)
    except TypeError as exc:
        raise ValueError("FRAME_STEPS 必须是非空的正整数列表，例如 [1, 5]。") from exc
    if not values or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1 for v in values):
        raise ValueError("FRAME_STEPS 必须是非空的正整数列表，例如 [1, 5]。")
    return list(dict.fromkeys(int(v) for v in values))


def _record_row(lookup, observed_frames, identity, anchor, start, end, step, expected_end):
    """建立一个采样行：frame_range 只列实际存在的观测，绝不补造缺失实例。"""
    endpoint = lookup.get(end, {})
    startpoint = lookup.get(start, {})
    frame_range = observed_frames[bisect_left(observed_frames, start):bisect_right(observed_frames, end)]
    missing_ranges, cursor = [], start
    for frame in frame_range:
        if frame > cursor:
            missing_ranges.append([cursor, frame - 1])
        cursor = frame + 1
    if cursor <= end:
        missing_ranges.append([cursor, end])
    missing_count = end - start + 1 - len(frame_range)
    missing = ([frame for first, last in missing_ranges for frame in range(first, last + 1)]
               if missing_count <= MAX_EXPLICIT_MISSING_FRAMES else [])
    row = {k: math.nan for k in FRAME_FIELDS}
    row.update(frame_step=step, anchor_frame=anchor, start_frame=start, expected_end_frame=expected_end,
               frame=end, frame_range=frame_range, missing_frames=missing,
               missing_frame_count=missing_count, missing_frame_ranges=missing_ranges,
               window_complete=int(end - start == step and missing_count == 0), id=identity,
               start_time_s=_number(startpoint.get("time_s")), time_s=_number(endpoint.get("time_s")),
               filename=endpoint.get("filename", ""), distance_valid=0, direction_valid=0,
               quality_reason="", quality_detail="")
    for key in ("cx", "cy", "area", "heading_deg", "head_confidence"):
        row[key] = _number(endpoint.get(key))
    row["dt_s"] = row["time_s"] - row["start_time_s"]
    return row


def _measure_window(row, lookup):
    """检查窗口所有邻帧，计算首末质心位移；失败保留原因，不跨缺口强算。"""
    start, end = row["start_frame"], row["frame"]
    if end == start:
        row.update(quality_reason="first_observation", dt_s=math.nan)
        return
    if end - start < row["frame_step"]:
        row.update(quality_reason="incomplete_window", quality_detail=f"需要 {row['frame_step']} 个帧间隔，末尾只有 {end-start} 个")
        return
    if row["missing_frame_count"]:
        row.update(quality_reason="frame_gap", quality_detail=f"缺失{row['missing_frame_count']}帧，闭区间段 " + json.dumps(row["missing_frame_ranges"]))
        return
    window = [lookup[f] for f in range(start, end + 1)]
    for i, item in enumerate(window):
        frame = start + i
        if not np.isfinite([_number(item.get("cx")), _number(item.get("cy"))]).all():
            row.update(quality_reason="invalid_centroid", quality_detail=f"frame={frame}")
            return
        if not math.isfinite(_number(item.get("time_s"))):
            row.update(quality_reason="invalid_time_interval", quality_detail=f"frame={frame} 时间非有限数")
            return
        if not math.isfinite(_number(item.get("area"))) or float(item["area"]) <= 0:
            row.update(quality_reason="invalid_mask_area", quality_detail=f"frame={frame}")
            return
        if i:
            previous = window[i - 1]
            if float(item["time_s"]) <= float(previous["time_s"]):
                row.update(quality_reason="invalid_time_interval", quality_detail=f"frame={frame-1}→{frame}")
                return
            ratio = max(float(item["area"]), float(previous["area"])) / min(float(item["area"]), float(previous["area"]))
            if ratio > MAX_AREA_CHANGE_RATIO:
                row.update(quality_reason="mask_area_jump", quality_detail=f"frame={frame-1}→{frame}, ratio={ratio:.6g}")
                return
    delta = np.array([float(window[-1]["cx"]) - float(window[0]["cx"]),
                      float(window[-1]["cy"]) - float(window[0]["cy"])])
    distance = float(np.linalg.norm(delta))
    dt = float(window[-1]["time_s"]) - float(window[0]["time_s"])
    row.update(dt_s=dt, dx_px=float(delta[0]), dy_image_px=float(delta[1]),
               step_distance_px=distance, speed_px_s=distance / dt, distance_valid=1)
    axes = []
    for i, item in enumerate(window):
        frame = start + i
        if not _valid(item.get("valid")):
            row.update(quality_reason="head_axis_invalid", quality_detail=f"frame={frame}: {item.get('quality_reason', '')}")
            return
        axis = np.array([_number(item.get("head_dx")), _number(item.get("head_dy"))])
        if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
            row.update(quality_reason="invalid_head_vector", quality_detail=f"frame={frame}")
            return
        axis /= np.linalg.norm(axis)
        if axes:
            turn = math.degrees(math.acos(float(np.clip(np.dot(axes[-1], axis), -1, 1))))
            if turn > MAX_HEAD_TURN_DEG + 1e-9:
                row.update(quality_reason="head_axis_jump", quality_detail=f"frame={frame-1}→{frame}, turn={turn:.6g} deg")
                return
        axes.append(axis)
    # 仅首末两端构成前向均值。五帧总转角可以超过90°，只要每个邻帧转角合格。
    forward = axes[0] + axes[-1]
    if np.linalg.norm(forward) < 1e-8:
        row.update(quality_reason="endpoint_head_axis_cancellation", quality_detail="首末头轴反向，平均前向无定义")
        return
    forward /= np.linalg.norm(forward)
    projection = float(np.dot(delta, forward))
    # 原图 y 向下；此垂向轴在鱼朝右时指向图像上方，因此向上侧移为正。
    lateral = float(np.dot(delta, [forward[1], -forward[0]]))
    tolerance = SIGN_DOT_TOLERANCE * max(1.0, distance)
    sign = 1 if projection >= -tolerance else -1
    # 新定义：保留完整欧氏距离的大小；投影只判符号，纯侧移按正向。
    signed = sign * distance
    row.update(signed_distance_px=signed, signed_speed_px_s=signed / dt,
               forward_projection_px=projection, lateral_distance_px=lateral,
               direction_sign=sign, sign_tolerance_px=tolerance, direction_valid=1)


def calculate_speed(records, frame_steps=None):
    """返回各步长合并的区间记录，主键为 (frame_step, id, frame)。

    records 是 geometry.load_geometry 的字典列表：frame/id 为整数，time_s
    单位秒，cx/cy 是图像坐标（像素，y向下），area 为像素面积；valid 表示
    中轴及头尾朝向可靠，head_dx/head_dy 为指向头部的二维向量。heading_deg
    和 head_confidence 只复制用于追溯，不参与速度计算。输入列表不会被修改。
    每个 ID 首次观测帧为固定锚。step=5 时产生首行和 A→A+5→A+10，
    漏帧不重定锚；末尾不足完整步长也输出 incomplete_window 行。每个
    frame_range 只列该窗口实际存在的观测帧。完全没有任何观测（含边界）的
    窗口不逐一展开；含观测的窗口保持原固定锚，绝不跨越空白重新连段。
    missing_frame_count记录缺帧数，missing_frame_ranges以闭区间压缩缺帧，
    小缺口另以missing_frames逐帧展开；大缺口该列表为空而count仍大于0。
    位移大小是窗口首末质心的欧氏距离，不是窗口内小步距离之和。
    所有 *_px 为像素，*_px_s 为像素/秒；direction_sign 为+1或-1，
    tolerance 为点积判号容差。无效测量为 NaN，CSV 导出时留空；有效标记
    使用整数0/1。quality_reason 是稳定原因代码，quality_detail 解释具体帧。
    """
    steps = _steps(frame_steps)
    if not math.isfinite(MAX_AREA_CHANGE_RATIO) or MAX_AREA_CHANGE_RATIO < 1:
        raise ValueError("MAX_AREA_CHANGE_RATIO 必须是 >=1 的有限数。")
    if not math.isfinite(MAX_HEAD_TURN_DEG) or not 0 < MAX_HEAD_TURN_DEG <= 180:
        raise ValueError("MAX_HEAD_TURN_DEG 必须在 (0,180]。")
    if not math.isfinite(SIGN_DOT_TOLERANCE) or SIGN_DOT_TOLERANCE < 0:
        raise ValueError("SIGN_DOT_TOLERANCE 必须是非负有限数。")
    if not isinstance(MAX_EXPLICIT_MISSING_FRAMES, int) or isinstance(MAX_EXPLICIT_MISSING_FRAMES, bool) or MAX_EXPLICIT_MISSING_FRAMES < 0:
        raise ValueError("MAX_EXPLICIT_MISSING_FRAMES 必须是非负整数。")
    groups = defaultdict(dict)
    for source in records:
        frame, identity = int(source["frame"]), int(source["id"])
        if frame in groups[identity]:
            raise ValueError(f"同一 ID 存在重复帧：id={identity}, frame={frame}")
        groups[identity][frame] = source
    output = []
    for step in steps:
        for identity, lookup in sorted(groups.items()):
            observed_frames = sorted(lookup)
            anchor, last = observed_frames[0], observed_frames[-1]
            # 只从实际观测定位固定格点窗口，复杂度不随两个帧号之间的巨大空白增长。
            # 格点观测属于左窗口终点和右窗口起点，两个窗口均保留做连续性检查。
            starts = set()
            for frame in observed_frames:
                start = anchor + ((frame - anchor) // step) * step
                if frame == start and frame > anchor:
                    starts.add(start - step)
                if start < last:
                    starts.add(start)
            windows = [(anchor, anchor)] + [(start, min(start + step, last)) for start in sorted(starts)]
            path_total, signed_total = 0.0, 0.0
            for start, end in windows:
                row = _record_row(lookup, observed_frames, identity, anchor, start, end, step, start + step)
                _measure_window(row, lookup)
                if row["distance_valid"]:
                    path_total += row["step_distance_px"]
                if row["direction_valid"]:
                    signed_total += row["signed_distance_px"]
                row.update(cumulative_path_distance_px=path_total, cumulative_signed_distance_px=signed_total)
                output.append(row)
    return sorted(output, key=lambda r: (r["frame_step"], r["frame"], r["id"]))


def summarize(rows):
    """按 (frame_step,id) 汇总，均值=对应有效区间总距离/有效 dt 之和。

    frame_range 为该鱼实际观测帧的并集；coverage 的分母为首末观测时间
    跨度，缺失/不足窗口不会当作零速度。没有有效区间时平均值为 NaN，
    累计算术和为0，同时提供两个 summary_valid，供总入口明确标无效。
    """
    groups = defaultdict(list)
    for row in rows:
        groups[(row["frame_step"], row["id"])].append(row)
    output = []
    for (step, identity), group in sorted(groups.items()):
        group = sorted(group, key=lambda r: r["frame"])
        distance_rows = [r for r in group if r["distance_valid"]]
        signed_rows = [r for r in group if r["direction_valid"]]
        t_distance = sum(r["dt_s"] for r in distance_rows)
        t_signed = sum(r["dt_s"] for r in signed_rows)
        path = sum(r["step_distance_px"] for r in distance_rows)
        signed = sum(r["signed_distance_px"] for r in signed_rows)
        span = _number(group[-1]["time_s"]) - _number(group[0]["time_s"])
        observed = sorted({f for r in group for f in r["frame_range"]})
        output.append(dict(frame_step=step, id=identity, anchor_frame=group[0]["anchor_frame"],
                           start_frame=group[0]["frame"], end_frame=group[-1]["frame"], frame_range=observed,
                           observations=len(observed), sample_rows=len(group), candidate_intervals=max(0, len(group)-1),
                           incomplete_intervals=sum(r["quality_reason"] == "incomplete_window" for r in group),
                           track_span_s=span, valid_distance_intervals=len(distance_rows), valid_signed_intervals=len(signed_rows),
                           valid_distance_time_s=t_distance, valid_signed_time_s=t_signed,
                           distance_time_coverage=t_distance/span if math.isfinite(span) and span > 0 else math.nan,
                           signed_time_coverage=t_signed/span if math.isfinite(span) and span > 0 else math.nan,
                           path_distance_px=path, signed_distance_px=signed,
                           forward_distance_px=sum(max(0, r["signed_distance_px"]) for r in signed_rows),
                           backward_distance_px=sum(max(0, -r["signed_distance_px"]) for r in signed_rows),
                           mean_speed_px_s=path/t_distance if t_distance else math.nan,
                           mean_signed_speed_px_s=signed/t_signed if t_signed else math.nan,
                           distance_summary_valid=int(bool(distance_rows)), direction_summary_valid=int(bool(signed_rows))))
    return output


def save_csv(path, rows, fields):
    """输出 UTF-8 BOM 表格；列表写 JSON，非有限数留空，空输入仍保存表头。"""
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in fields:
                value = row.get(key)
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                elif value is None or isinstance(value, (float, np.floating)) and not math.isfinite(value):
                    value = ""
                clean[key] = value
            writer.writerow(clean)


def _plt():
    """延迟导入绘图模块，纯计算 API 不依赖图形界面。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _curve_panel(ax, rows, field, label, valid_key):
    """所有曲线以实际time_s为横轴；缺失时间保留NaN断点，不补造采样时刻。"""
    for identity in sorted({r["id"] for r in rows}):
        group = [r for r in rows if r["id"] == identity]
        cumulative = field.startswith("cumulative_")
        values = [r[field] if r[valid_key] or cumulative and r["quality_reason"] == "first_observation"
                  else math.nan for r in group]
        # 所有区间都无效时，不把“累计求和0”画成鱼静止的有效曲线。
        if not any(r[valid_key] for r in group):
            continue
        ax.plot([_number(r.get("time_s")) for r in group], values, ".-", label=f"ID {identity}", linewidth=1, markersize=2)
    if not ax.lines:
        ax.text(.5, .5, "No valid intervals", transform=ax.transAxes, ha="center", va="center")
    else:
        ax.legend(fontsize=8, ncol=3)
    if "signed" in field and not field.startswith("cumulative_path"):
        ax.axhline(0, color="black", linewidth=.6)
    ax.set(xlabel="Time (s)", ylabel=label)
    ax.grid(alpha=.2)


def plot_curves(rows, output, step):
    """每步长同时导出六张独立曲线和一张六面板总图，signed 为整段欧氏量。"""
    plt = _plt()
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), constrained_layout=True)
    for ax, (field, filename, label, valid_key) in zip(axes.ravel(), CURVES):
        _curve_panel(ax, rows, field, label, valid_key)
        single, single_ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
        _curve_panel(single_ax, rows, field, label, valid_key)
        single_ax.set_title(f"Step {step}: {label} | fixed non-overlapping windows")
        single.savefig(output / filename, dpi=150)
        plt.close(single)
    fig.suptitle(f"Step {step} | signed distance = Euclidean distance x direction sign; gaps excluded")
    fig.savefig(output / "speed_curves.png", dpi=150)
    plt.close(fig)


def plot_summary(summaries, output, steps, comparison=False):
    """绘制各鱼的两种平均速度与两种总距离；对照图并排显示所有帧步长。"""
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    identities = sorted({r["id"] for r in summaries})
    metrics = [("mean_speed_px_s", "Mean unsigned speed (px/s)", "distance_summary_valid"),
               ("mean_signed_speed_px_s", "Mean signed speed (px/s)", "direction_summary_valid"),
               ("path_distance_px", "Total valid endpoint distance (px)", "distance_summary_valid"),
               ("signed_distance_px", "Total signed endpoint distance (px)", "direction_summary_valid")]
    lookup = {(r["frame_step"], r["id"]): r for r in summaries}
    for ax, (field, title, flag) in zip(axes.ravel(), metrics):
        x = np.arange(len(identities))
        width = .8 / max(1, len(steps))
        for i, step in enumerate(steps):
            values = [lookup[(step, identity)][field] if (step, identity) in lookup and lookup[(step, identity)][flag]
                      else math.nan for identity in identities]
            if identities:
                ax.bar(x + (i - (len(steps)-1)/2) * width, values, width, label=f"step {step}")
        ax.set(title=title, xlabel="SAM3 ID")
        ax.set_xticks(x, [str(i) for i in identities])
        ax.axhline(0, color="black", linewidth=.7)
        ax.grid(axis="y", alpha=.2)
        if identities:
            ax.legend(fontsize=9)
        if not identities or not any(r[flag] for r in summaries):
            ax.text(.5, .5, "No valid intervals", transform=ax.transAxes, ha="center")
    fig.suptitle("Frame-step comparison: each step uses its own valid intervals" if comparison else f"Step {steps[0]}: valid-interval summaries")
    fig.savefig(output / ("frame_steps_comparison.png" if comparison else "speed_summary.png"), dpi=160)
    plt.close(fig)


def draw_video(rows, records, meta, output, step):
    """逐原始帧画质心与头轴，仅在当前采样终点显示该窗口数值，不保持旧值。

    原图路径全无或部分缺失时返回明确的skipped原因，不中止CSV/图导出，
    也不生成缺少若干原帧的缩短视频。frame_ids省略时使用已知原图和观测帧
    的并集，避免仅有部分image_paths时误当作完整视频。成功时返回saved。
    """
    paths = {int(k): v for k, v in (meta.get("image_paths") or {}).items()}
    if not paths:
        return "skipped_no_image_paths"
    fallback_frames = sorted(set(paths) | {int(record["frame"]) for record in records})
    frame_ids = [int(f) for f in meta.get("frame_ids", fallback_frames)]
    if not frame_ids:
        return "skipped_no_frames"
    missing_images = [frame for frame in frame_ids if not paths.get(frame) or not Path(paths[frame]).is_file()]
    if missing_images:
        return (f"skipped_incomplete_image_paths: {len(missing_images)} missing frames; "
                f"first frame IDs={missing_images[:10]}")
    width, height = int(meta["width"]), int(meta["height"])
    fps = float(meta["fps"])
    if not math.isfinite(fps) or fps <= 0 or any(b <= a for a, b in zip(frame_ids, frame_ids[1:])):
        raise ValueError("视频 fps 或 frame_ids 无效。")
    if not isinstance(TRAIL_FRAMES, int) or isinstance(TRAIL_FRAMES, bool) or TRAIL_FRAMES < 1:
        raise ValueError("TRAIL_FRAMES 必须是正整数。")
    by_frame = defaultdict(list)
    for record in records:
        by_frame[int(record["frame"])].append(record)
    samples = {(r["frame"], r["id"]): r for r in rows}
    trails, previous_frame = defaultdict(lambda: deque(maxlen=TRAIL_FRAMES)), {}
    writer = cv2.VideoWriter(str(output / "speed.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("无法创建 speed.mp4。")
    try:
        for frame in frame_ids:
            if frame not in paths:
                raise ValueError(f"frame={frame} 缺少原图路径。")
            image = cv2.imdecode(np.fromfile(str(paths[frame]), np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise ValueError(f"frame={frame} 原图缺失或尺寸不一致。")
            for record in by_frame[frame]:
                identity = int(record["id"])
                if not np.isfinite([_number(record.get("cx")), _number(record.get("cy"))]).all():
                    continue
                center = (int(round(record["cx"])), int(round(record["cy"])))
                sample = samples.get((frame, identity))
                trail = trails[identity]
                if previous_frame.get(identity) != frame - 1 or sample is not None and not sample["distance_valid"]:
                    trail.clear()
                trail.append(center)
                previous_frame[identity] = frame
                if len(trail) > 1:
                    cv2.polylines(image, [np.asarray(trail, np.int32)], False, (50, 220, 255), 2, cv2.LINE_AA)
                cv2.circle(image, center, 4, (0, 255, 255), -1)
                axis = np.array([_number(record.get("head_dx")), _number(record.get("head_dy"))])
                if _valid(record.get("valid")) and np.isfinite(axis).all() and np.linalg.norm(axis) > 1e-8:
                    axis /= np.linalg.norm(axis)
                    tip = tuple(np.rint(np.asarray(center) + 45 * axis).astype(int))
                    cv2.arrowedLine(image, center, tip, (255, 200, 30), 2, cv2.LINE_AA, tipLength=.2)
                if sample is None:
                    label = f"ID {identity} | not a step-{step} sample"
                elif sample["direction_valid"]:
                    label = f"ID {identity} [{sample['start_frame']}->{frame}] v={sample['signed_speed_px_s']:+.1f} d={sample['signed_distance_px']:+.2f}"
                else:
                    label = f"ID {identity} [{sample['start_frame']}->{frame}] N/A: {sample['quality_reason']}"
                x = max(2, min(center[0] - 80, width - 550))
                y = max(75, center[1] - 20)
                cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .44, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .44, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, f"SPEED step {step} | original frame {frame} | v:px/s d:px", (12, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "Numbers shown only at the current sample endpoint; lateral motion uses + sign", (12, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(image)
    finally:
        writer.release()
    capture, decoded = cv2.VideoCapture(str(output / "speed.mp4")), 0
    try:
        while capture.read()[0]:
            decoded += 1
    finally:
        capture.release()
    if decoded != len(frame_ids):
        raise RuntimeError(f"速度视频只有 {decoded}/{len(frame_ids)} 帧。")
    return "saved"


def export_results(records, meta, output: Path, frame_steps=None, save_video=True):
    """导出每步长 CSV、六类曲线、均值/总距图、对照图及可选视频。

    返回 (rows,summaries)，数值定义与 calculate_speed 一致。空输入或只有
    一帧时仍导出表头/无有效区间图；无 image_paths 或部分原图缺失时跳过
    视频，在 run_info.json 中明确记录原因，数值导出仍完成。
    调用者应提供新的输出目录避免覆盖。
    """
    steps = _steps(frame_steps)
    records = list(records)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    occupied = [p for p in [output / "run_info.json", output / "frame_steps_comparison.png"]
                + [output / f"step_{s}" for s in steps] if p.exists()]
    if occupied:
        raise FileExistsError(f"速度结果目录已包含本模块输出，请使用新目录：{occupied[0]}")
    rows = calculate_speed(records, steps)
    summaries = summarize(rows)
    info = dict(status="running", segmentation_dir=meta.get("segmentation_dir"),
                time_source=meta.get("time_source", "provided record time_s"), unit="px and px/s",
                frame_steps=steps, head_hints=meta.get("head_hints", HEAD_HINTS), centroid_smoothing=False,
                use_filename_timestamps=meta.get("use_filename_timestamps", USE_FILENAME_TIMESTAMPS), save_video_requested=bool(save_video),
                trail_frames=TRAIL_FRAMES, fps=meta.get("fps"),
                frame_ids=[int(f) for f in meta.get("frame_ids", sorted({int(r["frame"]) for r in records} |
                          {int(f) for f in (meta.get("image_paths") or {})}))],
                max_area_change_ratio=MAX_AREA_CHANGE_RATIO, max_adjacent_head_turn_deg=MAX_HEAD_TURN_DEG,
                sign_dot_tolerance=SIGN_DOT_TOLERANCE,
                max_explicit_missing_frames=MAX_EXPLICIT_MISSING_FRAMES,
                signed_distance_definition="Euclidean endpoint distance * sign(dot(displacement, normalized sum of normalized endpoint head axes)); dot>=-tolerance is positive",
                forward_projection_definition="Diagnostic only; NOT signed_distance_px",
                lateral_projection_definition="dot(displacement,[forward_y,-forward_x]); positive points up when head points right in image coordinates",
                forward_backward_totals_definition="Sums of positive signed distance and absolute negative signed distance; pure lateral intervals belong to the positive total",
                sampling="Fixed anchor at each ID first observed frame; non-overlapping; missing frames never reset anchor; incomplete tail excluded; entirely unobserved windows omitted without changing the anchor",
                frame_range_definition="Actually observed frame IDs in this window; missing_frame_count and inclusive missing_frame_ranges are always provided; missing_frames is expanded only within max_explicit_missing_frames",
                cumulative_definition="Sum of valid endpoint distances separately for each frame_step and ID; not the sum of inner one-frame path lengths",
                videos={}, observed_records=len(records), sample_rows=len(rows), summary_rows=len(summaries))
    info_path = output / "run_info.json"
    def save_info():
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    save_info()
    try:
        save_csv(output / "speed_frames_all_steps.csv", rows, FRAME_FIELDS)
        save_csv(output / "speed_summary_all_steps.csv", summaries, SUMMARY_FIELDS)
        for step in steps:
            folder = output / f"step_{step}"
            folder.mkdir()
            selected = [r for r in rows if r["frame_step"] == step]
            selected_summary = [r for r in summaries if r["frame_step"] == step]
            save_csv(folder / "speed_frames.csv", selected, FRAME_FIELDS)
            save_csv(folder / "speed_summary.csv", selected_summary, SUMMARY_FIELDS)
            plot_curves(selected, folder, step)
            plot_summary(selected_summary, folder, [step])
            info["videos"][f"step_{step}"] = draw_video(selected, records, meta, folder, step) if save_video else "disabled"
        plot_summary(summaries, output, steps, comparison=True)
        info["status"] = "complete"
    except BaseException as exc:
        info.update(status="failed", error=str(exc))
        raise
    finally:
        save_info()
    return rows, summaries


def main():
    """独立入口：从新 top 项目分割结果读取几何，再生成独立的速度分析批次。"""
    if not SEGMENTATION_DIR.is_absolute() or not OUTPUT_DIR.is_absolute():
        raise ValueError("SEGMENTATION_DIR 和 OUTPUT_DIR 必须使用绝对路径。")
    from geometry import load_geometry
    records, meta = load_geometry(SEGMENTATION_DIR, head_hints=HEAD_HINTS,
                                  use_filename_timestamps=USE_FILENAME_TIMESTAMPS)
    output = OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    rows, summaries = export_results(records, meta, output, FRAME_STEPS, SAVE_VIDEO)
    print(f"距离/速度分析完成：{output}\n步长={FRAME_STEPS}，采样记录={len(rows)}，汇总={len(summaries)}")
    return output


if __name__ == "__main__":
    main()
