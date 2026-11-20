"""top 项目的统一速度分析，按一个帧步长同时输出像素与 BL 结果。

先计算质心的像素运动，再除以该鱼整批有效中轴弧长的中位数 L。
距离 BL = 距离 px / L，速度 BL/s = 速度 px/s / L。每个 ID 使用自己的
固定标尺，整段输入共用，L 与所选速度步长无关。L 是分割中轴估计体长。
无有效体长时保留像素结果，BL 数值留空并标记原因。

两张 CSV 的每一行同时含像素和 BL 数值，共用帧号、时间、ID 和图像坐标。
图像定位始终用像素，运动量各自按 px 或 BL 标注并输出两段视频。
"""
from pathlib import Path
# 支持绝对路径直接运行本文件；正常包导入时无需修改模块搜索路径。
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------- 用户参数（均使用绝对路径） ----------------------
SEGMENTATION_DIR = Path(r"E:\Phd\temperature\sam3\top\output\tracking")
OUTPUT_DIR = Path(r"E:\Phd\temperature\sam3\top\output\behavior\speed")
FRAME_STEP = 1                  # 唯一默认步长。main与本脚本共用；5表示A→A+5→A+10采样。
HEAD_HINTS = {}                 # 首次头向提示，如 {0: "left"}，由公共 geometry 解释
USE_FILENAME_TIMESTAMPS = True  # 优先采用原图文件名的时间；geometry 会把起点归零
MAX_AREA_CHANGE_RATIO = 1.5     # 窗口内每一对相邻帧均检查，不仅检查首末帧
MAX_HEAD_TURN_DEG = 90.0        # 窗口内每一对相邻头轴最大转角；不限制窗口累计总转角
SIGN_DOT_TOLERANCE = 1e-9       # 点积容差=此系数×max(1,欧氏距离)，容差内侧移取正
MAX_EXPLICIT_MISSING_FRAMES = 1000  # 缺帧多于此值只写压缩段，避免展开巨量帧号
SAVE_VIDEO = True               # True输出px和BL两段视频；False仍输出合并CSV和四张图
TRAIL_FRAMES = 30               # 仅控制视频显示，计算始终采用未经平滑的掩膜质心
# --------------------------------------------------------------------

import csv
import json
from data_format import prepare_output_directory
import math
from collections import defaultdict, deque
from bisect import bisect_left, bisect_right
from statistics import median

import cv2
import numpy as np


def prepare_speed_output(output):
    """两单位直接写入同一速度目录，清理本模块已知旧文件与被关闭的视频。"""
    output = Path(output)
    if not output.is_absolute():
        raise ValueError("速度输出目录必须使用绝对路径。")
    # 兼容迁移旧布局，并清理本轮生成的固定文件名。只匹配已知输出。
    common = ["speed_frames.csv", "speed_summary.csv", "speed_curves.png", "speed_summary.png", "speed.mp4"]
    files = [*common, "run_info.json", "speed_frames_all_steps.csv", "speed_summary_all_steps.csv",
             "frame_steps_comparison.png"]
    for suffix in ("px", "bl"):
        files.extend([f"speed_curves_{suffix}.png", f"speed_summary_{suffix}.png", f"speed_{suffix}.mp4"])
    old_singles = ["unsigned_speed.png", "signed_speed.png", "unsigned_distance.png",
                   "signed_distance.png", "cumulative_unsigned_distance.png", "cumulative_signed_distance.png"]
    for directory in output.glob("step_*"):
        # 只认形如step_5的旧目录，避免匹配step_5_notes等自建文件夹。
        number = directory.name[5:]
        if number.isascii() and number.isdecimal():
            files.extend(f"{directory.name}/{name}" for name in common + old_singles)
    return prepare_output_directory(output, files, invalidate_combined=True)


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
PIXEL_FRAME_FIELDS = [
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
PIXEL_SUMMARY_FIELDS = [
    "frame_step", "id", "anchor_frame", "start_frame", "end_frame", "frame_range", "observations",
    "sample_rows", "candidate_intervals", "incomplete_intervals", "track_span_s",
    "valid_distance_intervals", "valid_signed_intervals", "valid_distance_time_s", "valid_signed_time_s",
    "distance_time_coverage", "signed_time_coverage", "path_distance_px", "signed_distance_px",
    "forward_distance_px", "backward_distance_px", "mean_speed_px_s", "mean_signed_speed_px_s",
    "distance_summary_valid", "direction_summary_valid",
]
CURVES = [
    ("speed_px_s", "Unsigned speed (px/s)", "distance_valid"),
    ("signed_speed_px_s", "Signed speed (px/s)", "direction_valid"),
    ("step_distance_px", "Endpoint distance (px)", "distance_valid"),
    ("signed_distance_px", "Signed endpoint distance (px)", "direction_valid"),
    ("cumulative_path_distance_px", "Cumulative valid distance (px)", "distance_valid"),
    ("cumulative_signed_distance_px", "Cumulative signed distance (px)", "direction_valid"),
]
SUMMARY_METRICS = [
    ("mean_speed_px_s", "Mean unsigned speed (px/s)", "distance_summary_valid"),
    ("mean_signed_speed_px_s", "Mean signed speed (px/s)", "direction_summary_valid"),
    ("path_distance_px", "Total valid endpoint distance (px)", "distance_summary_valid"),
    ("signed_distance_px", "Total signed endpoint distance (px)", "direction_summary_valid"),
]

# 对应关系只列需要按体长换算的运动量。cx/cy、area、时间、角度和帧号共用
# 原列，避免误把图像位置或面积也除以体长。投影与容差仅作换算展示，符号
# 始终由像素运动确定。累计和均值也采用对应有效区间，不能跨缺帧补值。
METRIC_NAMES = {
    "dx_px": "dx_bl", "dy_image_px": "dy_image_bl",
    "step_distance_px": "step_distance_bl", "speed_px_s": "speed_bl_s",
    "signed_distance_px": "signed_distance_bl", "signed_speed_px_s": "signed_speed_bl_s",
    "forward_projection_px": "forward_projection_bl", "lateral_distance_px": "lateral_distance_bl",
    "sign_tolerance_px": "sign_tolerance_bl",
    "cumulative_path_distance_px": "cumulative_path_distance_bl",
    "cumulative_signed_distance_px": "cumulative_signed_distance_bl",
    "path_distance_px": "path_distance_bl", "forward_distance_px": "forward_distance_bl",
    "backward_distance_px": "backward_distance_bl",
    "mean_speed_px_s": "mean_speed_bl_s", "mean_signed_speed_px_s": "mean_signed_speed_bl_s",
}
# 四个体长列分别记录本鱼的固定像素体长、有效样本数、是否可用和原因。
# 仅 valid 为真且 body_length_px 有限并大于0的帧参与体长中位数计算。
# BL有效性=原运动有效性且体长可用。像素标记、覆盖率和质量原因独立保留。
# _bl用于距离，_bl_s用于速度。无效BL量在内存为NaN、导出CSV为空。
REFERENCE_FIELDS = ["reference_body_length_px", "reference_length_sample_count",
                    "body_length_valid", "body_length_reason"]
FRAME_FIELDS = PIXEL_FRAME_FIELDS + [METRIC_NAMES[k] for k in PIXEL_FRAME_FIELDS if k in METRIC_NAMES] + REFERENCE_FIELDS + [
    "bl_distance_valid", "bl_direction_valid"]
SUMMARY_FIELDS = PIXEL_SUMMARY_FIELDS + [METRIC_NAMES[k] for k in PIXEL_SUMMARY_FIELDS if k in METRIC_NAMES] + REFERENCE_FIELDS + [
    "bl_distance_summary_valid", "bl_direction_summary_valid"]
BL_CURVES = [(METRIC_NAMES[field], label.replace("px", "BL"), "bl_" + flag)
             for field, label, flag in CURVES]
BL_SUMMARY_METRICS = [(METRIC_NAMES[field], label.replace("px", "BL"), "bl_" + flag)
                      for field, label, flag in SUMMARY_METRICS]


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


def resolve_frame_step(frame_step=None, *, frame_steps=None):
    """返回本次使用的一个正整数。省略时读取本文件顶部的 FRAME_STEP。

    新调用使用 frame_step=5。旧调用若只指定了一个步长，如 frame_steps=[5]，
    仍使用同一批窗口。多个步长报错，防止静默选错或写出混合结果。
    """
    if frame_step is not None and frame_steps is not None:
        raise ValueError("请只指定一个 frame_step 参数。")
    if frame_steps is not None or isinstance(frame_step, (list, tuple)):
        legacy = frame_steps if frame_steps is not None else frame_step
        try:
            (value,) = legacy
        except (TypeError, ValueError):
            raise ValueError("每次只能使用一个正整数步长，例如 frame_step=5。") from None
    else:
        value = FRAME_STEP if frame_step is None else frame_step
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("frame_step 必须是正整数，例如 1 或 5。")
    return int(value)


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
    row = {k: math.nan for k in PIXEL_FRAME_FIELDS}
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


def calculate_speed(records, frame_step=None, *, frame_steps=None):
    """按一个固定步长生成区间明细，主键为 (frame_step, id, frame)。

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
    step = resolve_frame_step(frame_step, frame_steps=frame_steps)
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
    for identity, lookup in sorted(groups.items()):
        observed_frames = sorted(lookup)
        anchor, last = observed_frames[0], observed_frames[-1]
        # 从实际观测定位固定窗口。格点上的帧同时是前一区间终点和后一区间起点。
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
    return sorted(output, key=lambda r: (r["frame"], r["id"]))


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


def estimate_body_lengths(records):
    """按ID返回参考体长信息，只读取输入，不写文件。

    L是该鱼本批有效中轴长度的中位数。缺失、非有限、<=0及无效几何不参与。
    无合格样本时返回长度NaN、样本数0、有效性0、原因no_valid_body_length。
    各鱼分别估计，不能用其他鱼的体长补值；只有一个合格样本也会记录计数1。
    """
    lengths = defaultdict(list)
    for record in records:
        values = lengths[int(record["id"])]
        length = _number(record.get("body_length_px"))
        if _valid(record.get("valid")) and math.isfinite(length) and length > 0:
            values.append(length)
    return {
        identity: dict(reference_body_length_px=float(median(values)) if values else math.nan,
                       reference_length_sample_count=len(values), body_length_valid=int(bool(values)),
                       body_length_reason="ok" if values else "no_valid_body_length")
        for identity, values in sorted(lengths.items())
    }


def _add_body_length_units(rows, references, summary=False):
    """复制每一行并追加BL数值、标尺和BL有效性，像素列与共享字段完整保留。"""
    flags = ("distance_summary_valid", "direction_summary_valid") if summary else ("distance_valid", "direction_valid")
    output = []
    for source in rows:
        ref = references[int(source["id"])]
        length = ref["reference_body_length_px"]
        row = dict(source)
        for pixel_field, bl_field in METRIC_NAMES.items():
            if pixel_field in source:
                row[bl_field] = _number(source[pixel_field]) / length if ref["body_length_valid"] else math.nan
        row.update(ref)
        for flag in flags:
            row["bl_" + flag] = int(_valid(source[flag]) and ref["body_length_valid"])
        output.append(row)
    return output


def analyze_records(records, frame_step=None, *, frame_steps=None):
    """统一纯计算入口，返回(像素与BL合并窗口明细, 像素与BL合并个体汇总)。

    Python调用示例：
        records, meta = geometry.load_geometry(绝对分割目录)
        rows, summaries = speed.analyze_records(records, frame_step=5)

    records为同一视频、同一顶部视角的完整几何批次，可为列表或迭代器。
    frame/id为整数，time_s为秒，cx/cy和body_length_px为像素，area为像素平方。
    valid/head_dx/head_dy提供中轴与头向质量，body_length_px提供逐帧中轴弧长。
    frame_step是本次唯一的正整数采样间隔，省略时读取顶部FRAME_STEP。
    每条鱼只估一次参考体长L，使用本批所有合格帧，与采样终点是否选中无关。
    rows每行一个(frame_step,id,frame)窗口，共53列，字段顺序见FRAME_FIELDS。
    summaries每行一条鱼，共37列，字段顺序见SUMMARY_FIELDS。
    像素列在前，对应BL列及体长/BL有效标记追加在后。原图位置等共享列只写一次。
    BL距离=原像素距离/L，BL速度=原像素速度/L。原前进/后退符号不重新判定。
    体长无效时本鱼全部BL运动量为NaN，CSV留空；像素量仍按自身质量解释。
    有效静止可输出数值0，帧不足或缺帧按原窗口原因标无效。
    函数只返回结果，不修改输入、不跨调用积累历史、不写文件。
    """
    step = resolve_frame_step(frame_step, frame_steps=frame_steps)
    records = list(records)
    pixel_rows = calculate_speed(records, step)
    pixel_summaries = summarize(pixel_rows)
    references = estimate_body_lengths(records)
    return (_add_body_length_units(pixel_rows, references),
            _add_body_length_units(pixel_summaries, references, summary=True))


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


def plot_curves(rows, output, step, *, curves=CURVES, unit="px"):
    """把六类趋势画在同一张图中。像素和BL各自传入对应的单位与有效性字段。"""
    plt = _plt()
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), constrained_layout=True)
    for ax, (field, label, valid_key) in zip(axes.ravel(), curves):
        _curve_panel(ax, rows, field, label, valid_key)
    fig.suptitle(f"Frame step {step} | Speed and distance over time ({unit}/s, {unit})")
    fig.savefig(output / f"speed_curves_{unit.lower()}.png", dpi=150)
    plt.close(fig)


def plot_summary(summaries, output, step, *, metrics=SUMMARY_METRICS, unit="px"):
    """画当前步长下各鱼的平均速度与累计距离，分别保留有符号和无符号统计。"""
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fish = sorted(summaries, key=lambda row: row["id"])
    x = np.arange(len(fish))
    for ax, (field, title, flag) in zip(axes.ravel(), metrics):
        values = [row[field] if row[flag] else math.nan for row in fish]
        ax.bar(x, values)
        ax.set(title=title, xlabel="SAM3 ID")
        ax.set_xticks(x, [str(row["id"]) for row in fish])
        ax.axhline(0, color="black", linewidth=.7)
        ax.grid(axis="y", alpha=.2)
        if not any(row[flag] for row in fish):
            ax.text(.5, .5, "No valid intervals", transform=ax.transAxes, ha="center")
    fig.suptitle(f"Frame step {step} | Mean speed and total distance by fish ({unit}/s, {unit})")
    fig.savefig(output / f"speed_summary_{unit.lower()}.png", dpi=160)
    plt.close(fig)


def draw_video(rows, records, meta, output, step, *, unit="px"):
    """用同一绘制流程输出px或BL视频，切换数值字段、单位和有效性。

    unit取px或BL。位置、头向、轨迹始终用原图像素，两个视频使用同批记录。
    仅在当前采样终点显示该窗口数值；非采样帧显示采样状态，缺体长显示N/A。
    原图路径全无或部分缺失时返回明确的skipped原因，不中止CSV/图导出，
    也不生成缺少若干原帧的缩短视频。frame_ids省略时使用已知原图和观测帧
    的并集，避免仅有部分image_paths时误当作完整视频。成功时返回saved。
    """
    if unit not in ("px", "BL"):
        raise ValueError("速度视频unit只接受px或BL。")
    is_bl = unit == "BL"
    suffix = unit.lower()
    speed_field = "signed_speed_bl_s" if is_bl else "signed_speed_px_s"
    distance_field = "signed_distance_bl" if is_bl else "signed_distance_px"
    valid_field = "bl_direction_valid" if is_bl else "direction_valid"
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
    video_path = output / f"speed_{suffix}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建{video_path.name}。")
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
                elif sample[valid_field]:
                    label = (f"ID {identity} [{sample['start_frame']}->{frame}] "
                             f"v={sample[speed_field]:+.3f} d={sample[distance_field]:+.4f}")
                else:
                    reason = sample["body_length_reason"] if is_bl and not sample["body_length_valid"] else sample["quality_reason"]
                    label = f"ID {identity} [{sample['start_frame']}->{frame}] N/A: {reason}"
                # 统一以深色底框提高两单位视频的文字可读性。
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .44, 1)
                x = max(2, min(center[0] - 80, width - label_width - 5))
                y = max(75, center[1] - 20)
                cv2.rectangle(image, (x - 2, y - label_height - 3),
                              (min(width - 1, x + label_width + 2), min(height - 1, y + baseline + 2)),
                              (0, 0, 0), -1)
                cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .44, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, f"SPEED {unit} step {step} | original frame {frame} | v:{unit}/s d:{unit}", (12, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "Numbers shown only at the current sample endpoint; lateral motion uses + sign", (12, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(image)
    finally:
        writer.release()
    capture, decoded = cv2.VideoCapture(str(video_path)), 0
    try:
        while capture.read()[0]:
            decoded += 1
    finally:
        capture.release()
    if decoded != len(frame_ids):
        raise RuntimeError(f"速度视频只有 {decoded}/{len(frame_ids)} 帧。")
    return "saved"


def _check_result_step(rows, summaries, step):
    """覆盖前检查预计算结果的步长和合并列，拒绝部分结果以免写出空BL报告。"""
    for group, fields in ((rows, FRAME_FIELDS), (summaries, SUMMARY_FIELDS)):
        if any(row["frame_step"] != step for row in group):
            raise ValueError(f"预计算结果中含有与 frame_step={step} 不一致的行。")
        for row in group:
            missing = [field for field in fields if field not in row]
            if missing:
                raise ValueError(f"预计算结果缺少合并字段{missing}，请使用analyze_records生成结果。")


def export_results(records, meta, output: Path, frame_step=None, save_video=True, *, results=None, frame_steps=None):
    """把同一批像素/BL明细和汇总导出到一个固定目录，返回(rows,summaries)。

    records：完整几何批次，输入字段及换算条件见analyze_records。
    meta：来源信息、时间来源、帧清单。视频还需image_paths（帧号→原图绝对路径）、
          width/height、fps。空输入或只有一帧时也导出字段表头/占位及无有效结果图。
    output：绝对结果目录，重跑覆盖本模块生成的固定文件。
    frame_step：本次唯一正整数步长，省略时读取FRAME_STEP。
    save_video：True请求px和BL两段视频；缺原图时分别记录跳过原因，CSV正常写入。
    results：None时本函数调用analyze_records。综合入口可传本批预计算的合并
             (rows,summaries)，直接导出，不重复计算像素运动或参考体长。

    两张CSV每行同时含像素、BL及共用字段，采用UTF-8 BOM，缺失数值为空。
    两张speed_curves_*.png各含六类时间曲线，两张speed_summary_*.png各含
    个体均速和总距离的四面板。speed_px.mp4/speed_bl.mp4在同一帧显示各自单位。
    run_info.json保存参数、每鱼参考体长/样本数和video_status字典，px和bl各有
    saved/disabled/skipped_*等状态。发生视频错误时该项记failed并保留错误原因。
    单独重算后，综合标准JSON由main重新生成。
    """
    step = resolve_frame_step(frame_step, frame_steps=frame_steps)
    records = list(records)
    output = Path(output)
    # results只供统一入口复用同批次预计算结果；独立运行仍自行计算。
    if results is None:
        rows, summaries = analyze_records(records, step)
    else:
        rows, summaries = results
        rows, summaries = list(rows), list(summaries)
        _check_result_step(rows, summaries, step)
    output = prepare_speed_output(output)
    # 每鱼标尺已经由analyze_records计算，直接从汇总行读取用于审计。
    references = {
        str(row["id"]): {field: (None if isinstance(row[field], float) and not math.isfinite(row[field]) else row[field])
                         for field in REFERENCE_FIELDS}
        for row in summaries
    }
    info = dict(status="running", segmentation_dir=meta.get("segmentation_dir"),
                time_source=meta.get("time_source", "provided record time_s"),
                units=["pixel", "pixel/s", "BL", "BL/s"], coordinate_unit="pixel", area_unit="pixel^2",
                frame_step=step, head_hints=meta.get("head_hints", HEAD_HINTS), centroid_smoothing=False,
                use_filename_timestamps=meta.get("use_filename_timestamps", USE_FILENAME_TIMESTAMPS), save_video_requested=bool(save_video),
                video_status={"px": "pending" if save_video else "disabled", "bl": "pending" if save_video else "disabled"},
                trail_frames=TRAIL_FRAMES, fps=meta.get("fps"),
                frame_ids=[int(f) for f in meta.get("frame_ids", sorted({int(r["frame"]) for r in records} |
                          {int(f) for f in (meta.get("image_paths") or {})}))],
                max_area_change_ratio=MAX_AREA_CHANGE_RATIO, max_adjacent_head_turn_deg=MAX_HEAD_TURN_DEG,
                sign_dot_tolerance=SIGN_DOT_TOLERANCE,
                max_explicit_missing_frames=MAX_EXPLICIT_MISSING_FRAMES,
                body_length_method="Per-ID median of finite positive body_length_px from all valid geometry records, independent of frame_step",
                body_length_definition="Image-estimated head-to-tail centerline arc length",
                body_lengths=references,
                conversion="BL = pixel / reference_body_length_px; BL/s = (pixel/s) / reference_body_length_px",
                validity_definition="Pixel validity, time coverage and observations are shared; BL validity additionally requires this ID's reference body length",
                missing_length_policy="Pixel quantities retain their own validity; all BL quantities are blank, with body_length_valid=0 and bl_*_valid=0",
                signed_distance_definition="Euclidean endpoint distance * sign(dot(displacement, normalized sum of normalized endpoint head axes)); dot>=-tolerance is positive",
                forward_projection_definition="Diagnostic only; NOT signed_distance_px",
                lateral_projection_definition="dot(displacement,[forward_y,-forward_x]); positive points up when head points right in image coordinates",
                forward_backward_totals_definition="Sums of positive signed distance and absolute negative signed distance; pure lateral intervals belong to the positive total",
                sampling="Fixed anchor at each ID first observed frame; non-overlapping; missing frames never reset anchor; incomplete tail excluded; entirely unobserved windows omitted without changing the anchor",
                frame_range_definition="Actually observed frame IDs in this window; missing_frame_count and inclusive missing_frame_ranges are always provided; missing_frames is expanded only within max_explicit_missing_frames",
                cumulative_definition="Sum of valid endpoint distances separately for each frame_step and ID; not the sum of inner one-frame path lengths",
                average_definition="Corresponding valid total distance / corresponding valid total time; divide by per-ID body length for BL/s",
                observed_records=len(records), sample_rows=len(rows), summary_rows=len(summaries))
    info_path = output / "run_info.json"
    def save_info():
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    save_info()
    try:
        save_csv(output / "speed_frames.csv", rows, FRAME_FIELDS)
        save_csv(output / "speed_summary.csv", summaries, SUMMARY_FIELDS)
        for unit, curves, metrics in (("px", CURVES, SUMMARY_METRICS), ("BL", BL_CURVES, BL_SUMMARY_METRICS)):
            plot_curves(rows, output, step, curves=curves, unit=unit)
            plot_summary(summaries, output, step, metrics=metrics, unit=unit)
            if save_video:
                try:
                    info["video_status"][unit.lower()] = draw_video(rows, records, meta, output, step, unit=unit)
                except BaseException:
                    info["video_status"][unit.lower()] = "failed"
                    raise
            save_info()
        info["status"] = "complete"
    except BaseException as exc:
        info.update(status="failed", error=str(exc))
        raise
    finally:
        save_info()
    return rows, summaries


def main():
    r"""独立入口，与main共用output/behavior/speed，同时生成两种单位。

    Python函数调用方式，调用方需能导入项目根目录。
        from pathlib import Path
        from behavior import geometry, speed
        records, meta = geometry.load_geometry(
            Path(r"E:\Phd\temperature\sam3\top\output\tracking"))
        rows, summaries = speed.analyze_records(records, frame_step=5)
        speed.export_results(records, meta,
            Path(r"E:\Phd\temperature\sam3\top\output\behavior\speed"),
            frame_step=5, save_video=True, results=(rows, summaries))

    外部系统有标准分割data时，先用geometry.geometry_from_data(data)取得几何，
    再传给analyze_records；需要标准behavior_indicator返回时使用main.analyze_behavior。
    """
    if not SEGMENTATION_DIR.is_absolute() or not OUTPUT_DIR.is_absolute():
        raise ValueError("SEGMENTATION_DIR 和 OUTPUT_DIR 必须使用绝对路径。")
    from behavior.geometry import load_geometry
    step = resolve_frame_step()
    records, meta = load_geometry(SEGMENTATION_DIR, head_hints=HEAD_HINTS,
                                  use_filename_timestamps=USE_FILENAME_TIMESTAMPS)
    output = OUTPUT_DIR
    rows, summaries = export_results(records, meta, output, step, SAVE_VIDEO)
    print(f"距离/速度分析完成：{output}\n步长={step}，采样记录={len(rows)}，汇总={len(summaries)}")
    return output


if __name__ == "__main__":
    main()
