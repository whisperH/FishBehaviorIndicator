# 顶部视角维护负责人：YHC。此脚本仅属于 FishBehavior/Top，不依赖 Side 子项目。
"""独立运动分析：只读 Tracker/Output/Information/result.csv，不调用其他项目脚本。

速度采用区间首末质心的完整欧氏距离，正负只由首末鱼头平均方向决定；
纯侧移取正。BL 是每条鱼自己的有效中轴线弧长中位数，不是全鱼共用体长。
本脚本的数值输出仅使用 BL/s、BL/s²；cx/cy/参考体长仍用像素以便复核定位。
有符号加速度为二维加速度大小乘鱼头方向判号，不是有符号速度的数值差分。
各参数、无效原因与字段含义见下方注释及项目 README.md。
"""
from pathlib import Path
import csv
import json
import math
from collections import defaultdict, deque
from bisect import bisect_left, bisect_right
from statistics import median
import cv2
import numpy as np

# ==================== 用户修改区：全部使用绝对路径 ====================
INPUT_CSV = Path(r"E:\Phd\FishBehavior\Top\Tracker\Output\Information\result.csv")
OUTPUT_DIR = Path(r"E:\Phd\FishBehavior\Top\Behavior\Output\speed")
FRAME_STEP = 1                 # 一个步长；1=邻帧，5=首观测 A→A+5→A+10。缺帧不重新定锚。
MAX_AREA_CHANGE_RATIO = 1.5   # 窗口内部每对相邻掩膜面积比上限；超过则运动无效。
MAX_HEAD_TURN_DEG = 90.0      # 窗口内部每对相邻头向允许转角，不限制累计转角。
SIGN_DOT_TOLERANCE = 1e-9     # 容差内的纯侧移记正；投影仅用于符号，完整位移用于大小。
MAX_EXPLICIT_MISSING_FRAMES = 1000
SAVE_VIDEO = True            # 关闭后移除本模块原有视频，避免新表格搭配旧视频。
TRAIL_FRAMES = 30            # 仅控制轨迹显示，不平滑计算输入。
# ====================================================================

def _finite_number(value, name, optional=False):
    """CSV 缺失数值只允许空字符串；非法文本/Inf/NaN 不静默变成有效零。"""
    if value is None or str(value).strip() == "":
        if optional:
            return math.nan
        raise ValueError(f"缺少数值字段 {name}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"字段 {name} 不是数字: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"字段 {name} 必须是有限数或合法的空值")
    return number


def _int_value(value, name, minimum=0):
    number = _finite_number(value, name)
    if not number.is_integer() or number < minimum:
        raise ValueError(f"字段 {name} 必须是 >= {minimum} 的整数")
    return int(number)


def _bool_value(value, name):
    if str(value).strip().lower() in ("1", "true"):
        return True
    if str(value).strip().lower() in ("0", "false"):
        return False
    raise ValueError(f"字段 {name} 必须为 0/1 或 true/false")


def load_records(input_csv=INPUT_CSV):
    """读取唯一数值输入 result.csv，返回 (个体记录, 完整帧元信息)。

    每条 observation 是一帧中的一条鱼；empty_frame 仅保留完整视频时间线，
    不伪造鱼 ID。重复 ID/帧、混视频、时间倒退、参数不一致直接报错。
    position_valid 与 head_valid 独立：头向失败仍可估无符号运动；位置无效
    则不计算运动。参考 BL 与本批有效中轴长度再次核对，避免误用别的鱼标尺。
    """
    path = Path(input_csv)
    if not path.is_absolute():
        raise ValueError("输入 CSV 必须为绝对路径")
    required = set("schema_version run_id record_type video_id frame_id time_s fps time_source image_path image_width image_height track_id cx cy area_px head_dx head_dy heading_deg head_confidence body_length_px reference_body_length_px reference_length_sample_count position_valid geometry_valid head_valid quality_reason".split())
    records, timeline, seen, empty_frames = [], {}, set(), set()
    source_key = None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("result.csv 缺少字段: " + ", ".join(sorted(missing)))
        for line, item in enumerate(reader, 2):
            if _int_value(item["schema_version"], "schema_version") != 1:
                raise ValueError("不支持的 result.csv 版本")
            frame = _int_value(item["frame_id"], "frame_id")
            time_s = _finite_number(item["time_s"], "time_s")
            fps = _finite_number(item["fps"], "fps")
            width = _int_value(item["image_width"], "image_width", 1)
            height = _int_value(item["image_height"], "image_height", 1)
            if time_s < 0 or fps <= 0:
                raise ValueError("time_s 必须非负，fps 必须正数")
            if not item["video_id"] or not item["run_id"] or not item["time_source"]:
                raise ValueError("视频/运行标识或时间来源为空")
            key = (item["video_id"], item["run_id"], fps, width, height, item["time_source"])
            if source_key is not None and source_key != key:
                raise ValueError("同一张表不能混入不同视频、运行批次、帧率、尺寸或时间来源")
            source_key = key
            image_path = Path(item["image_path"])
            if not image_path.is_absolute():
                raise ValueError(f"第 {line} 行 image_path 必须为绝对路径")
            meta_row = dict(frame_id=frame, time_s=time_s, image_path=str(image_path))
            if frame in timeline and timeline[frame] != meta_row:
                raise ValueError("同帧的时间和原图路径不一致")
            timeline[frame] = meta_row
            if item["record_type"] == "empty_frame":
                if item["track_id"].strip() or frame in empty_frames:
                    raise ValueError("空帧不应有鱼 ID 或重复占位行")
                if any(_bool_value(item[name], name) for name in ("position_valid", "geometry_valid", "head_valid")):
                    raise ValueError("空帧的有效性必须全部为 0")
                empty_frames.add(frame)
                continue
            if item["record_type"] != "observation":
                raise ValueError(f"未知 record_type: {item['record_type']}")
            identity = _int_value(item["track_id"], "track_id")
            if (frame, identity) in seen:
                raise ValueError(f"重复观测: frame={frame}, track_id={identity}")
            seen.add((frame, identity))
            row = dict(item)
            numeric = "cx cy area_px head_dx head_dy heading_deg head_confidence body_length_px reference_body_length_px".split()
            for name in numeric:
                row[name] = _finite_number(item[name], name, optional=True)
            for name in ("position_valid", "geometry_valid", "head_valid"):
                row[name] = _bool_value(item[name], name)
            row["reference_length_sample_count"] = _int_value(item["reference_length_sample_count"], "reference_length_sample_count")
            if row["position_valid"]:
                if not (math.isfinite(row["cx"]) and math.isfinite(row["cy"]) and 0 <= row["cx"] < width and 0 <= row["cy"] < height and row["area_px"] > 0):
                    raise ValueError("位置有效但质心/面积无效")
            else:
                row["cx"] = row["cy"] = math.nan
            if row["head_valid"] and (not row["geometry_valid"] or not np.isfinite([row["head_dx"], row["head_dy"]]).all() or math.hypot(row["head_dx"], row["head_dy"]) < 1e-8):
                raise ValueError("头向有效标记与实际头轴不一致")
            row.update(frame=frame, frame_id=frame, id=identity, track_id=identity, time_s=time_s,
                       area=row["area_px"], valid=row["head_valid"], filename=image_path.name)
            records.append(row)
    if not timeline:
        raise ValueError("result.csv 没有任何帧记录")
    if empty_frames.intersection(frame for frame, _ in seen):
        raise ValueError("同一帧不能既是空帧又有鱼观测")
    frames = sorted(timeline)
    if any(timeline[b]["time_s"] <= timeline[a]["time_s"] for a, b in zip(frames, frames[1:])):
        raise ValueError("帧号递增时采集时间必须严格递增")
    records.sort(key=lambda row: (row["frame"], row["id"]))
    references_for_records(records)  # 只核对标尺，不修改输入。
    return records, dict(timeline=timeline, frame_ids=frames, fps=source_key[2], width=source_key[3], height=source_key[4],
                         video_id=source_key[0], run_id=source_key[1], time_source=source_key[5])


def references_for_records(records):
    """每鱼参考体长来自基础表；纯函数调用未提供参考列时由有效中轴估计。

    所有帧共用同一参考长度；缺失 BL 不借用其他鱼，不以零代替。
    CSV 已声明参考长度时，长度与样本数必须与本批有效中轴一致。
    """
    references = estimate_body_lengths([dict(r, valid=r.get("geometry_valid", r.get("valid", False))) for r in records])
    for r in records:
        if "reference_body_length_px" not in r:
            continue
        ref = references[int(r["id"])]
        supplied = _number(r["reference_body_length_px"])
        expected = ref["reference_body_length_px"]
        same = (math.isnan(supplied) and math.isnan(expected)) or (math.isfinite(supplied) and math.isfinite(expected) and math.isclose(supplied, expected, rel_tol=1e-9, abs_tol=1e-9))
        if not same or int(r["reference_length_sample_count"]) != ref["reference_length_sample_count"]:
            raise ValueError(f"鱼 {r['id']} 的参考体长或样本数与有效中轴不一致")
    return references


def _prepare_output(output_dir, names):
    """只清理本模块确知的固定文件，不递归删除任何目录。"""
    output = Path(output_dir)
    if not output.is_absolute():
        raise ValueError("输出目录必须为绝对路径")
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = output / name
        if path.is_file():
            path.unlink()
    return output


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取原图: {path}")
    return image


def _write_csv(path, rows, fields):
    """字段顺序固定；列表/字典写 JSON，非有限数写空单元格，0 仍保留 0。"""
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for name in fields:
                value = row.get(name)
                if value is None or isinstance(value, (float, np.floating)) and not math.isfinite(value):
                    value = ""
                elif isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                clean[name] = value
            writer.writerow(clean)


def _canonical(row):
    """导出与基础表一致的 frame_id/track_id 命名；算法内部保留原版别名便于复核。"""
    result = dict(row)
    if "frame" in result:
        result["frame_id"] = result["frame"]
    result["track_id"] = result["id"]
    if "area" in result:
        result["area_px"] = result["area"]
    return result

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

PIXEL_SUMMARY_FIELDS = [
    "frame_step", "id", "anchor_frame", "start_frame", "end_frame", "frame_range", "observations",
    "sample_rows", "candidate_intervals", "incomplete_intervals", "track_span_s",
    "valid_distance_intervals", "valid_signed_intervals", "valid_distance_time_s", "valid_signed_time_s",
    "distance_time_coverage", "signed_time_coverage", "path_distance_px", "signed_distance_px",
    "forward_distance_px", "backward_distance_px", "mean_speed_px_s", "mean_signed_speed_px_s",
    "distance_summary_valid", "direction_summary_valid",
]

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

REFERENCE_FIELDS = ["reference_body_length_px", "reference_length_sample_count",
                    "body_length_valid", "body_length_reason"]

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
        if "position_valid" in item and not _valid(item["position_valid"]):
            row.update(quality_reason="invalid_centroid", quality_detail=f"frame={frame}: position_valid=0")
            return
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

    records 是 load_records 的字典列表：frame/id 为整数，time_s
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
    """纯计算入口：同一批几何记录 -> (窗口明细, 每鱼汇总)。

    Python 调用：records, meta = load_records(INPUT_CSV)
    rows, summaries = analyze_records(records, frame_step=5)
    frame_step 为一个正整数；以每鱼第一次观测帧为固定采样锚。
    内部保留像素/BL，导出层按本指标选择列；不写文件、不修改输入。
    """
    step = resolve_frame_step(frame_step, frame_steps=frame_steps)
    records = list(records)
    pixel_rows = calculate_speed(records, step)
    pixel_summaries = summarize(pixel_rows)
    references = references_for_records(records)
    return (_add_body_length_units(pixel_rows, references),
            _add_body_length_units(pixel_summaries, references, summary=True))


def add_acceleration(rows, records):
    """新增加速度：相邻且连续的两个有效速度窗口之间作二维向量差分。

    v=(dx/dt,dy/dt)，时间位置取各速度窗口的中点；a=(v2-v1)/(tmid2-tmid1)。
    无符号值=||a||/L；有符号值=无符号值×sign(a·平均头向)，纯侧向取正。
    这不是 d(signed_speed)/dt。前进减速时可为负，恒速转弯仍可有加速度。
    中间缺帧/运动无效会切断差分，不能越过缺口继续计算。加速度的证据帧
    acceleration_frame_range 包含两个窗口，和仅描述本次速度的 frame_range 分开。
    """
    lookup = {(r["id"], r["frame"]): r for r in records}
    groups = defaultdict(list)
    for row in rows:
        groups[row["id"]].append(row)
        row.update(velocity_x_bl_s=math.nan, velocity_y_bl_s=math.nan,
                   acceleration_x_bl_s2=math.nan, acceleration_y_bl_s2=math.nan,
                   acceleration_bl_s2=math.nan, signed_acceleration_bl_s2=math.nan,
                   acceleration_dt_s=math.nan, acceleration_time_s=math.nan,
                   acceleration_frame_range=[], acceleration_valid=0, signed_acceleration_valid=0,
                   acceleration_reason="needs_two_contiguous_velocity_windows", signed_acceleration_reason="needs_two_contiguous_velocity_windows")
        if row["bl_distance_valid"]:
            row["velocity_x_bl_s"] = row["dx_bl"] / row["dt_s"]
            row["velocity_y_bl_s"] = row["dy_image_bl"] / row["dt_s"]
    for identity, group in groups.items():
        previous = None
        for row in sorted(group, key=lambda r: r["frame"]):
            if not row["distance_valid"]:
                row["acceleration_reason"] = row["signed_acceleration_reason"] = row["quality_reason"]
                previous = None
                continue
            if previous is not None and row["start_frame"] == previous["frame"]:
                tm1 = (previous["start_time_s"] + previous["time_s"]) / 2
                tm2 = (row["start_time_s"] + row["time_s"]) / 2
                dt = tm2 - tm1
                row.update(acceleration_dt_s=dt, acceleration_time_s=(tm1 + tm2)/2,
                           acceleration_frame_range=sorted(set(previous["frame_range"] + row["frame_range"])))
                if dt > 0 and row["body_length_valid"]:
                    v1 = np.array([previous["dx_px"], previous["dy_image_px"]]) / previous["dt_s"]
                    v2 = np.array([row["dx_px"], row["dy_image_px"]]) / row["dt_s"]
                    acc = (v2-v1) / dt
                    length = row["reference_body_length_px"]
                    magnitude = float(np.linalg.norm(acc))
                    row.update(acceleration_x_bl_s2=float(acc[0])/length, acceleration_y_bl_s2=float(acc[1])/length,
                               acceleration_bl_s2=magnitude/length, acceleration_valid=1, acceleration_reason="ok")
                    if row["direction_valid"] and previous["direction_valid"]:
                        axes = []
                        for window in (previous, row):
                            first, last = lookup[(identity, window["start_frame"])], lookup[(identity, window["frame"])]
                            a = np.array([first["head_dx"], first["head_dy"]], dtype=float)
                            b = np.array([last["head_dx"], last["head_dy"]], dtype=float)
                            axis = a/np.linalg.norm(a) + b/np.linalg.norm(b)
                            axes.append(axis/np.linalg.norm(axis))
                        head = axes[0]+axes[1]
                        if np.linalg.norm(head) >= 1e-8:
                            head /= np.linalg.norm(head)
                            sign = 1 if np.dot(acc, head) >= -SIGN_DOT_TOLERANCE*max(1, magnitude) else -1
                            row.update(signed_acceleration_bl_s2=sign*magnitude/length,
                                       signed_acceleration_valid=1, signed_acceleration_reason="ok")
                        else:
                            row["signed_acceleration_reason"] = "acceleration_head_axis_cancellation"
                    else:
                        row["signed_acceleration_reason"] = "head_axis_invalid_in_acceleration_evidence"
                else:
                    row["acceleration_reason"] = row["signed_acceleration_reason"] = (
                        "no_valid_body_length" if not row["body_length_valid"] else "invalid_acceleration_time")
            previous = row
    return rows


# 输出字段定义：列表字段都是可解析 JSON，所有无效测量为空，累计量须结合有效性使用。
INTERVAL_FIELDS = ["frame_id", "track_id", "frame_step", "anchor_frame", "start_frame", "expected_end_frame",
    "frame_range", "window_complete", "missing_frame_count", "missing_frame_ranges", "start_time_s", "time_s", "dt_s",
    "cx", "cy", "heading_deg", "reference_body_length_px", "reference_length_sample_count", "body_length_valid", "body_length_reason"]
SPEED_FRAME_FIELDS = INTERVAL_FIELDS + ["speed_bl_s", "signed_speed_bl_s", "velocity_x_bl_s", "velocity_y_bl_s",
    "bl_distance_valid", "bl_direction_valid", "quality_reason", "quality_detail", "acceleration_x_bl_s2", "acceleration_y_bl_s2",
    "acceleration_bl_s2", "signed_acceleration_bl_s2", "acceleration_time_s", "acceleration_dt_s", "acceleration_frame_range",
    "acceleration_valid", "signed_acceleration_valid", "acceleration_reason", "signed_acceleration_reason"]
SPEED_SUMMARY_FIELDS = ["track_id", "frame_step", "start_frame", "end_frame", "observations", "track_span_s",
    "valid_distance_intervals", "valid_signed_intervals", "valid_distance_time_s", "valid_signed_time_s", "distance_time_coverage", "signed_time_coverage",
    "reference_body_length_px", "reference_length_sample_count", "body_length_valid", "body_length_reason",
    "mean_speed_bl_s", "mean_signed_speed_bl_s", "bl_distance_summary_valid", "bl_direction_summary_valid"]


def plot_speed(rows, summaries, output):
    """ABC 为无符号速度/加速度/每鱼平均速度，DEF 对应有符号量；平均值为柱图。"""
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["id"]].append(row)
    panels = [(axes[0,0], "speed_bl_s", "bl_distance_valid", "A  Speed (BL/s)", "time_s"),
              (axes[0,1], "acceleration_bl_s2", "acceleration_valid", "B  Acceleration (BL/s²)", "acceleration_time_s"),
              (axes[1,0], "signed_speed_bl_s", "bl_direction_valid", "D  Signed speed (BL/s)", "time_s"),
              (axes[1,1], "signed_acceleration_bl_s2", "signed_acceleration_valid", "E  Signed acceleration (BL/s²)", "acceleration_time_s")]
    for ax, field, flag, label, time_field in panels:
        for identity, group in sorted(groups.items()):
            ax.plot([r[time_field] for r in group], [r[field] if r[flag] else np.nan for r in group], label=f"ID {identity}", lw=1)
        ax.set(title=label, xlabel="Time (s)")
        ax.grid(alpha=.25)
        ax.axhline(0, color="0.6", lw=.5)
        if groups:
            ax.legend(fontsize=8, ncol=2)
    for ax, field, flag, label in [(axes[0,2], "mean_speed_bl_s", "bl_distance_summary_valid", "C  Mean speed (BL/s)"),
                                  (axes[1,2], "mean_signed_speed_bl_s", "bl_direction_summary_valid", "F  Mean signed speed (BL/s)")]:
        ax.bar([str(r["id"]) for r in summaries], [r[field] if r[flag] else np.nan for r in summaries])
        ax.set(title=label, xlabel="Fish ID")
        ax.grid(axis="y", alpha=.25)
    if not groups:
        axes[0,0].text(.5,.5,"No fish observations",ha="center", transform=axes[0,0].transAxes)
    fig.suptitle("Head-relative sign; lateral positive. Gaps = unavailable; means use valid elapsed time.")
    fig.savefig(output / "speed_bl.png", dpi=150)
    plt.close(fig)


def make_speed_video(rows, records, meta, output, frame_step):
    """完整帧时间线逐帧写视频；稀疏采样显示最近已完成窗口并注明 sample 帧号。

    播放帧率使用基础表 fps；计算时间始终使用实际 time_s，二者不混用。
    原图不全则报错，不把上次视频留在新结果中。没有鱼也写背景帧。
    """
    target = output / "speed_bl.mp4"
    temporary = output / ".speed_bl.part.mp4"
    for frame in meta["frame_ids"]:
        if not Path(meta["timeline"][frame]["image_path"]).is_file():
            raise FileNotFoundError(meta["timeline"][frame]["image_path"])
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (meta["width"], meta["height"]))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("无法打开 mp4v 视频编码器")
    by_frame = defaultdict(list)
    samples = defaultdict(list)
    for record in records:
        by_frame[record["frame"]].append(record)
    for row in rows:
        samples[row["frame"]].append(row)
    latest, trails, last_frame = {}, defaultdict(lambda: deque(maxlen=TRAIL_FRAMES)), {}
    try:
        for frame in meta["frame_ids"]:
            image = _read_image(meta["timeline"][frame]["image_path"])
            if image.shape[:2] != (meta["height"], meta["width"]):
                raise ValueError("原图尺寸与 result.csv 不一致")
            for row in samples[frame]:
                latest[row["id"]] = row
            cv2.putText(image, f"frame {frame} | step {frame_step} | signed v: BL/s | signed a: BL/s2", (15,25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2, cv2.LINE_AA)
            for record in by_frame[frame]:
                identity = record["id"]
                if not record.get("position_valid", True) or not np.isfinite([record["cx"],record["cy"]]).all():
                    trails[identity].clear()
                    continue
                color = tuple(int(v) for v in np.random.default_rng(identity).integers(80,255,3))
                center = (int(round(record["cx"])), int(round(record["cy"])))
                if last_frame.get(identity, frame-1) != frame-1:
                    trails[identity].clear()
                trails[identity].append(center); last_frame[identity]=frame
                if len(trails[identity]) > 1:
                    cv2.polylines(image, [np.array(trails[identity],dtype=np.int32)], False, color, 2)
                cv2.circle(image, center, 5, color, -1)
                row=latest.get(identity)
                speed = f"{row['signed_speed_bl_s']:+.3f}" if row and row["bl_direction_valid"] else "n/a"
                acc = f"{row['signed_acceleration_bl_s2']:+.3f}" if row and row["signed_acceleration_valid"] else "n/a"
                sample = row["frame"] if row else "-"
                label=f"ID {identity} v={speed} a={acc} sample={sample}"
                cv2.putText(image,label,(max(0,center[0]-80),max(45,center[1]-18)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,2,cv2.LINE_AA)
            writer.write(image)
    except BaseException:
        writer.release()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        writer.release()
    capture=cv2.VideoCapture(str(temporary))
    count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));capture.release()
    if count != len(meta["frame_ids"]):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"视频帧数不一致: {count}")
    temporary.replace(target)


def run(input_csv=INPUT_CSV, output_dir=OUTPUT_DIR, *, frame_step=None, save_video=SAVE_VIDEO):
    """独立运行/API：run(绝对 result.csv 路径, 绝对输出目录, frame_step=1, save_video=True)。

    input_csv 为 geometry.py 输出，output_dir 固定文件名覆盖，frame_step 为一个
    正整数，save_video 仅控制渲染。返回明细/汇总字典列表，不跨调用保存状态。
    只有 bl_*_valid 或 acceleration_valid=1 才可用于统计；空值不等于静止。
    """
    step = resolve_frame_step(frame_step)
    records, meta = load_records(input_csv)
    rows, summaries = analyze_records(records, frame_step=step)
    add_acceleration(rows, records)
    output=_prepare_output(output_dir, ["speed_frames.csv","speed_summary.csv","speed_bl.png","speed_bl.mp4",".speed_bl.part.mp4"])
    _write_csv(output/"speed_frames.csv", map(_canonical, rows), SPEED_FRAME_FIELDS)
    _write_csv(output/"speed_summary.csv", map(_canonical, summaries), SPEED_SUMMARY_FIELDS)
    plot_speed(rows,summaries,output)
    if save_video:
        make_speed_video(rows,records,meta,output,step)
    print(f"速度分析完成：{len(summaries)} 条鱼，步长 {step}，输出 {output}")
    return {"frames": [{k: _canonical(r).get(k) for k in SPEED_FRAME_FIELDS} for r in rows],
            "summary": [{k: _canonical(r).get(k) for k in SPEED_SUMMARY_FIELDS} for r in summaries]}


if __name__ == "__main__":
    run()
