# 顶部视角维护负责人：YHC。此脚本仅属于 FishBehavior/Top，不依赖 Side 子项目。
"""独立静止时间分析：只读 Tracker/Output/Information/result.csv，不调用其他项目脚本。

速度采用区间首末质心的完整欧氏距离，正负只由首末鱼头平均方向决定；
纯侧移取正。BL 是每条鱼自己的有效中轴线弧长中位数，不是全鱼共用体长。
本脚本保存静止持续时间（秒）及比例，以无符号 BL/s 速率判定。
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
OUTPUT_DIR = Path(r"E:\Phd\FishBehavior\Top\Behavior\Output\resting_time")
FRAME_STEP = 1                 # 一个步长；1=邻帧，5=首观测 A→A+5→A+10。缺帧不重新定锚。
MAX_AREA_CHANGE_RATIO = 1.5   # 窗口内部每对相邻掩膜面积比上限；超过则运动无效。
MAX_HEAD_TURN_DEG = 90.0      # 窗口内部每对相邻头向允许转角，不限制累计转角。
SIGN_DOT_TOLERANCE = 1e-9     # 容差内的纯侧移记正；投影仅用于符号，完整位移用于大小。
MAX_EXPLICIT_MISSING_FRAMES = 1000
SAVE_VIDEO = True            # 关闭后移除本模块原有视频，避免新表格搭配旧视频。
TRAIL_FRAMES = 30            # 仅控制轨迹显示，不平滑计算输入。
REST_SPEED_THRESHOLD_BL_S = 0.05  # resting_time.py 使用；小于阈值才算静止。
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


INTERVAL_FIELDS = ["frame_id", "track_id", "frame_step", "anchor_frame", "start_frame", "expected_end_frame",
    "frame_range", "window_complete", "missing_frame_count", "missing_frame_ranges", "start_time_s", "time_s", "dt_s",
    "cx", "cy", "heading_deg", "reference_body_length_px", "reference_length_sample_count", "body_length_valid", "body_length_reason"]

REST_SUMMARY_FIELDS=["track_id","start_frame","end_frame","observations","track_span_s","threshold_bl_s",
    "reference_body_length_px","reference_length_sample_count","body_length_valid","body_length_reason",
    "valid_intervals","rest_intervals","moving_intervals","valid_time_s","rest_time_s","moving_time_s","unknown_time_s",
    "valid_time_coverage","rest_fraction","rest_event_count","longest_rest_s","rest_episodes","summary_valid","quality_reason"]


def analyze_rest(records, threshold_bl_s=REST_SPEED_THRESHOLD_BL_S):
    """用相邻原始帧无符号速率判定静止，固定步长1，不受速度脚本参数影响。

    速率 < threshold_bl_s 为静止，等于阈值为运动；倒退先取移动大小，不能因
    负号判静止。时间累计每个相邻有效区间实际 dt；首帧不虚加1/fps，尾帧不
    外推。缺帧、异常面积、无效位置或无参考体长均为未知。连续静止区间合并
    为事件，缺口或运动立即切断。rest_fraction分母仅为可判断的有效时长。
    """
    if isinstance(threshold_bl_s,bool) or not math.isfinite(float(threshold_bl_s)) or threshold_bl_s<0:
        raise ValueError("静止阈值必须为非负有限 BL/s 数值")
    rows,summaries=analyze_records(records,frame_step=1)
    groups=defaultdict(list)
    for row in rows:groups[row["id"]].append(row)
    output=[]
    for source in summaries:
        identity=source["id"]
        group=sorted(groups[identity],key=lambda r:r["frame"])
        events=[];active=None;rest_time=0.;moving_time=0.;rest_count=0;moving_count=0
        for row in group:
            valid=bool(row["bl_distance_valid"])
            resting=valid and row["speed_bl_s"]<threshold_bl_s
            row["rest_state"]="resting" if resting else "moving" if valid else "unknown"
            if resting:
                rest_time+=row["dt_s"];rest_count+=1
                if active is None or active["end_frame"]!=row["start_frame"]:
                    if active is not None:events.append(active)
                    active=dict(start_frame=row["start_frame"],end_frame=row["frame"],start_time_s=row["start_time_s"],end_time_s=row["time_s"],duration_s=row["dt_s"])
                else:
                    active.update(end_frame=row["frame"],end_time_s=row["time_s"],duration_s=active["duration_s"]+row["dt_s"])
            else:
                if active is not None:events.append(active);active=None
                if valid:moving_time+=row["dt_s"];moving_count+=1
        if active is not None:events.append(active)
        valid_time=rest_time+moving_time
        span=source["track_span_s"]
        unknown=max(0.,span-valid_time) if math.isfinite(span) else math.nan
        good=valid_time>0
        reason="ok" if good else "no_valid_body_length" if not source["body_length_valid"] else "no_valid_adjacent_intervals"
        result=dict(track_id=identity,start_frame=source["start_frame"],end_frame=source["end_frame"],observations=source["observations"],
            track_span_s=span,threshold_bl_s=threshold_bl_s,valid_intervals=rest_count+moving_count,rest_intervals=rest_count,moving_intervals=moving_count,
            valid_time_s=valid_time,rest_time_s=rest_time if good else math.nan,moving_time_s=moving_time if good else math.nan,
            unknown_time_s=unknown,valid_time_coverage=valid_time/span if span>0 else math.nan,
            rest_fraction=rest_time/valid_time if good else math.nan,rest_event_count=len(events) if good else None,
            longest_rest_s=max((e["duration_s"] for e in events),default=0.) if good else math.nan,
            rest_episodes=events,summary_valid=int(good),quality_reason=reason)
        for key in REFERENCE_FIELDS:result[key]=source[key]
        output.append(result)
    return rows,output


def plot_rest(rows,summaries,output):
    """每鱼一个时间条：蓝色静止、橙色运动、灰色未知；未知缺口不画成静止。"""
    plt=_plt()
    from matplotlib.patches import Patch
    groups=defaultdict(list)
    for row in rows:groups[row["id"]].append(row)
    n=max(1,len(summaries))
    fig,axes=plt.subplots(n,1,figsize=(12,max(3,2*n)),squeeze=False,constrained_layout=True)
    for ax,summary in zip(axes.flat,summaries):
        group=groups[summary["track_id"]]
        begin=min((r["time_s"] for r in group if math.isfinite(r["time_s"])),default=0.)
        span=summary["track_span_s"]
        if math.isfinite(span) and span>0:ax.broken_barh([(begin,span)],(.1,.8),facecolors="#bdbdbd")
        for row in group:
            if row["rest_state"] in ("resting","moving"):
                color="#0077b6" if row["rest_state"]=="resting" else "#ee9b00"
                ax.broken_barh([(row["start_time_s"],row["dt_s"])],(.1,.8),facecolors=color)
        fraction=f"{summary['rest_fraction']:.1%}" if summary["summary_valid"] else "unavailable"
        ax.set(title=f"ID {summary['track_id']} | rest fraction {fraction} | threshold < {summary['threshold_bl_s']:g} BL/s",
               xlabel="Time (s)",yticks=[],ylim=(0,1))
        ax.legend(handles=[Patch(color="#0077b6",label="Resting"),Patch(color="#ee9b00",label="Moving"),Patch(color="#bdbdbd",label="Unknown")],loc="upper right",ncol=3,fontsize=8)
    if not summaries:axes[0,0].text(.5,.5,"No fish observations",ha="center",transform=axes[0,0].transAxes)
    fig.savefig(output/"resting_time.png",dpi=150);plt.close(fig)


def run(input_csv=INPUT_CSV, output_dir=OUTPUT_DIR, *, threshold_bl_s=REST_SPEED_THRESHOLD_BL_S):
    """独立 API：run(result.csv绝对路径, 输出绝对目录, threshold_bl_s=0.05)。

    不调用 speed.py 或读取任何其他行为结果。返回 per-fish summary 和内部
    adjacent intervals。CSV 一行一条鱼，rest_episodes 是每次静止起止与秒数
    的 JSON 数组，无需另存重复明细表。summary_valid=0 的空值不能作为0秒。
    """
    records,meta=load_records(input_csv)
    rows,summaries=analyze_rest(records,threshold_bl_s)
    output=_prepare_output(output_dir,["resting_time_summary.csv","resting_time.png"])
    _write_csv(output/"resting_time_summary.csv",summaries,REST_SUMMARY_FIELDS)
    plot_rest(rows,summaries,output)
    print(f"静止时间分析完成：{len(summaries)} 条鱼，阈值 {threshold_bl_s} BL/s，输出 {output}")
    return {"frames":[_canonical(r) for r in rows],"summary":summaries}


if __name__=="__main__":
    run()
