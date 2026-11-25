# 顶部视角维护负责人：YHC。此脚本仅属于 FishBehavior/Top，不依赖 Side 子项目。
"""从 frames.csv 提取鱼体基础几何（独立运行，第 2 步）。

本文件不加载 SAM3，不调用任何其他项目脚本。解码完整掩膜后提取质心、中轴、
头尾方向、摆尾角及体长；保持原项目数值算法和默认参数。完成后写 result.csv，
全部 Behavior 脚本只读取这张结果表。质量失败保留该鱼及原因，不能伪造有效零值。
"""
from pathlib import Path
import csv
import json
import math
import statistics
from collections import defaultdict
import cv2
import numpy as np
from pycocotools import mask as coco
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.signal import savgol_filter

# ======================= 日常修改区（绝对路径） ======================
INPUT_CSV = Path(r"E:\Phd\FishBehavior\Top\Tracker\Output\Information\frames.csv")
OUTPUT_DIR = Path(r"E:\Phd\FishBehavior\Top\Tracker\Output\Information")
HEAD_HINTS = {}                  # 例如 {0:"left"}，仅初始化首次成功判头；ID=0 合法。
CENTERLINE_POINTS = 51           # 保持原来 51 点中轴、两端延伸及空间平滑规则。
MIN_MASK_AREA = 100
MIN_COMPONENT_FRACTION = 0.90
MIN_HEAD_WIDTH_CONTRAST = 0.18
HEAD_AXIS_RANGE = (0.10, 0.48)
TAIL_AXIS_RANGE = (0.65, 0.95)
MAX_MASK_PIXELS = 64_000_000      # 防止错误尺寸触发过量内存分配。
# ===================================================================
FRAME_FIELDS = "schema_version,run_id,record_type,video_id,frame_id,time_s,fps,time_source,image_path,image_width,image_height,track_id,category_id,score,area_px,bbox_x,bbox_y,bbox_width,bbox_height,mask_height,mask_width,mask_rle".split(",")
RESULT_FIELDS = "schema_version,run_id,record_type,video_id,frame_id,time_s,fps,time_source,image_path,image_width,image_height,track_id,category_id,score,cx,cy,area_px,bbox_x,bbox_y,bbox_width,bbox_height,centerline,head_dx,head_dy,tail_dx,tail_dy,heading_deg,tail_angle_deg,body_length_px,reference_body_length_px,reference_length_sample_count,position_valid,geometry_valid,head_valid,head_confidence,quality_reason".split(",")

def zhang_suen(binary):
    """8邻域Zhang–Suen细化，把实心轮廓缩成单像素骨架，不依赖skimage。

    一轮分两个子阶段：前景邻居数需2~6，环形邻域0->1转换恰好一次，
    再检查各子阶段的三邻居乘积条件，防止同时删断必要连接。迭代至稳定。
    输入是局部二值轮廓，函数再补一圈零边界；返回与输入同尺寸bool骨架。
    """
    a = np.pad(binary.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p = [a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:], a[2:, 2:],
                 a[2:, 1:-1], a[2:, :-2], a[1:-1, :-2], a[:-2, :-2]]
            n = sum(p)
            transitions = sum(((p[i] == 0) & (p[(i + 1) % 8] == 1)).astype(np.uint8)
                              for i in range(8))
            remove = (a[1:-1, 1:-1] == 1) & (n >= 2) & (n <= 6) & (transitions == 1)
            if step == 0:
                remove &= (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                remove &= (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            if remove.any():
                a[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            return a[1:-1, 1:-1].astype(bool)

def skeleton_longest_path(skeleton):
    """用加权骨架图的最长端点路径保留主干，剪去鳍和毛刺分支。

    每个骨架像素是节点，横竖相连权重1，斜向权重sqrt(2)，度数1为端点。
    求端点两两最短路，选距离最大的那一对并回溯其路径；不是可绕圈的
    一般最长路径问题。骨架不足5点、端点少于2/超过24或回溯失败则拒绝。
    第二返回值是未进入主路径的骨架点比例，>35%仅由后续流程附加警告。
    """
    y, x = np.nonzero(skeleton)
    if len(x) < 5:
        raise ValueError("skeleton_too_short")
    index = np.full(skeleton.shape, -1, dtype=np.int32)
    index[y, x] = np.arange(len(x))
    edges_a, edges_b, weights = [], [], []
    h, w = skeleton.shape
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        yy, xx = y + dy, x + dx
        inside = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        a = np.flatnonzero(inside)
        b = index[yy[inside], xx[inside]]
        keep = b >= 0
        a, b = a[keep], b[keep]
        edges_a.extend([a, b]); edges_b.extend([b, a])
        weights.extend([np.full(len(a), math.hypot(dx, dy))] * 2)
    graph = csr_matrix((np.concatenate(weights),
                        (np.concatenate(edges_a), np.concatenate(edges_b))),
                       shape=(len(x), len(x)))
    degree = np.diff(graph.indptr)
    ends = np.flatnonzero(degree == 1)
    if len(ends) < 2:
        raise ValueError("skeleton_no_two_endpoints")
    # 极碎或高度分叉的轮廓通常来自遮挡/错误分割，不把任意曲线当成鱼中轴。
    if len(ends) > 24:
        raise ValueError("skeleton_excessive_branches")
    dist, predecessors = dijkstra(graph, directed=False, indices=ends, return_predecessors=True)
    pairs = dist[:, ends]
    pairs[~np.isfinite(pairs)] = -1
    start_row, end_col = np.unravel_index(np.argmax(pairs), pairs.shape)
    current, start = int(ends[end_col]), int(ends[start_row])
    path = [current]
    while current != start:
        current = int(predecessors[start_row, current])
        if current < 0 or len(path) > len(x):
            raise ValueError("skeleton_path_failure")
        path.append(current)
    xy = np.column_stack((x[path], y[path])).astype(np.float64)
    return xy, 1 - len(path) / len(x)

def resample_curve(points, count=CENTERLINE_POINTS):
    """按累计弧长等距插值51点，再做7点2阶空间Savitzky–Golay平滑。

    零长度重复点先去掉，长度<15像素或有效点<5则拒绝；平滑后恢复原两端。
    最终点距会受平滑轻微改变，因此是先等弧长采样再平滑，不宣称最终严格等距。
    """
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-8]
    if distance[-1] < 15 or keep.sum() < 5:
        raise ValueError("centerline_too_short")
    target = np.linspace(0, distance[-1], count)
    result = np.column_stack([np.interp(target, distance[keep], points[keep, dim]) for dim in (0, 1)])
    result = savgol_filter(result, 7, 2, axis=0, mode="interp")
    result[0], result[-1] = points[0], points[-1]
    return result

def axis_toward_head(centerline, fractions):
    """对指定中轴段做PCA/SVD直线拟合，并把单位轴向设为尾侧->头侧。

    centerline必须是头->尾点序；fraction乘(N-1)后round，首尾索引均包含。
    默认头段0.10~0.48对应5~24点，尾段0.65~0.95对应32~48点（51点时）。
    两根轴同向定义使直鱼夹角为0，避免因箭头反向产生180度伪角。
    """
    start, stop = (int(round(v * (len(centerline) - 1))) for v in fractions)
    points = centerline[start:stop + 1]
    _, _, vectors = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    axis = vectors[0]
    if np.dot(axis, points[0] - points[-1]) < 0:
        axis = -axis
    return axis / max(np.linalg.norm(axis), 1e-12)

def mask_centerline(mask):
    """完整掩膜->局部主体->小孔修补->骨架主路径->边界延伸->51点中轴。

    主体采用8邻接最大区域，至少占原mask90%；仅填面积<=max(16,主体0.5%)
    的小孔。欧氏距离的2倍近似局部体宽；末端沿内侧8点方向每步0.5像素
    延伸，最多搜索3倍局部半径+3像素。>8%中轴点在外或长宽比<3时拒绝。
    第7:21与30:44点体宽均值之差/较大值作为初始头尾证据。坐标最终回原图。
    注意主体修补不改外部记录的原mask质心/面积，返回的length是平滑折线长。
    """
    y, x = np.nonzero(mask)
    if len(x) < MIN_MASK_AREA:
        raise ValueError("mask_too_small")
    x0, y0, x1, y1 = x.min(), y.min(), x.max() + 1, y.max() + 1
    roi = np.pad(mask[y0:y1, x0:x1].astype(np.uint8), 3)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    selected = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    fraction = stats[selected, cv2.CC_STAT_AREA] / len(x)
    if fraction < MIN_COMPONENT_FRACTION:
        raise ValueError("fragmented_mask")
    clean = labels == selected
    # 只修小孔，不填大洞或改变两条鱼之间的拓扑。
    holes = ndimage.binary_fill_holes(clean) & ~clean
    hole_labels, _ = ndimage.label(holes)
    hole_sizes = np.bincount(hole_labels.ravel())
    repair = (hole_labels != 0) & (hole_sizes[hole_labels] <= max(16, int(clean.sum() * .005)))
    clean |= repair
    radius = cv2.distanceTransform(clean.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    raw, branch_fraction = skeleton_longest_path(zhang_suen(clean))
    # 由骨架末端沿局部切线延伸到掩膜边界，补齐被细化截短的鼻尖/尾尖。
    for reverse in (False, True):
        seq = raw[::-1] if reverse else raw
        end = seq[0].copy()
        span = min(8, len(seq) - 1)
        direction = end - seq[span]
        direction /= max(np.linalg.norm(direction), 1e-12)
        end_radius = radius[int(round(end[1])), int(round(end[0]))]
        extension = end.copy()
        for step in np.arange(.5, max(3.0, 3 * end_radius + 3), .5):
            point = end + direction * step
            xx, yy = np.rint(point).astype(int)
            if not (0 <= yy < clean.shape[0] and 0 <= xx < clean.shape[1] and clean[yy, xx]):
                break
            extension = point
        if reverse:
            raw = np.vstack((raw, extension))
        else:
            raw = np.vstack((extension, raw))
    curve = resample_curve(raw)
    samples = np.rint(curve).astype(int)
    samples[:, 0] = np.clip(samples[:, 0], 0, clean.shape[1] - 1)
    samples[:, 1] = np.clip(samples[:, 1], 0, clean.shape[0] - 1)
    if np.mean(~clean[samples[:, 1], samples[:, 0]]) > .08:
        raise ValueError("centerline_outside_mask")
    widths = 2 * radius[samples[:, 1], samples[:, 0]]
    a, b = np.mean(widths[7:21]), np.mean(widths[30:44])
    contrast = (a - b) / max(a, b, 1e-6)
    length = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
    if length < 3 * max(widths):
        raise ValueError("shape_not_elongated")
    curve += [x0 - 3, y0 - 3]
    return curve, contrast, branch_fraction, length

def _orient(curve, contrast, previous, hint, time_s):
    """用体宽初始化、同ID最近有效朝向延续方向，不依据鱼的移动方向。

    0.35秒内选择与历史最接近的头向，超过70度或强宽度冲突则拒绝。
    人工锚定的历史只警告宽度冲突；长时间失去历史后不擅自反转/重置。
    返回头到尾点序、启发式评分、质量来源；评分不是准确率概率。
    """
    confidence = abs(float(contrast))
    inferred = curve if contrast >= 0 else curve[::-1].copy()
    source = "width"
    if previous is not None and time_s - previous["time_s"] <= .35:
        candidates = (curve, curve[::-1].copy())
        dots = [float(np.dot(axis_toward_head(c, HEAD_AXIS_RANGE), previous["axis"])) for c in candidates]
        selected = int(np.argmax(dots))
        if dots[selected] < math.cos(math.radians(70)):
            raise ValueError("heading_discontinuity")
        inferred = candidates[selected]
        evidence_agrees = (selected == 0) == (contrast >= 0)
        manual = previous.get("manual", False)
        if not evidence_agrees and confidence > .55 and not manual:
            raise ValueError("head_width_temporal_conflict")
        confidence = max(confidence if evidence_agrees else 0,
                         previous["confidence"] * .98)
        source = "temporal" if not evidence_agrees or abs(contrast) < MIN_HEAD_WIDTH_CONTRAST else "width_temporal"
        if manual:
            source = "manual_temporal" + (";width_conflict" if not evidence_agrees else "")
    elif previous is not None and previous.get("manual", False):
        # 初帧人工方向不能跨长遮挡直接套用；也不能悄悄被宽度启发式改写。
        raise ValueError("manual_orientation_lost_after_gap")
    elif hint is not None:
        vectors = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
        if hint not in vectors:
            raise ValueError(f"HEAD_HINTS 只能填 left/right/up/down，当前是 {hint!r}")
        delta = curve[0] - curve[-1]
        if np.dot(delta, vectors[hint]) < 0:
            inferred = curve[::-1].copy()
        else:
            inferred = curve
        confidence, source = 1.0, "manual_hint"
    elif confidence < MIN_HEAD_WIDTH_CONTRAST:
        raise ValueError("head_tail_ambiguous")
    return inferred, confidence, source

def _measure_mask(mask, frame, identity, time_s, filename, history, seen_ids, hints, x, y, area):
    """单实例几何，与已验证的原150帧算法保持相同运算顺序。

    cx/cy/area来自未修补的完整掩膜；主连通域、小孔修补、骨架等仅影响中轴。
    失败仍保留质心，valid=False且角度/轴向为NaN；quality_reason保留具体原因。
    历史只在全部检查通过时更新；手动头向只在该ID首次成功定向时使用。
    """
    height, width = mask.shape
    record = {"frame": frame, "time_s": time_s, "id": identity,
              "cx": float(x.mean()), "cy": float(y.mean()), "area": area,
              "centerline": None, "head_dx": float("nan"), "head_dy": float("nan"),
              "heading_deg": float("nan"), "tail_angle_deg": float("nan"),
              "valid": False, "quality_reason": "", "head_confidence": 0.0,
              "filename": filename, "body_length_px": float("nan"),
              "tail_dx": float("nan"), "tail_dy": float("nan")}
    try:
        if x.min() == 0 or x.max() == width - 1 or y.min() == 0 or y.max() == height - 1:
            raise ValueError("mask_touches_image_boundary")
        curve, contrast, branches, length = mask_centerline(mask)
        previous = history.get(identity)
        hint = hints.get(identity) if identity not in seen_ids else None
        curve, confidence, source = _orient(curve, contrast, previous, hint, time_s)
        if previous is not None and time_s - previous["time_s"] <= .1:
            ratio = area / previous["area"]
            if not .55 <= ratio <= 1.65:
                raise ValueError("abrupt_mask_area_change")
        head = axis_toward_head(curve, HEAD_AXIS_RANGE)
        tail = axis_toward_head(curve, TAIL_AXIS_RANGE)
        angle = math.degrees(math.atan2(-(head[0] * tail[1] - head[1] * tail[0]), np.dot(head, tail)))
        record.update(centerline=curve, head_dx=float(head[0]), head_dy=float(head[1]),
                      tail_dx=float(tail[0]), tail_dy=float(tail[1]),
                      heading_deg=math.degrees(math.atan2(-head[1], head[0])),
                      tail_angle_deg=angle, valid=True, head_confidence=confidence,
                      quality_reason=source + (";branched_outline" if branches > .35 else ""),
                      body_length_px=length)
        history[identity] = {"time_s": time_s, "axis": head, "confidence": confidence,
                             "area": area, "manual": source.startswith("manual")}
        seen_ids.add(identity)
    except ValueError as exc:
        record["quality_reason"] = str(exc)
    return record


def _finite(value, label, *, positive=False):
    """输入格式校验：空值/NaN/Infinity 不能充当有效基础信息。"""
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} 必须是有限数值，得到 {value!r}") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{label} 必须是{'正的' if positive else ''}有限数值。")
    return number


def _integer(value, label, minimum=0):
    number = _finite(value, label)
    if number != int(number) or number < minimum:
        raise ValueError(f"{label} 必须是 >= {minimum} 的整数。")
    return int(number)


def _validate_rle(text, pixels):
    """先用 Python 检查压缩游程总数，再交给 C 扩展解码，防止损坏 RLE 越界。

    COCO counts 每字符 5 个数据位，带符号差分从第 4 个 run 开始引用前两段；
    背景从第一段开始，所有还原长度非负，和必须恰好为 H*W。不更改掩膜像素。
    """
    if not isinstance(text, str) or not text:
        raise ValueError("observation 必须包含非空 mask_rle。")
    counts, position, total = [], 0, 0
    while position < len(text):
        value, shift = 0, 0
        while True:
            if position >= len(text):
                raise ValueError("RLE 编码中途截断。")
            c = ord(text[position]) - 48
            position += 1
            if not 0 <= c <= 63:
                raise ValueError("RLE 包含非法字符。")
            value |= (c & 31) << shift
            shift += 5
            if shift > 60:
                raise ValueError("RLE 单段编码过长。")
            if not c & 32:
                if c & 16:
                    value |= -1 << shift
                break
        if len(counts) > 2:
            value += counts[-2]
        if value < 0:
            raise ValueError("RLE 还原长度为负数。")
        total += value
        if total > pixels:
            raise ValueError("RLE 超出掩膜尺寸。")
        counts.append(value)
    if total != pixels:
        raise ValueError(f"RLE 像素总数 {total} 与尺寸 {pixels} 不一致。")


def load_geometry(input_csv=INPUT_CSV, head_hints=None):
    """流式解码 frames.csv，返回 (结果行列表, 元信息)，不写文件、不读原图。

    input_csv：Tracker/sam3.py 完整生成的绝对路径；head_hints：可选 {鱼ID:方向}。
    只同时持有当前实例的全分辨率掩膜，长期保存的是小型 51 点几何记录。
    帧清单必须顺序完整，同一帧可以多条鱼；没有鱼必须有 empty_frame 行。
    所有行共享视频、run_id、尺寸、fps/time_source；同帧时间和路径完全一致。
    严格校验损坏的 CSV/RLE；中轴测量失败则保留质心并标记几何无效。
    """
    path = Path(input_csv)
    if not path.is_absolute():
        raise ValueError("input_csv 必须是绝对路径。")
    if not path.is_file():
        raise FileNotFoundError(f"未找到 {path}，请先运行 sam3.py。")
    hints = dict(HEAD_HINTS if head_hints is None else head_hints)
    hints = {_integer(key, "head_hints ID"): value for key, value in hints.items()}
    if any(v not in ("left", "right", "up", "down") for v in hints.values()):
        raise ValueError("头向提示只允许 left/right/up/down。")
    csv.field_size_limit(64 * 1024 * 1024)
    history, seen_ids = {}, set()
    rows, frame_meta, identity_keys, lengths = [], {}, set(), defaultdict(list)
    previous_frame, signature = None, None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        table = csv.DictReader(stream)
        if table.fieldnames is None or len(table.fieldnames) != len(set(table.fieldnames)):
            raise ValueError("CSV 表头为空或包含重复列名。")
        missing = set(FRAME_FIELDS) - set(table.fieldnames)
        if missing:
            raise ValueError(f"frames.csv 缺少字段：{sorted(missing)}")
        for line, source in enumerate(table, 2):
            if None in source or any(v is None for v in source.values()):
                raise ValueError(f"第 {line} 行列数不符合表头。")
            if _integer(source["schema_version"], "schema_version", 1) != 1:
                raise ValueError("不支持的 schema_version。")
            frame = _integer(source["frame_id"], "frame_id")
            time_s = _finite(source["time_s"], "time_s")
            fps = _finite(source["fps"], "fps", positive=True)
            width = _integer(source["image_width"], "image_width", 1)
            height = _integer(source["image_height"], "image_height", 1)
            if time_s < 0 or width * height > MAX_MASK_PIXELS:
                raise ValueError("相对时间为负或图像尺寸过大。")
            image_path = Path(source["image_path"])
            if not image_path.is_absolute():
                raise ValueError("image_path 必须是原图绝对路径。")
            if not source["run_id"] or not source["video_id"] or not source["time_source"]:
                raise ValueError("run_id、video_id 和 time_source 不得为空。")
            current_signature = (source["run_id"], source["video_id"], fps, width, height, source["time_source"])
            if signature is None:
                signature = current_signature
            elif signature != current_signature:
                raise ValueError("不能混合不同视频、运行批次、尺寸或帧率。")
            if previous_frame is not None and frame < previous_frame:
                raise ValueError("frames.csv 必须按 frame_id 排序。")
            if frame not in frame_meta:
                if previous_frame is not None:
                    if frame != previous_frame + 1:
                        raise ValueError("完整帧清单有缺口；无鱼帧应写 empty_frame。")
                    if time_s <= frame_meta[previous_frame]["time_s"]:
                        raise ValueError("跨帧 time_s 必须严格递增。")
                frame_meta[frame] = {"time_s": time_s, "image_path": str(image_path), "empty": False, "observations": 0}
            state = frame_meta[frame]
            if state["time_s"] != time_s or state["image_path"] != str(image_path):
                raise ValueError("同一帧的时间或图片路径不一致。")
            previous_frame = frame
            row = {key: source[key] for key in RESULT_FIELDS if key in source}
            row.update(schema_version=1, frame_id=frame, time_s=time_s, fps=fps,
                       image_width=width, image_height=height,
                       position_valid=0, geometry_valid=0, head_valid=0,
                       reference_length_sample_count=0, head_confidence=0.0)
            kind = source["record_type"]
            if kind == "empty_frame":
                if state["empty"] or state["observations"] or any(source[k] for k in
                        ("track_id", "mask_rle", "mask_height", "mask_width", "area_px")):
                    raise ValueError("empty_frame 必须唯一且没有实例/掩膜信息。")
                state["empty"] = True
                row.update(quality_reason="no_detections")
                rows.append(row)
                continue
            if kind != "observation" or state["empty"]:
                raise ValueError("record_type 不合法或空帧混入实例。")
            identity = _integer(source["track_id"], "track_id")
            key = frame, identity
            if key in identity_keys:
                raise ValueError(f"重复的 (frame_id,track_id)={key}")
            identity_keys.add(key)
            state["observations"] += 1
            category = _integer(source["category_id"], "category_id")
            score = _finite(source["score"], "score")
            if not 0 <= score <= 1:
                raise ValueError("score 应在 [0,1] 内。")
            mh = _integer(source["mask_height"], "mask_height", 1)
            mw = _integer(source["mask_width"], "mask_width", 1)
            if (mh, mw) != (height, width):
                raise ValueError("掩膜必须为原图全尺寸，不能缺少裁剪偏移。")
            _validate_rle(source["mask_rle"], height * width)
            mask = coco.decode({"size": [height, width], "counts": source["mask_rle"].encode("ascii")}).astype(bool)
            y, x = np.nonzero(mask)
            if not len(x):
                raise ValueError("observation 掩膜为空；应过滤此对象或使用 empty_frame。")
            area = int(len(x))
            bbox = [float(x.min()), float(y.min()), float(x.max() - x.min() + 1), float(y.max() - y.min() + 1)]
            supplied_area = _finite(source["area_px"], "area_px", positive=True)
            supplied_bbox = [_finite(source[k], k) for k in ("bbox_x", "bbox_y", "bbox_width", "bbox_height")]
            if not math.isclose(supplied_area, area, abs_tol=1e-6, rel_tol=0) or not np.allclose(supplied_bbox, bbox, atol=1e-6, rtol=0):
                raise ValueError(f"frame={frame}, ID={identity}: 注释面积/外接框与实际掩膜不一致。")
            measurement = _measure_mask(mask, frame, identity, time_s, image_path.name,
                                        history, seen_ids, hints, x, y, area)
            curve = measurement["centerline"]
            row.update(track_id=identity, category_id=category, score=score,
                       cx=measurement["cx"], cy=measurement["cy"], area_px=area,
                       bbox_x=bbox[0], bbox_y=bbox[1], bbox_width=bbox[2], bbox_height=bbox[3],
                       centerline=json.dumps(curve.tolist(), separators=(",", ":")) if curve is not None else "",
                       position_valid=1, geometry_valid=int(measurement["valid"]),
                       head_valid=int(measurement["valid"]),
                       head_confidence=measurement["head_confidence"], quality_reason=measurement["quality_reason"])
            for name in ("head_dx", "head_dy", "tail_dx", "tail_dy", "heading_deg", "tail_angle_deg", "body_length_px"):
                row[name] = measurement[name]
            if row["geometry_valid"] and math.isfinite(row["body_length_px"]) and row["body_length_px"] > 0:
                lengths[identity].append(row["body_length_px"])
            rows.append(row)
    if not frame_meta:
        raise ValueError("frames.csv 没有帧记录，无法建立视频时间轴。")
    references = {identity: statistics.median(values) for identity, values in lengths.items()}
    for row in rows:
        if row["record_type"] == "observation":
            identity = row["track_id"]
            row["reference_body_length_px"] = references.get(identity, math.nan)
            row["reference_length_sample_count"] = len(lengths.get(identity, []))
    meta = {"frame_count": len(frame_meta), "observations": len(identity_keys), "fps": signature[2],
            "image_width": signature[3], "image_height": signature[4], "run_id": signature[0],
            "video_id": signature[1], "time_source": signature[5], "head_hints": hints}
    return rows, meta


def run(input_csv=INPUT_CSV, output_dir=OUTPUT_DIR, *, head_hints=None):
    r"""Python 调用：基础分割表 -> result.csv，返回完成文件的绝对 Path。

    示例（将本脚本作为独立模块加载后）：
        geometry.run(input_csv=r"E:\Phd\FishBehavior\Top\Tracker\Output\Information\frames.csv",
                     output_dir=r"E:\Phd\FishBehavior\Top\Tracker\Output\Information",
                     head_hints={0:"left"})
    input_csv：分割跟踪表；output_dir：固定输出目录；head_hints：可选首次头向锚点。
    CSV 无效数值为空；几何失败不删除个体，其 position_valid 仍可为 1；因此
    速度/聚集等基于质心的指标不会因单纯判头失败而丢失可用位置。
    reference_body_length_px 是每条鱼整批有效中轴弧长中位数；无样本则空且计数 0。
    centerline 是头到尾的 51 点 JSON 数组，轴线单位向量、角度 degree、长度 pixel。
    本函数原子发布 result.csv，不更改 frames.csv，不自动运行行为分析。
    """
    output = Path(output_dir)
    if not output.is_absolute():
        raise ValueError("output_dir 必须为绝对路径。")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "result.csv"
    if destination.resolve() == Path(input_csv).resolve():
        raise ValueError("输入路径不能与输出 result.csv 相同。")
    # 重算失败时不能让旧几何表看起来仍是本次输出。
    destination.unlink(missing_ok=True)
    temporary = output / ".result.pending.csv"
    temporary.unlink(missing_ok=True)
    try:
        rows, meta = load_geometry(input_csv, head_hints=head_hints)
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: "" if isinstance(value, (float, np.floating)) and not math.isfinite(value) else value
                                 for key, value in row.items()})
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"完成几何提取：{meta['frame_count']} 帧、{meta['observations']} 条个体观测。\n{destination}", flush=True)
    return destination


if __name__ == "__main__":
    run()
