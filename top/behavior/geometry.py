"""顶部视角鱼体几何公共模块：标准数据 API 与分割目录共用同一套算法。

本模块不加载模型、不写文件，也不缓存整批掩膜。图像坐标为 x 向右、y 向下；
centerline 按头到尾排列，head/tail 两根轴均指向头侧。速度或游动方向不参与判头。
"""
from pathlib import Path
# 支持绝对路径直接运行本文件；正常包导入时无需修改模块搜索路径。
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_format import DEFAULT_SEGMENTATION_DIR, _integer, iter_standard_records, map_standard_data

import math

import cv2
import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.signal import savgol_filter

# 与参考150帧算法保持一致；这些参数改变会影响中轴、头向和摆尾角。
CENTERLINE_POINTS = 51
MIN_MASK_AREA = 100
MIN_COMPONENT_FRACTION = 0.90
MIN_HEAD_WIDTH_CONTRAST = 0.18
HEAD_AXIS_RANGE = (0.10, 0.48)
TAIL_AXIS_RANGE = (0.65, 0.95)
HEAD_HINTS = {}

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


def geometry_from_data(payload, head_hints=None, fps=None):
    """几何入口：标准输入 -> (几何records,meta)，格式统一交给data_format校验。

    head_hints为{鱼ID: left/right/up/down}，仅初始化首次成功判头；
    fps为缺省帧率补充。中轴线测量、头向历史及质量规则全部在本模块维护。
    每条测量记录保存质心、51点中轴、头尾轴、体长及有效性，不保存整幅掩膜。
    函数每次创建独立历史，不跨视频或跨调用沿用鱼的状态。
    """
    hints = HEAD_HINTS if head_hints is None else head_hints
    if not isinstance(hints, dict):
        raise ValueError("head_hints必须是{track_id: left/right/up/down}字典。")
    hints = {_integer(key, "head_hints中的ID"): value for key, value in hints.items()}
    if any(value not in ("left", "right", "up", "down") for value in hints.values()):
        raise ValueError("head_hints的值只能是left/right/up/down。")
    history, seen_ids = {}, set()
    def measure(mask, frame, identity, time_s, filename, x, y, area):
        return _measure_mask(mask, frame, identity, time_s, filename,
                             history, seen_ids, hints, x, y, area)
    records, meta = map_standard_data(payload, measure, fps=fps)
    meta["head_hints"] = dict(hints)
    return records, meta


def load_geometry(segmentation_dir=DEFAULT_SEGMENTATION_DIR, head_hints=None, use_filename_timestamps=True):
    """文件入口：只负责读取/适配，全部实际几何仍由geometry_from_data计算。"""
    stream = iter_standard_records(segmentation_dir, use_filename_timestamps)
    return geometry_from_data({"data": stream}, head_hints=head_hints)
