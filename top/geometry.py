"""顶部视角鱼体几何公共模块：标准数据 API 与分割目录共用同一套算法。

本模块不加载模型、不写文件，也不缓存整批掩膜。图像坐标为 x 向右、y 向下；
centerline 按头到尾排列，head/tail 两根轴均指向头侧。速度或游动方向不参与判头。
"""
from pathlib import Path
import csv
import gzip
import json
import math
import numbers
import re
from collections.abc import Iterable

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
DEFAULT_SEGMENTATION_DIR = Path(r"E:\Phd\temperature\sam3\top\outputs\20261113_122814_121729")
MAX_MASK_PIXELS = 64_000_000  # 解码安全上限，避免损坏尺寸触发意外巨额内存分配。

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


def _integer(value, name, minimum=0):
    """JSON整数和NumPy整数均可，拒绝布尔值、1.0和数字字符串。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} 必须是整数，不能是布尔值、浮点数或字符串。")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}。")
    return value


def _number(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} 必须是有限数字。")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} 必须是{'有限正数' if positive else '有限数字'}。")
    return value


def _decode_compressed_counts(text, pixels):
    """安全解析COCO压缩字符串，不把未经校验的长度交给C扩展。

    COCO按每字符5个数据位保存带符号整数，从第4个run起还要加上前两个run
    的长度。带符号的是差分编码值，恢复后的每个run长度必须非负。
    限制单run编码位数并逐段检查总长度，避免损坏输入造成越界写或巨额分配。
    """
    if isinstance(text, bytes):
        try:
            text = text.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("压缩RLE counts 必须是ASCII字符串。") from exc
    counts, position, total = [], 0, 0
    while position < len(text):
        value, shift = 0, 0
        while True:
            if position >= len(text):
                raise ValueError("压缩RLE字符串在run编码中间截断。")
            c = ord(text[position]) - 48
            position += 1
            if not 0 <= c <= 63:
                raise ValueError("压缩RLE含非法字符。")
            value |= (c & 31) << shift
            shift += 5
            if shift > 60:
                raise ValueError("压缩RLE单个run编码过长。")
            if not c & 32:
                if c & 16:
                    value |= -1 << shift
                break
        if len(counts) > 2:
            value += counts[-2]
        if value < 0:
            raise ValueError("压缩RLE解码出了负run长度。")
        total += value
        if total > pixels:
            raise ValueError("RLE run总长度超过size指定的像素数。")
        counts.append(value)
    if total != pixels:
        raise ValueError(f"RLE run总长度{total}与size像素数{pixels}不一致。")
    return counts


def decode_rle(segmentation):
    """恢复二维bool掩膜，支持标准整数run列表与旧COCO压缩counts。

    segmentation={'size':[height,width], 'counts':...}；按COCO列优先顺序
    （先从上到下，再向右换列）展开，run依次为背景、前景、背景……。
    如果首像素是前景，counts第一个背景run必须是0。所有run必须为非负整数，
    包括零长度run；总和必须恰好等于height*width。size为正整数且不超过
    MAX_MASK_PIXELS。检查完成后才分配掩膜；不使用不安全的原生RLE解码入口。
    """
    if not isinstance(segmentation, dict):
        raise ValueError("segmentation 必须是含size和counts的字典。")
    size = segmentation.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("segmentation.size 必须为 [height, width]。")
    height, width = [_integer(v, "segmentation.size", 1) for v in size]
    pixels = height * width
    if pixels > MAX_MASK_PIXELS:
        raise ValueError(f"掩膜尺寸超过安全上限 {MAX_MASK_PIXELS:,} 像素。")
    counts = segmentation.get("counts")
    if isinstance(counts, (str, bytes)):
        counts = _decode_compressed_counts(counts, pixels)
    elif isinstance(counts, (list, tuple)):
        validated, total = [], 0
        for value in counts:
            run = _integer(value, "segmentation.counts中的run")
            total += run
            if total > pixels:
                raise ValueError("RLE run总长度超过size指定的像素数。")
            validated.append(run)
        if total != pixels:
            raise ValueError(f"RLE run总长度{total}与size像素数{pixels}不一致。")
        counts = validated
    else:
        raise ValueError("segmentation.counts 必须是整数列表或COCO压缩字符串。")
    flat = np.zeros(pixels, dtype=bool)
    position = 0
    for index, run in enumerate(counts):
        if index % 2:
            flat[position:position + run] = True
        position += run
    return flat.reshape((height, width), order="F")


def encode_uncompressed_rle(mask):
    """把二维二值掩膜编码为可直接JSON序列化的COCO整数run列表。

    返回counts列表[背景长度,前景长度,...]，调用者另附size=[height,width]。
    非零值按前景处理；输出采用列优先，首像素为前景时自动补0背景run。
    仅编码，不写文件。与decode_rle互为逆操作。
    """
    array = np.asarray(mask)
    if array.ndim != 2 or min(array.shape) <= 0:
        raise ValueError("encode_uncompressed_rle 需要非空二维掩膜。")
    flat = (array != 0).ravel(order="F")
    changes = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1, flat.size]
    counts = np.diff(changes).astype(np.int64).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return counts


def _validate_identity(row):
    if not isinstance(row, dict):
        raise ValueError("data中的每一条记录必须是字典。")
    if "frame_id" not in row or "track_id" not in row:
        raise ValueError("每条记录必须提供非负整数frame_id和track_id。")
    return _integer(row["frame_id"], "frame_id"), _integer(row["track_id"], "track_id")


def _video_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, numbers.Integral)):
        raise ValueError("video_info.video_id 必须是非空字符串或非负整数。")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("video_info.video_id 不能为空。")
        return value
    return _integer(value, "video_info.video_id")


def _inspect_annotation(row, mask, warnings, frame, identity):
    """验证外部area/bbox的合法性；几何一律使用掩膜实测值。

    不合法数字或越界框是输入错误；合法但与mask不一致是注释偏差，记入meta
    warnings而不覆盖mask实测质心/面积。bbox明确使用COCO[x,y,width,height]。
    """
    height, width = mask.shape
    y, x = np.nonzero(mask)
    if len(x) == 0:
        raise ValueError(f"frame_id={frame}, track_id={identity} 的实例掩膜为空。")
    area = len(x)
    bbox = [float(x.min()), float(y.min()), float(x.max()-x.min()+1), float(y.max()-y.min()+1)]
    if row.get("area") is not None:
        supplied = _number(row["area"], "area")
        if not 0 <= supplied <= height * width:
            raise ValueError("area必须处于0与图像像素总数之间。")
        if not math.isclose(supplied, area, abs_tol=1e-6, rel_tol=0):
            warnings.append(f"frame_id={frame},track_id={identity}: area与掩膜不符，采用掩膜面积{area}。")
    if row.get("bbox") is not None:
        supplied = row["bbox"]
        if not isinstance(supplied, (list, tuple)) or len(supplied) != 4:
            raise ValueError("bbox必须是COCO [x,y,width,height] 四个数字。")
        supplied = [_number(v, "bbox") for v in supplied]
        xx, yy, ww, hh = supplied
        if min(supplied) < 0 or xx + ww > width + 1e-6 or yy + hh > height + 1e-6:
            raise ValueError("bbox必须非负且位于segmentation.size的图像范围内。")
        if not np.allclose(supplied, bbox, atol=1e-6, rtol=0):
            warnings.append(f"frame_id={frame},track_id={identity}: bbox与掩膜不符，采用掩膜外接框{bbox}。")
    return x, y, area, bbox


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
    """唯一几何入口：标准payload -> (逐对象几何records, 全批次meta)。

    payload={'data':[记录,...]}，每记录必须有frame_id、track_id、segmentation。
    ID为非负整数并原样保留，category_id缺省1（鱼），model_info缺省{}，
    video_info缺省{}，view_info缺省'top'且仅支持'top'。video_id缺省'unspecified'。
    帧率必须由参数fps、video_info.fps或目录适配元数据提供，不猜测30fps。
    同一批次必须同一视频、视角、图像尺寸及帧率；同帧时间/路径一致。

    时间：timestamp_seconds要么所有记录都有，要么全部没有，不接受混用。
    有时间戳时同帧一致、跨帧严格递增，time_s减去首帧时间；无时间戳时使用
    (frame_id-first_frame_id)/fps。因此非零起始帧和原始帧号间隙都保留。
    文件适配器还能提供空鱼帧清单，完整保留这些帧的时间和图片路径。

    list允许乱序，会按frame_id/track_id排序；其他可迭代对象必须按帧非降序。
    逐记录解码、即时提取几何，不保存所有掩膜/RLE，适用于逐行文件流。
    返回records按(frame,id)排序，仅含几何及小型元数据；重复(frame,id)拒绝。
    空data返回[]及可用meta，未知宽高为0、未知fps为None，不编造采集信息。
    输入结构/时间/RLE错误抛ValueError；几何不可测则留行并标valid=False。
    """
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError("输入必须是含data的字典：{'data': [...]}。")
    data = payload["data"]
    if isinstance(data, (str, bytes, dict)) or not isinstance(data, Iterable):
        raise ValueError("data必须是记录列表或按帧顺序提供记录的可迭代对象。")
    directory_meta = getattr(data, "meta", None) if isinstance(data, _DirectoryRecords) else None
    requested_fps = _number(fps, "fps参数", positive=True) if fps is not None else None
    if directory_meta is not None:
        source_fps = directory_meta["fps"]
        if requested_fps is not None and not math.isclose(requested_fps, source_fps, rel_tol=1e-9):
            raise ValueError("fps参数与分割目录元数据fps不一致。")
        requested_fps = source_fps
    selected_fps = requested_fps
    if isinstance(data, list):
        # 只复制记录引用并排序，不额外复制/解码RLE。
        data = sorted(data, key=_validate_identity)
        if selected_fps is None:
            for row in data:
                info = row.get("video_info", {})
                if isinstance(info, dict) and info.get("fps") is not None:
                    selected_fps = _number(info["fps"], "video_info.fps", positive=True)
                    break
    hints = HEAD_HINTS if head_hints is None else head_hints
    if not isinstance(hints, dict):
        raise ValueError("head_hints必须是{track_id: left/right/up/down}字典。")
    hints = {_integer(key, "head_hints中的ID"): value for key, value in hints.items()}
    if any(value not in ("left", "right", "up", "down") for value in hints.values()):
        raise ValueError("head_hints的值只能是left/right/up/down。")
    records, warnings, history, seen_ids, seen_keys = [], [], {}, set(), set()
    frame_ids, frame_times, image_paths = [], {}, {}
    width = height = 0
    video_id = "unspecified"
    timestamp_mode, origin_timestamp, previous_timestamp, previous_frame = None, None, None, None
    first_frame = directory_meta["frame_ids"][0] if directory_meta and directory_meta["frame_ids"] else None
    if directory_meta is not None:
        width, height, video_id = directory_meta["width"], directory_meta["height"], directory_meta["video_id"]
        frame_ids = list(directory_meta["frame_ids"])
        frame_times = dict(directory_meta["frame_times"])
        image_paths = dict(directory_meta["image_paths"])
        origin_timestamp = 0.0  # 目录适配器已提供相对整个视频首帧的秒数，包括空鱼帧。
    for row in data:
        frame, identity = _validate_identity(row)
        if previous_frame is not None and frame < previous_frame:
            raise ValueError("流式data必须按frame_id非降序排列；乱序数据请先转为list。")
        if (frame, identity) in seen_keys:
            raise ValueError(f"重复记录：frame_id={frame}, track_id={identity}。")
        seen_keys.add((frame, identity))
        category = _integer(row.get("category_id", 1), "category_id")
        model_info = row.get("model_info", {})
        info = row.get("video_info", {})
        if not isinstance(model_info, dict) or not isinstance(info, dict):
            raise ValueError("model_info和video_info必须是字典。")
        view = row.get("view_info", "top")
        if view != "top":
            raise ValueError("本项目仅支持view_info='top'，不得混入侧视角或其他视角。")
        this_video = _video_id(info.get("video_id", "unspecified"))
        if records or directory_meta is not None:
            if type(this_video) is not type(video_id) or this_video != video_id:
                raise ValueError("同一批data只能包含同一个video_id。")
        else:
            video_id = this_video
        if info.get("fps") is not None:
            supplied_fps = _number(info["fps"], "video_info.fps", positive=True)
            if selected_fps is None:
                selected_fps = supplied_fps
            elif not math.isclose(selected_fps, supplied_fps, rel_tol=1e-9, abs_tol=0):
                raise ValueError("同一批data的fps必须一致，且与fps参数一致。")
        if selected_fps is None:
            raise ValueError("缺少fps。请调用时传fps=实际帧率，或在video_info中提供fps；程序不会猜测30fps。")
        has_timestamp = info.get("timestamp_seconds") is not None
        if timestamp_mode is None:
            timestamp_mode = has_timestamp
        elif timestamp_mode != has_timestamp:
            raise ValueError("timestamp_seconds必须全部提供或全部省略，不能混用。")
        if first_frame is None:
            first_frame = frame
        if has_timestamp:
            timestamp = _number(info["timestamp_seconds"], "video_info.timestamp_seconds")
            if previous_frame == frame:
                if timestamp != previous_timestamp:
                    raise ValueError("同一frame_id中所有对象的timestamp_seconds必须完全一致。")
            elif previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError("不同frame_id的timestamp_seconds必须严格递增。")
            if origin_timestamp is None:
                origin_timestamp = timestamp
            time_s = float(timestamp - origin_timestamp)
            previous_timestamp = timestamp
        else:
            time_s = float((frame - first_frame) / selected_fps)
        image_path = info.get("image_path")
        if image_path is not None:
            if not isinstance(image_path, str) or not Path(image_path).is_absolute():
                raise ValueError("video_info.image_path如提供，必须是绝对路径字符串。")
            if frame in image_paths and image_paths[frame] != image_path:
                raise ValueError("同一frame_id不能对应不同image_path。")
            image_paths[frame] = image_path
        filename = row.get("filename") or (Path(image_path).name if image_path else f"frame_{frame:06d}")
        if not isinstance(filename, str):
            raise ValueError("filename如提供，必须是字符串。")
        segmentation = row.get("segmentation", row.get("rle"))
        mask = decode_rle(segmentation)
        if width == height == 0:
            height, width = mask.shape
        if mask.shape != (height, width):
            raise ValueError("同一批data的所有segmentation.size必须一致。")
        x, y, area, bbox = _inspect_annotation(row, mask, warnings, frame, identity)
        record = _measure_mask(mask, frame, identity, time_s, filename, history, seen_ids, hints, x, y, area)
        record.update(frame_id=frame, track_id=identity, category_id=category, bbox=bbox,
                      model_info=dict(model_info), video_info=dict(info), view_info="top")
        records.append(record)
        if directory_meta is not None:
            if frame not in frame_times or not math.isclose(frame_times[frame], time_s, abs_tol=1e-7, rel_tol=0):
                raise ValueError("标准记录的帧号/时间与frames.csv清单不一致。")
            # 标准记录有自己的采集时刻时保留其原数值；清单只负责时间源核验及空鱼帧。
            frame_times[frame] = time_s
        elif previous_frame != frame:
            frame_ids.append(frame)
            frame_times[frame] = time_s
        previous_frame = frame
        del mask, x, y  # 不保留完整图像掩膜；仅records中的51点中轴进入结果。
    records.sort(key=lambda item: (item["frame"], item["id"]))
    source = ("video_info.timestamp_seconds, relative to first frame" if timestamp_mode else
              "(frame_id - first_frame_id) / fps")
    meta = {"fps": selected_fps, "width": width, "height": height, "frame_count": len(frame_ids),
            "frame_ids": frame_ids, "image_paths": image_paths, "frame_times": frame_times,
            "time_source": directory_meta["time_source"] if directory_meta is not None else source,
            "video_id": video_id, "view_info": "top", "warnings": warnings,
            "head_hints": dict(hints),
            "use_filename_timestamps": directory_meta["use_filename_timestamps"] if directory_meta is not None else None,
            "requested_use_filename_timestamps": directory_meta["requested_use_filename_timestamps"] if directory_meta is not None else None,
            "used_filename_timestamps": directory_meta["used_filename_timestamps"] if directory_meta is not None else None}
    if directory_meta is not None:
        meta["segmentation_dir"] = directory_meta["segmentation_dir"]
    return records, meta


class _DirectoryRecords:
    """具有meta清单的惰性记录流。仅清单常驻内存，RLE逐行读取。

    同一实例可重新迭代，文件每次独立打开。标准文件优先；没有标准文件时
    适配旧native_masks.jsonl.gz。空鱼帧不产生伪实例，由meta完整保留。
    """
    def __init__(self, directory, use_filename_timestamps):
        if not isinstance(use_filename_timestamps, bool):
            raise ValueError("use_filename_timestamps必须是True或False。")
        self.root = Path(directory)
        if not self.root.is_absolute():
            raise ValueError("分割结果目录必须使用绝对路径。")
        info = json.loads((self.root / "run_info.json").read_text(encoding="utf-8-sig"))
        if info.get("status") != "complete":
            raise ValueError(f"只读取完整分割结果，当前status={info.get('status')}。")
        standard = self.root / "standard_data.jsonl.gz"
        self.standard = standard.is_file()
        self.path = standard if self.standard else self.root / "native_masks.jsonl.gz"
        # 标准记录的时间已经由上游确定；这个布尔开关只控制旧native兼容入口。
        # 新标准按run_info记录的原时间源核验，不能因为调用方传False而换一套时钟。
        declared_source = info.get("standard_time_source") if self.standard else None
        if declared_source is not None and declared_source not in (
                "filename _t seconds, relative to first frame", "frame_index / fps"):
            raise ValueError("run_info.standard_time_source不是支持的标准时间源。")
        parse_filename_time = ((declared_source != "frame_index / fps") if self.standard
                               else use_filename_timestamps)
        fps = _number(info.get("fps"), "run_info.fps", positive=True)
        resolution = info.get("resolution")
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            raise ValueError("run_info.resolution必须是[width,height]。")
        width, height = [_integer(v, "run_info.resolution", 1) for v in resolution]
        with (self.root / "frames.csv").open(encoding="utf-8-sig", newline="") as file:
            self.manifest = list(csv.DictReader(file))
        if len(self.manifest) != int(info["frames"]) or int(info.get("completed_frames", -1)) != len(self.manifest):
            raise ValueError("run_info与frames.csv中的帧数不一致。")
        frame_ids = []
        image_paths = {}
        for row in self.manifest:
            text = row.get("frame", row.get("frame_id", ""))
            if not isinstance(text, str) or not text.isdigit():
                raise ValueError("frames.csv帧号必须是非负整数字符串。")
            frame = int(text)
            if frame_ids and frame <= frame_ids[-1]:
                raise ValueError("frames.csv帧号必须严格递增且不能重复。")
            frame_ids.append(frame)
            row["frame"] = frame
            path = row.get("absolute_path") or row.get("image_path")
            if path:
                if not Path(path).is_absolute():
                    raise ValueError("frames.csv中的图片路径必须为绝对路径。")
                image_paths[frame] = path
            row["filename"] = row.get("filename") or (Path(path).name if path else f"frame_{frame:06d}")
        times, source, used_filename_time = [], "frame_index / fps", False
        if parse_filename_time:
            for row in self.manifest:
                match = re.search(r"_t(-?\d+(?:\.\d+)?)", row["filename"])
                if match is None:
                    times = []; break
                times.append(float(match.group(1)))
            if len(times) == len(self.manifest) and np.isfinite(times).all() and np.all(np.diff(times) > 0):
                times = np.asarray(times) - times[0] if times else []
                source = "filename _t seconds, relative to first frame"
                used_filename_time = bool(self.manifest)
            else:
                times = []
                source = "frame_index / fps (filename timestamps missing or non-increasing)"
        if not len(times):
            times = [(frame - frame_ids[0]) / fps for frame in frame_ids]
        if self.standard and declared_source == "filename _t seconds, relative to first frame" and not used_filename_time and self.manifest:
            raise ValueError("标准元数据声明使用文件名采集时间，但frames.csv文件名时间缺失或不递增。")
        video_id = _video_id(info.get("video_id", Path(info.get("image_directory", self.root)).name))
        if info.get("view_info", "top") != "top":
            raise ValueError("该分割目录不是top顶部视角。")
        self.meta = {"fps": fps, "width": width, "height": height, "frame_count": len(frame_ids),
                     "frame_ids": frame_ids, "image_paths": image_paths,
                     "frame_times": dict(zip(frame_ids, map(float, times))), "time_source": source,
                     "video_id": video_id, "view_info": "top", "segmentation_dir": str(self.root),
                     "use_filename_timestamps": None if self.standard else use_filename_timestamps,
                     "requested_use_filename_timestamps": use_filename_timestamps,
                     "used_filename_timestamps": used_filename_time}
        self.info = info

    def __iter__(self):
        line_count, supplied_time_mode = 0, None
        with gzip.open(self.path, "rt", encoding="utf-8") as file:
            for index, line in enumerate(file):
                if index >= len(self.manifest):
                    raise ValueError("掩膜文件含有超出frames.csv清单的多余帧。")
                data = json.loads(line)
                manifest = self.manifest[index]
                frame, filename = manifest["frame"], manifest["filename"]
                time_s = self.meta["frame_times"][frame]
                path = self.meta["image_paths"].get(frame)
                if self.standard:
                    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
                        raise ValueError("standard_data.jsonl.gz每行必须是{'data':[该帧实例]}。")
                    objects = data["data"]
                else:
                    if int(data["frame"]) != frame or data["filename"] != filename:
                        raise ValueError("旧掩膜文件帧号/文件名与frames.csv不一致。")
                    objects = data["objects"]
                    if not isinstance(objects, list):
                        raise ValueError("旧掩膜objects必须是列表。")
                for obj in objects:
                    if not isinstance(obj, dict):
                        raise ValueError("每个掩膜对象必须是字典。")
                    if self.standard:
                        item = dict(obj)
                        this_frame, _ = _validate_identity(item)
                        if this_frame != frame:
                            raise ValueError("standard_data行中的frame_id与frames.csv对应帧不一致。")
                        video = item.get("video_info", {})
                        if not isinstance(video, dict):
                            raise ValueError("video_info必须是字典。")
                        video = dict(video)
                        has_time = video.get("timestamp_seconds") is not None
                        if supplied_time_mode is None:
                            supplied_time_mode = has_time
                        elif supplied_time_mode != has_time:
                            raise ValueError("标准文件中的timestamp_seconds不能部分有、部分无。")
                        if has_time:
                            value = _number(video["timestamp_seconds"], "timestamp_seconds")
                            if not math.isclose(value, time_s, abs_tol=1e-7, rel_tol=0):
                                raise ValueError("标准文件时间戳与frames.csv按当前时间设置得到的相对秒数不一致。")
                        video.setdefault("video_id", self.meta["video_id"])
                        video.setdefault("fps", self.meta["fps"])
                        if path is not None:
                            video.setdefault("image_path", path)
                    else:
                        item = {"track_id": obj["id"], "frame_id": frame, "category_id": 1,
                                "segmentation": obj["rle"], "bbox": obj.get("bbox_xywh"),
                                "model_info": self.info.get("model_info", {"name": "SAM3"}), "view_info": "top"}
                        if obj.get("area") is not None:
                            item["area"] = obj["area"]
                        video = {"video_id": self.meta["video_id"], "fps": self.meta["fps"]}
                        if path is not None:
                            video["image_path"] = path
                    # 清单统一决定相对时间，完整保留开头/中间无鱼帧的时间基准。
                    if not self.standard or video.get("timestamp_seconds") is None:
                        video["timestamp_seconds"] = time_s
                    item["video_info"] = video
                    item.setdefault("filename", filename)
                    yield item
                line_count += 1
        if line_count != len(self.manifest):
            raise ValueError(f"掩膜文件帧数不完整：{line_count}/{len(self.manifest)}。")


def iter_standard_records(segmentation_dir=DEFAULT_SEGMENTATION_DIR, use_filename_timestamps=True):
    """把完整分割目录转成标准记录流，返回值另带只读使用的.meta清单。

    每帧占一行的standard_data.jsonl.gz优先；旧native_masks逐行适配成同一字段。
    旧native先读取全部frames.csv判定文件名时间是否全部有效递增，否则整段回退；
    与原150帧流程相同。新标准优先保留记录时间，按run_info.standard_time_source
    核验；use_filename_timestamps仅用于旧native，不覆盖新标准采集时钟。
    新标准meta.use_filename_timestamps为None（不适用），另记requested和used。
    标准时间相对首帧；记录时间须与原时间源相符，不悄悄改写时间尺度。
    输出counts可保持旧压缩字符串；如需导出整数列表，可调用decode_rle再encode。
    """
    return _DirectoryRecords(segmentation_dir, use_filename_timestamps)


def load_geometry(segmentation_dir=DEFAULT_SEGMENTATION_DIR, head_hints=None, use_filename_timestamps=True):
    """文件入口：只负责读取/适配，全部实际几何仍由geometry_from_data计算。"""
    stream = iter_standard_records(segmentation_dir, use_filename_timestamps)
    return geometry_from_data({"data": stream}, head_hints=head_hints)
