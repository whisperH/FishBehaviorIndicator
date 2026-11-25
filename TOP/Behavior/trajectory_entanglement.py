# 顶部视角维护负责人：YHC。此脚本仅属于 FishBehavior/Top，不依赖 Side 子项目。
"""个体二维轨迹近交分析：独立读取 result.csv，不调用其他项目脚本。

本指标描述接近时间内的二维轨迹接近，不具备深度信息，不能证明身体碰撞。
允许近距离并行/静止接近；连续重复候选合并为一次持续事件，不宣称绕结拓扑数。
按鱼对检测事件，再计入两个参与鱼的个体汇总。主事件率单位：次/1000有效鱼对帧。
"""
from pathlib import Path

# ==================== 可修改参数和绝对路径 ====================
INPUT_CSV = Path(r"E:\Phd\FishBehavior\Top\Tracker\Output\Information\result.csv")
OUTPUT_DIR = Path(r"E:\Phd\FishBehavior\Top\Behavior\Output\trajectory_entanglement")
WINDOW_FRAMES = 8             # 每窗8个样本，首末跨7个时间间隔；窗口每帧向前移动1帧。
TIME_TOLERANCE_FRAMES = 2.0    # 先除以fps转秒；不同帧率比较时应设成相同物理时间尺度。
DISTANCE_THRESHOLD = 0.5      # 宽高归一化距离，无单位；建议结合标记视频核对适用性。
MERGE_GAP_FRAMES = 8.0        # 同鱼对同连续段候选时刻相邻差<=8/fps则合并。
MAX_TIME_GAP_FACTOR = 1.8     # dt>1.8/fps 的采集时间断点不能连接轨迹。
SAVE_VIDEO = True             # False也会撤下本模块旧视频，防止与新CSV混淆。
# 两条鱼每窗各自bbox宽/高中位数，再两鱼平均，为该窗口固定归一化宽/高。
# =============================================================

import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


def _float(value, name, allow_empty=False):
    """CSV 空值代表不可测；非法文本不能悄悄变成有效零。"""
    if value is None or str(value).strip() == "":
        if allow_empty:
            return math.nan
        raise ValueError(f"缺少数值: {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须有限，CSV 不使用 NaN/Infinity 字符串")
    return result


def _integer(value, name, minimum=0):
    result = _float(value, name)
    if result != int(result) or result < minimum:
        raise ValueError(f"{name} 必须是 >= {minimum} 的整数")
    return int(result)


def _boolean(value, name):
    if str(value).strip() not in {"0", "1"}:
        raise ValueError(f"{name} 必须为 0 或 1")
    return str(value).strip() == "1"


def load_result(input_csv):
    """唯一数值输入是 geometry.py 生成的 result.csv。

    一行是一帧的一条鱼；empty_frame 行只构建完整时间轴，不虚构鱼。
    位置是否有效与中轴/头向是否有效分开，本模块不要求识别出鱼头。
    meta['timeline'] 保存每帧路径和时间，视频读取这些原图；不会读取其他指标输出。
    """
    path = Path(input_csv)
    if not path.is_absolute():
        raise ValueError("input_csv 必须使用绝对路径")
    required = {"schema_version", "run_id", "record_type", "video_id", "frame_id", "time_s",
                "fps", "image_path", "image_width", "image_height", "track_id", "cx", "cy",
                "position_valid", "geometry_valid", "head_valid", "area_px", "bbox_width", "bbox_height",
                "reference_body_length_px", "reference_length_sample_count", "quality_reason"}
    rows, timeline, seen, reference_values = [], {}, set(), {}
    signature = None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"result.csv 缺少列: {sorted(required-set(reader.fieldnames or []))}")
        for number, raw in enumerate(reader, 2):
            if _integer(raw["schema_version"], "schema_version") != 1:
                raise ValueError("不支持的 result.csv 版本")
            fps = _float(raw["fps"], "fps")
            if fps <= 0:
                raise ValueError("fps 必须大于零")
            width = _integer(raw["image_width"], "image_width", 1)
            height = _integer(raw["image_height"], "image_height", 1)
            this_signature = (raw["video_id"], raw["run_id"], fps, width, height)
            if not raw["video_id"] or not raw["run_id"]:
                raise ValueError("video_id/run_id 不能为空")
            if signature is not None and this_signature != signature:
                raise ValueError("一张表只能包含同一视频、运行批次、帧率和图像尺寸")
            signature = this_signature
            frame = _integer(raw["frame_id"], "frame_id")
            time_s = _float(raw["time_s"], "time_s")
            if time_s < 0:
                raise ValueError("time_s 不得为负")
            image_path = Path(raw["image_path"])
            if not image_path.is_absolute():
                raise ValueError("原图路径必须为绝对路径")
            item = {"frame_id": frame, "time_s": time_s, "image_path": str(image_path)}
            if frame in timeline and timeline[frame] != item:
                raise ValueError(f"同帧的时间或原图路径不一致: {frame}")
            timeline[frame] = item
            for field in ("position_valid", "geometry_valid", "head_valid"):
                _boolean(raw[field], field)
            kind = raw["record_type"]
            if kind == "empty_frame":
                if raw["track_id"].strip() or any(_boolean(raw[k], k) for k in ("position_valid", "geometry_valid", "head_valid")):
                    raise ValueError("空帧不能带鱼 ID 或有效性标记")
                key = (frame, None)
                if key in seen or any(f == frame for f, _ in seen):
                    raise ValueError("空帧重复或与观测行冲突")
                seen.add(key)
                continue
            if kind != "observation":
                raise ValueError(f"未知 record_type: {kind}")
            identity = _integer(raw["track_id"], "track_id")
            if (frame, identity) in seen or (frame, None) in seen:
                raise ValueError(f"重复/冲突的 frame_id,track_id: {frame},{identity}")
            seen.add((frame, identity))
            row = {**item, "track_id": identity, "position_valid": _boolean(raw["position_valid"], "position_valid"),
                   "quality_reason": raw["quality_reason"], "reference_length_sample_count": _integer(raw["reference_length_sample_count"], "reference_length_sample_count")}
            for field in ("cx", "cy", "area_px", "bbox_width", "bbox_height", "reference_body_length_px"):
                row[field] = _float(raw[field], field, allow_empty=True)
            if row["position_valid"] and (not all(math.isfinite(row[k]) for k in ("cx", "cy", "area_px"))
                    or not 0 <= row["cx"] < width or not 0 <= row["cy"] < height or row["area_px"] <= 0):
                raise ValueError(f"第 {number} 行声称位置有效但坐标/面积不合法")
            length = row["reference_body_length_px"]
            count = row["reference_length_sample_count"]
            if math.isfinite(length) and (length <= 0 or count <= 0):
                raise ValueError("有效参考体长及其样本数必须大于零")
            if not math.isfinite(length) and count != 0:
                raise ValueError("无参考体长时样本数应为零")
            ref = (None if not math.isfinite(length) else length, count)
            if identity in reference_values and reference_values[identity] != ref:
                raise ValueError("同一鱼的参考体长必须固定")
            reference_values[identity] = ref
            rows.append(row)
    if signature is None:
        raise ValueError("result.csv 没有帧信息")
    ordered = sorted(timeline.values(), key=lambda r: r["frame_id"])
    if any(b["time_s"] <= a["time_s"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("帧号递增时，时间必须严格递增")
    meta = {"video_id": signature[0], "run_id": signature[1], "fps": signature[2],
            "width": signature[3], "height": signature[4], "timeline": ordered, "input_csv": str(path)}
    return sorted(rows, key=lambda r: (r["frame_id"], r["track_id"])), meta


def write_csv(path, rows, descriptions):
    """中文 Excel 可识别的 UTF-8 BOM；列表以 JSON 文本保存；不可测数值为空。"""
    fields = list(descriptions)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            converted = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, (list, tuple, dict)):
                    value = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                elif value is None or isinstance(value, (float, np.floating)) and not math.isfinite(value):
                    value = ""
                elif isinstance(value, (bool, np.bool_)):
                    value = int(value)
                converted[field] = value
            writer.writerow(converted)


def _plot_setup(number):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    count = max(1, number)
    columns = 2 if count > 1 else 1
    fig, axes = plt.subplots(math.ceil(count / columns), columns, figsize=(13, max(3.5, math.ceil(count / columns)*3.2)), squeeze=False)
    for ax in axes.ravel()[count:]:
        ax.set_visible(False)
    return plt, fig, axes.ravel()


def _color(identity):
    return tuple(int(v) for v in np.random.default_rng(int(identity)).integers(70, 235, 3))


def _save_video(path, records, meta, draw):
    """每个源帧输出一帧，包括空鱼帧；按标称 fps 播放，计算始终使用实际时间。

    原图缺失/损坏应失败，不回退到上次生成的视频。先写临时 MP4，检查后发布。
    """
    path = Path(path)
    temp = path.with_name(path.stem + ".partial.mp4")
    path.unlink(missing_ok=True)
    temp.unlink(missing_ok=True)
    groups = defaultdict(list)
    for row in records:
        groups[row["frame_id"]].append(row)
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (meta["width"], meta["height"]))
    try:
        if not writer.isOpened():
            raise RuntimeError("无法打开 MP4 编码器")
        for frame in meta["timeline"]:
            image = cv2.imdecode(np.fromfile(frame["image_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (meta["height"], meta["width"]):
                raise ValueError(f"原图缺失、损坏或尺寸变化: {frame['image_path']}")
            draw(image, frame, groups[frame["frame_id"]])
            writer.write(image)
    except BaseException:
        writer.release()
        temp.unlink(missing_ok=True)
        raise
    writer.release()
    cap = cv2.VideoCapture(str(temp))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if count != len(meta["timeline"]) or (width, height) != (meta["width"], meta["height"]):
        temp.unlink(missing_ok=True)
        raise RuntimeError("生成的视频帧数或尺寸不正确")
    temp.replace(path)


def _output_directory(output_dir, names):
    output = Path(output_dir)
    if not output.is_absolute():
        raise ValueError("output_dir 必须使用绝对路径")
    output.mkdir(parents=True, exist_ok=True)
    # 仅清理本模块明确拥有的输出，不递归删除目录或用户文件。
    for name in names:
        (output/name).unlink(missing_ok=True)
    return output


from itertools import combinations


EVENT_DESCRIPTIONS = {
    "event_id": "本次运行去重后的事件编号，从1开始", "track_id_i": "鱼对中较小的鱼 ID", "track_id_j": "鱼对中较大的鱼 ID",
    "continuity_segment": "该鱼对连续可判断窗口段编号；缺帧/无效窗口会切段",
    "start_time_s": "合并事件中候选最近位置的最早到达时间（秒）", "end_time_s": "最晚到达时间（秒）",
    "representative_frame": "与代表事件时刻最近的源帧", "representative_time_s": "最小归一化距离候选的两条鱼到达时间均值",
    "fish_i_time_s": "代表候选中鱼 i 到达最近位置的插值时间", "fish_j_time_s": "鱼 j 对应的插值时间",
    "arrival_time_difference_s": "两条鱼到达时间的绝对差（秒）",
    "min_distance_normalized": "满足时间约束的最小尺寸归一化二维距离，无单位",
    "normalization_width_px": "两鱼窗口外接框宽度中位数的均值（像素）", "normalization_height_px": "两鱼高度中位数的均值（像素）",
    "fish_i_x": "代表候选中鱼 i 插值横坐标（像素）", "fish_i_y": "鱼 i 插值纵坐标（像素）",
    "fish_j_x": "代表候选中鱼 j 插值横坐标（像素）", "fish_j_y": "鱼 j 插值纵坐标（像素）",
    "candidate_windows": "合并进本事件的滑动窗口候选数量", "evidence_start_frame": "参与判断的最早窗口起帧", "evidence_end_frame": "最晚窗口末帧",
    "frame_range": "所有候选窗口实际参与的源帧 ID，JSON 数组",
    "window_frames": "滑动窗口样本帧数（8帧有7个时间间隔）", "time_tolerance_s": "允许两鱼到达最近位置的最大时间差",
    "distance_threshold": "归一化近交距离阈值", "merge_gap_s": "同一鱼对候选事件时间的最大合并间隔",
}
SUMMARY_DESCRIPTIONS = {
    "track_id": "鱼 ID", "start_frame": "首次观测帧", "end_frame": "最后观测帧",
    "observed_frames": "该鱼实际观测行数", "valid_position_frames": "位置有效的实际观测数",
    "valid_observation_time_s": "相邻连续有效位置之间的实际时间之和（秒）",
    "eligible_window_frames": "该鱼至少有一个可比较伙伴的完整窗口末帧数",
    "valid_pair_frames": "各完整窗口内可比较伙伴数量累计，作为事件率分母",
    "valid_pair_time_s": "按窗口末端真实帧间隔加权的有效伙伴时间（鱼对秒）",
    "event_count": "该鱼参与的去重事件数；同一事件给两条鱼各计一次",
    "partner_count": "至少发生一次事件的不同伙伴数", "partner_ids": "这些伙伴的 ID 列表",
    "events_per_1000_pair_frames": "event_count/valid_pair_frames*1000；无有效暴露时留空",
    "window_frames": "滑动窗口样本帧数", "time_tolerance_s": "最近位置最大到达时间差（秒）",
    "distance_threshold": "归一化近交阈值", "merge_gap_s": "候选合并最大时间间隔（秒）",
    "status": "ok、insufficient_frames 或 invalid", "reason": "不足或无效的原因，正常为空",
}


def _clip_polygon(poly, normal, limit):
    """将参数空间多边形裁到 normal·(u,v)<=limit；不裁原始鱼轮廓。"""
    if len(poly) == 0:
        return []
    output = []
    previous = np.asarray(poly[-1], dtype=float)
    prev_value = float(normal@previous-limit)
    for value in poly:
        point = np.asarray(value, dtype=float)
        now_value = float(normal@point-limit)
        prev_inside, now_inside = prev_value <= 1e-12, now_value <= 1e-12
        if prev_inside != now_inside:
            fraction = prev_value/(prev_value-now_value)
            output.append(previous+fraction*(point-previous))
        if now_inside:
            output.append(point)
        previous, prev_value = point, now_value
    return output


def segment_minimum(pa, qa, ta, ua, pb, qb, tb, ub, tolerance):
    """两条带时间的线段之间，受时间差约束的精确凸二次最小距离。

    参数 u,v∈[0,1] 分别表示两鱼在各自线段上的进度；时间差是一条线性带状约束。
    把单位正方形裁成可行多边形后，检查无约束驻点以及每条边上的一维最小点。
    因目标 ||pa+u(qa-pa)-pb-v(qb-pb)||² 为凸函数，这覆盖全局最优点。
    平行/静止/交叉线段也可处理；不是仅用8个采样点，也不是先忽略时间再过滤。
    输入坐标已用该窗口的宽/高尺度归一化。返回(distance²,u,v)，无可行点返回None。
    """
    normal = np.asarray([ua-ta, -(ub-tb)], dtype=float)
    offset = ta-tb
    poly = _clip_polygon([np.array([0.,0.]), np.array([1.,0.]), np.array([1.,1.]), np.array([0.,1.])], normal, tolerance-offset)
    poly = _clip_polygon(poly, -normal, tolerance+offset)
    if not poly:
        return None
    c = np.asarray(pa)-np.asarray(pb)
    matrix = np.column_stack((np.asarray(qa)-pa, -(np.asarray(qb)-pb)))
    candidates = []
    for a, b in zip(poly, poly[1:]+poly[:1]):
        edge = b-a
        origin, delta = c+matrix@a, matrix@edge
        norm = float(delta@delta)
        fraction = float(np.clip(-(origin@delta)/norm, 0, 1)) if norm > 1e-24 else 0.
        candidates.append(a+fraction*edge)
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) > 1e-12:
        point = np.linalg.solve(matrix, -c)
        if np.all(point >= -1e-12) and np.all(point <= 1+1e-12) and abs(float(normal@point+offset)) <= tolerance+1e-12:
            candidates.append(np.clip(point, 0, 1))
    results = [(float(np.dot(c+matrix@point, c+matrix@point)), float(point[0]), float(point[1])) for point in candidates]
    return min(results, key=lambda x: (round(x[0], 13), (ta+x[1]*(ua-ta)+tb+x[2]*(ub-tb))/2))


def closest_candidate(a, b, tolerance, threshold):
    """只在真实可能低于阈值的线段对上做精确搜索；包围盒下界过滤不会漏近交。"""
    scale = np.asarray([(np.median([r["bbox_width"] for r in a])+np.median([r["bbox_width"] for r in b]))/2,
                        (np.median([r["bbox_height"] for r in a])+np.median([r["bbox_height"] for r in b]))/2])
    ap = np.asarray([[r["cx"],r["cy"]] for r in a])/scale
    bp = np.asarray([[r["cx"],r["cy"]] for r in b])/scale
    at, bt = np.asarray([r["time_s"] for r in a]), np.asarray([r["time_s"] for r in b])
    amin, amax = np.minimum(ap[:-1],ap[1:]), np.maximum(ap[:-1],ap[1:])
    bmin, bmax = np.minimum(bp[:-1],bp[1:]), np.maximum(bp[:-1],bp[1:])
    separation = np.maximum(0, np.maximum(amin[:,None,:]-bmax[None,:,:], bmin[None,:,:]-amax[:,None,:]))
    lower = np.sum(separation**2,axis=2)
    possible = (lower <= threshold**2+1e-12) & (np.maximum(at[:-1,None]-bt[None,1:],bt[None,:-1]-at[1:,None]) <= tolerance+1e-12)
    indices = np.argwhere(possible)
    if not len(indices):
        return None
    indices = sorted(indices, key=lambda ij: lower[tuple(ij)])
    best, best_value = None, threshold**2+1e-12
    for i,j in indices:
        if lower[i,j] > best_value+1e-12:
            continue
        found = segment_minimum(ap[i],ap[i+1],at[i],at[i+1],bp[j],bp[j+1],bt[j],bt[j+1],tolerance)
        if found is None or found[0] > best_value+1e-12:
            continue
        value,u,v = found
        ti,tj = float(at[i]+u*(at[i+1]-at[i])),float(bt[j]+v*(bt[j+1]-bt[j]))
        representative = (ti+tj)/2
        if best is not None and abs(value-best_value) <= 1e-12 and representative >= best["representative_time_s"]:
            continue
        pi,pj = (ap[i]+u*(ap[i+1]-ap[i]))*scale,(bp[j]+v*(bp[j+1]-bp[j]))*scale
        best_value = value
        best = dict(min_distance_normalized=math.sqrt(max(0.,value)), fish_i_time_s=ti,fish_j_time_s=tj,
                    arrival_time_difference_s=abs(ti-tj), representative_time_s=representative,
                    normalization_width_px=float(scale[0]),normalization_height_px=float(scale[1]),
                    fish_i_x=float(pi[0]),fish_i_y=float(pi[1]),fish_j_x=float(pj[0]),fish_j_y=float(pj[1]))
    return best


def calculate(records, meta, *, window_frames=WINDOW_FRAMES, time_tolerance_frames=TIME_TOLERANCE_FRAMES,
              distance_threshold=DISTANCE_THRESHOLD, merge_gap_frames=MERGE_GAP_FRAMES):
    """纯计算，返回 events/summary；不读取图像、不写文件。

    8帧窗口以每个源帧为末端、步长1。只有每一帧都有该鱼且位置/框尺寸有效、
    时间连续时才可比较。每个完整鱼对窗口贡献1个pair-frame，不因没发现事件而
    丢弃分母。时间容差及合并间隔换算成秒，跨帧率比较必须统一实际时间尺度。
    缺帧/位置无效切断鱼对连续段；不会通过缺口连接轨迹或合并事件。
    """
    if isinstance(window_frames,bool) or int(window_frames)!=window_frames or window_frames<2:
        raise ValueError("window_frames 必须是 >=2 的整数")
    window_frames=int(window_frames)
    if any(not math.isfinite(float(x)) or x<0 for x in (time_tolerance_frames,merge_gap_frames)) or not math.isfinite(distance_threshold) or distance_threshold<=0:
        raise ValueError("时间参数必须有限非负，距离阈值必须有限且大于零")
    tolerance,merge_gap=time_tolerance_frames/meta["fps"],merge_gap_frames/meta["fps"]
    lookup, by_id = {},defaultdict(list)
    for row in records:
        lookup[(row["frame_id"],row["track_id"])]=row;by_id[row["track_id"]].append(row)
    identities=sorted(by_id)
    times={r["frame_id"]:r["time_s"] for r in meta["timeline"]}
    exposures=defaultdict(int);eligible=defaultdict(set);pair_seconds=defaultdict(float)
    pair_last={};segments=defaultdict(int);candidates=[]
    for frame in sorted(times):
        frame_range=list(range(frame-window_frames+1,frame+1))
        if any(f not in times for f in frame_range):
            continue
        if any(not 0<times[q]-times[p]<=MAX_TIME_GAP_FACTOR/meta["fps"] for p,q in zip(frame_range,frame_range[1:])):
            continue
        windows={}
        for identity in identities:
            fish=[lookup.get((f,identity)) for f in frame_range]
            if all(r is not None and r["position_valid"] and np.isfinite([r["cx"],r["cy"],r["bbox_width"],r["bbox_height"]]).all()
                   and r["bbox_width"]>0 and r["bbox_height"]>0 for r in fish):
                windows[identity]=fish
        for i,j in combinations(sorted(windows),2):
            pair=(i,j)
            if pair_last.get(pair)!=frame-1:
                segments[pair]+=1
            pair_last[pair]=frame
            for identity in pair:
                exposures[identity]+=1;eligible[identity].add(frame)
                pair_seconds[identity]+=times[frame]-times[frame-1]
            closest=closest_candidate(windows[i],windows[j],tolerance,distance_threshold)
            if closest is not None and closest["min_distance_normalized"]<=distance_threshold+1e-10:
                closest.update(track_id_i=i,track_id_j=j,continuity_segment=segments[pair],
                               evidence_start_frame=frame_range[0],evidence_end_frame=frame,
                               frame_range=frame_range,window_frames=window_frames,time_tolerance_s=tolerance,
                               distance_threshold=distance_threshold,merge_gap_s=merge_gap)
                candidates.append(closest)
    events=merge_candidates(candidates,meta,merge_gap)
    summaries=[]
    for identity in identities:
        fish=sorted(by_id[identity],key=lambda r:r["frame_id"])
        involving=[r for r in events if identity in (r["track_id_i"],r["track_id_j"])]
        partners=sorted({r["track_id_j"] if r["track_id_i"]==identity else r["track_id_i"] for r in involving})
        duration=sum(b["time_s"]-a["time_s"] for a,b in zip(fish,fish[1:]) if a["position_valid"] and b["position_valid"]
                     and b["frame_id"]==a["frame_id"]+1 and 0<b["time_s"]-a["time_s"]<=MAX_TIME_GAP_FACTOR/meta["fps"])
        ready=exposures[identity]>0
        short=len(fish)<window_frames
        summaries.append(dict(track_id=identity,start_frame=fish[0]["frame_id"],end_frame=fish[-1]["frame_id"],
                              observed_frames=len(fish),valid_position_frames=sum(r["position_valid"] for r in fish),
                              valid_observation_time_s=duration,eligible_window_frames=len(eligible[identity]),
                              valid_pair_frames=exposures[identity],valid_pair_time_s=pair_seconds[identity],
                              event_count=len(involving),partner_count=len(partners),partner_ids=partners,
                              events_per_1000_pair_frames=1000*len(involving)/exposures[identity] if ready else math.nan,
                              window_frames=window_frames,time_tolerance_s=tolerance,distance_threshold=distance_threshold,merge_gap_s=merge_gap,
                              status="ok" if ready else ("insufficient_frames" if short else "invalid"),
                              reason="" if ready else ("insufficient_observations" if short else "no_valid_pair_windows")))
    return events,summaries


def merge_candidates(candidates,meta,merge_gap):
    """先按鱼对、连续段和真正候选时刻排序，再合并，不用窗口编号代替事件时间。

    相邻候选间隔不超过阈值时连成一个持续事件，因此长期贴近不会逐窗口重复计数。
    这是二维近交/接近事件，近距离平行游动也可满足条件；不宣称碰撞或攻击。
    """
    groups=defaultdict(list)
    for row in candidates:
        groups[(row["track_id_i"],row["track_id_j"],row["continuity_segment"])].append(row)
    events=[]
    for key,group in sorted(groups.items()):
        batches=[];current=[]
        for row in sorted(group,key=lambda r:r["representative_time_s"]):
            if current and row["representative_time_s"]-current[-1]["representative_time_s"]>merge_gap+1e-10:
                batches.append(current);current=[]
            current.append(row)
        if current:batches.append(current)
        for batch in batches:
            best=min(batch,key=lambda r:(r["min_distance_normalized"],r["representative_time_s"]))
            event=dict(best)
            event.update(start_time_s=min(min(r["fish_i_time_s"],r["fish_j_time_s"]) for r in batch),
                         end_time_s=max(max(r["fish_i_time_s"],r["fish_j_time_s"]) for r in batch),
                         candidate_windows=len(batch),frame_range=sorted({f for r in batch for f in r["frame_range"]}),
                         evidence_start_frame=min(r["evidence_start_frame"] for r in batch),evidence_end_frame=max(r["evidence_end_frame"] for r in batch))
            event["representative_frame"]=min(meta["timeline"],key=lambda f:abs(f["time_s"]-best["representative_time_s"]))["frame_id"]
            events.append(event)
    events.sort(key=lambda r:(r["representative_time_s"],r["track_id_i"],r["track_id_j"]))
    for index,row in enumerate(events,1):row["event_id"]=index
    return events


def plot_results(events,summaries,meta,path):
    plt,fig,axes=_plot_setup(len(summaries))
    times=np.asarray([f["time_s"] for f in meta["timeline"]])
    for ax,summary in zip(axes,summaries):
        selected=[e for e in events if summary["track_id"] in (e["track_id_i"],e["track_id_j"])]
        event_times=sorted(e["representative_time_s"] for e in selected)
        if summary["status"]=="ok":
            ax.step(times,np.searchsorted(event_times,times,side="right"),where="post",color="#ca6702")
            for time_s in event_times:ax.axvline(time_s,color="#bb3e03",alpha=.35)
            rate=f"{summary['events_per_1000_pair_frames']:.3g}/1000 pair-frames"
        else:
            ax.text(.5,.5,"Unavailable: "+summary["reason"],ha="center",va="center",transform=ax.transAxes);rate="unavailable"
        ax.set(title=f"Fish {summary['track_id']} | {rate}",xlabel="Time (s)",ylabel="Cumulative incident events")
        ax.grid(alpha=.25)
    if not summaries:axes[0].text(.5,.5,"No fish observations",ha="center",va="center",transform=axes[0].transAxes)
    fig.suptitle("Individual 2D trajectory near-crossing | pair events count once for each participant")
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(path,dpi=160);plt.close(fig)


def make_video(records,events,summaries,meta,path,window_frames):
    lookup={(r["frame_id"],r["track_id"]):r for r in records}
    summaries={r["track_id"]:r for r in summaries}
    def draw(image,frame,fish):
        f=frame["frame_id"]
        for row in fish:
            if not row["position_valid"]:continue
            identity=row["track_id"];color=_color(identity)
            points=[]
            for k in range(f-window_frames+1,f+1):
                old=lookup.get((k,identity))
                if old is None or not old["position_valid"]:
                    if len(points)>1:cv2.polylines(image,[np.asarray(points,dtype=np.int32)],False,color,2,cv2.LINE_AA)
                    points=[]
                else:points.append([round(old["cx"]),round(old["cy"])])
            if len(points)>1:cv2.polylines(image,[np.asarray(points,dtype=np.int32)],False,color,2,cv2.LINE_AA)
            total=sum(identity in (e["track_id_i"],e["track_id_j"]) and e["representative_time_s"]<=frame["time_s"] for e in events)
            label=str(total) if summaries[identity]["status"]=="ok" else "unavailable"
            x,y=int(round(row["cx"])),int(round(row["cy"]))
            cv2.circle(image,(x,y),5,color,-1)
            cv2.putText(image,f"ID {identity}: events={label}",(max(0,x-50),max(20,y-15)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2,cv2.LINE_AA)
        for event in events:
            if abs(frame["time_s"]-event["representative_time_s"])<=.20:
                a=(round(event["fish_i_x"]),round(event["fish_i_y"]));b=(round(event["fish_j_x"]),round(event["fish_j_y"]))
                cv2.line(image,a,b,(0,0,255),3,cv2.LINE_AA)
                cv2.circle(image,a,12,(0,0,255),2);cv2.circle(image,b,12,(0,0,255),2)
                cv2.putText(image,f"Event {event['event_id']}: {event['track_id_i']}-{event['track_id_j']}",(max(0,a[0]-50),max(20,a[1]-30)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(image,f"Frame {f} t={frame['time_s']:.3f}s | 2D near-crossing, no depth",(15,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,220,255),2,cv2.LINE_AA)
    _save_video(path,records,meta,draw)


def run(input_csv=INPUT_CSV,output_dir=OUTPUT_DIR,*,save_video=SAVE_VIDEO,window_frames=WINDOW_FRAMES,
        time_tolerance_frames=TIME_TOLERANCE_FRAMES,distance_threshold=DISTANCE_THRESHOLD,merge_gap_frames=MERGE_GAP_FRAMES):
    """独立调用 run(绝对result.csv路径,绝对输出目录,save_video=True,window_frames=8)。

    返回 {'events':鱼对事件列表,'summary':每鱼汇总列表}。事件表一行一个事件，
    汇总表一行一条鱼。某事件两条鱼各加1，因此每鱼事件数之和=2×全体鱼对事件数。
    无暴露时事件率为空且状态无效；有暴露而无事件时事件率为有效0。
    """
    records,meta=load_result(input_csv)
    events,summaries=calculate(records,meta,window_frames=window_frames,time_tolerance_frames=time_tolerance_frames,
                              distance_threshold=distance_threshold,merge_gap_frames=merge_gap_frames)
    output=_output_directory(output_dir,["entanglement_events.csv","entanglement_summary.csv","entanglement.png","entanglement.mp4"])
    write_csv(output/"entanglement_events.csv",events,EVENT_DESCRIPTIONS)
    write_csv(output/"entanglement_summary.csv",summaries,SUMMARY_DESCRIPTIONS)
    plot_results(events,summaries,meta,output/"entanglement.png")
    if save_video:make_video(records,events,summaries,meta,output/"entanglement.mp4",int(window_frames))
    print(f"Entanglement: {len(events)} pair events, {len(summaries)} fish; saved {output}")
    return {"events":events,"summary":summaries}


if __name__=="__main__":
    run()
