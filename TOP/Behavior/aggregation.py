# 顶部视角维护负责人：YHC。此脚本仅属于 FishBehavior/Top，不依赖 Side 子项目。
"""个体局部聚集分析：独立读取 result.csv，不调用任何其他项目脚本。

主指标：Delaunay 三角网络中当前鱼到直接邻鱼的平均像素距离，越小越集中。
BL 版本仅把该鱼的平均距离除以它自己的固定参考体长，不改变三角剖分坐标。
输出为每鱼逐帧 CSV、每鱼汇总 CSV、合并曲线图和三角网络叠加视频。
"""
from pathlib import Path

# ==================== 可修改参数和绝对路径 ====================
INPUT_CSV = Path(r"E:\Phd\FishBehavior\Top\Tracker\Output\Information\result.csv")
OUTPUT_DIR = Path(r"E:\Phd\FishBehavior\Top\Behavior\Output\aggregation")
SAVE_VIDEO = True              # False 不生成视频，并清除本模块已有同名视频，避免新表配旧视频。
MAX_TIME_GAP_FACTOR = 1.8       # 仅累计相邻帧且 dt<=1.8/fps 的有效观察时间；剖分本身逐帧计算。
# 参数不是自动训练的阈值；本指标不使用鱼头方向，也不把中轴失败直接视为位置失败。
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


from scipy.spatial import Delaunay, QhullError


FRAME_DESCRIPTIONS = {
    "frame_id": "原始帧号", "time_s": "相对第一张图片的时间（秒）", "track_id": "鱼 ID，0 合法",
    "cx": "当前鱼质心横坐标（像素）", "cy": "当前鱼质心纵坐标（像素）",
    "valid_fish_count": "本帧位置有效的鱼数", "triangle_count": "本帧 Delaunay 三角形数量",
    "neighbor_count": "与当前鱼直接相连的不同邻鱼数量", "neighbor_ids": "相邻鱼 ID 的 JSON 列表",
    "mean_neighbor_distance_px": "当前鱼到所有直接邻鱼的平均欧氏距离（像素）；越小越集中",
    "mean_neighbor_distance_bl": "平均邻鱼距离除以当前鱼自己的固定参考体长（BL）",
    "reference_body_length_px": "当前鱼整批有效中轴弧长的中位数（像素）",
    "reference_length_sample_count": "参考体长的有效样本数",
    "valid": "二维邻鱼距离是否有效，1 是、0 否", "bl_valid": "BL 邻鱼距离是否有效",
    "quality_reason": "像素结果不可用原因；正常为空", "bl_quality_reason": "BL 结果不可用原因；正常为空",
}
SUMMARY_DESCRIPTIONS = {
    "track_id": "鱼 ID", "start_frame": "首次观测帧", "end_frame": "最后观测帧",
    "observed_frames": "实际观测行数，包括位置无效行", "missing_frames": "首末观测之间清单中没有该鱼的帧数",
    "valid_frames": "邻鱼距离有效的帧数", "invalid_frames": "实际观测中无效的帧数",
    "valid_fraction": "有效帧数除以实际观测帧数", "valid_time_s": "相邻连续有效观测之间累计的真实时间（秒）",
    "mean_neighbor_distance_px": "有效帧邻鱼距离的算术平均（像素，非按时间加权）",
    "median_neighbor_distance_px": "有效帧邻鱼距离中位数（像素）",
    "std_neighbor_distance_px": "有效帧邻鱼距离的总体标准差（像素）",
    "min_neighbor_distance_px": "有效帧中最小平均邻鱼距离（像素）",
    "max_neighbor_distance_px": "有效帧中最大平均邻鱼距离（像素）",
    "mean_neighbor_distance_bl": "每鱼固定体长换算后的有效帧平均邻鱼距离（BL）",
    "median_neighbor_distance_bl": "有效帧邻鱼距离中位数（BL）",
    "std_neighbor_distance_bl": "有效帧邻鱼距离总体标准差（BL）",
    "mean_neighbor_count": "有效帧中直接邻鱼数量的算术平均",
    "reference_body_length_px": "当前鱼固定参考体长（像素）", "reference_length_sample_count": "参考体长样本数",
    "status": "ok 或 invalid；无有效帧时均值留空", "reason": "无有效统计的原因",
    "bl_status": "BL 汇总是否有效", "bl_reason": "BL 汇总无效原因",
}


def calculate(records, meta):
    """纯计算：(基础观测列表,完整时间轴信息) -> (逐帧行,每鱼汇总)。

    必须先在原始像素坐标中做 Delaunay，再计算每鱼 UNIQUE incident edges。
    不对三角形重复边重复计数，也不先将各鱼坐标除以其各自 BL。
    少于三点、重复点、共线、Qhull 失败或遗漏顶点时该帧全部位置有效鱼
    均标记不可计算；不使用 QJ 随机扰动伪造可剖分的几何。位置无效鱼不入网。
    平均边长越小表示局部紧密程度越高；不是 0-1 概率、社会偏好或全群聚集度。
    """
    groups = defaultdict(list)
    for row in records:
        groups[row["frame_id"]].append(row)
    output = []
    for frame in meta["timeline"]:
        fish = sorted(groups[frame["frame_id"]], key=lambda r: r["track_id"])
        good = [r for r in fish if r["position_valid"] and np.isfinite([r["cx"], r["cy"]]).all()]
        points = np.asarray([[r["cx"], r["cy"]] for r in good], dtype=float).reshape(-1, 2)
        reason, triangle_count, neighbors = "", 0, defaultdict(set)
        if len(good) < 3:
            reason = "fewer_than_three_valid_fish"
        elif len(np.unique(points, axis=0)) != len(points):
            reason = "duplicate_centroids"
        elif np.linalg.matrix_rank(points-points.mean(axis=0)) < 2:
            reason = "collinear_centroids"
        else:
            try:
                tri = Delaunay(points)
                if len(np.unique(tri.simplices)) != len(points):
                    reason = "delaunay_omitted_vertices"
                else:
                    triangle_count = len(tri.simplices)
                    for a, b, c in tri.simplices:
                        for i, j in ((a, b), (a, c), (b, c)):
                            neighbors[int(i)].add(int(j)); neighbors[int(j)].add(int(i))
            except QhullError:
                reason = "delaunay_failed"
        index = {r["track_id"]: i for i, r in enumerate(good)}
        for row in fish:
            result = {k: math.nan for k in FRAME_DESCRIPTIONS}
            result.update(frame_id=row["frame_id"], time_s=row["time_s"], track_id=row["track_id"],
                          cx=row["cx"], cy=row["cy"], valid_fish_count=len(good), triangle_count=triangle_count,
                          neighbor_count=0, neighbor_ids=[], valid=0, bl_valid=0,
                          reference_body_length_px=row["reference_body_length_px"],
                          reference_length_sample_count=row["reference_length_sample_count"],
                          quality_reason=reason, bl_quality_reason="")
            if row["track_id"] not in index:
                result["quality_reason"] = "invalid_position"
            elif not reason:
                i = index[row["track_id"]]
                adjacent = sorted(neighbors[i])
                if not adjacent:
                    result["quality_reason"] = "no_neighbors"
                else:
                    distance = float(np.linalg.norm(points[adjacent]-points[i], axis=1).mean())
                    result.update(valid=1, neighbor_count=len(adjacent), neighbor_ids=sorted(good[j]["track_id"] for j in adjacent), mean_neighbor_distance_px=distance)
                    length = row["reference_body_length_px"]
                    if math.isfinite(length) and length > 0:
                        result.update(mean_neighbor_distance_bl=distance/length, bl_valid=1)
                    else:
                        result["bl_quality_reason"] = "no_valid_body_length"
            if not result["valid"]:
                result["bl_quality_reason"] = result["quality_reason"]
            output.append(result)
    return output, summarize(output, meta)


def summarize(rows, meta):
    groups = defaultdict(list)
    for row in rows:
        groups[row["track_id"]].append(row)
    timeline = [r["frame_id"] for r in meta["timeline"]]
    output = []
    for identity, fish in sorted(groups.items()):
        fish = sorted(fish, key=lambda r: r["frame_id"])
        valid = [r for r in fish if r["valid"]]
        values = np.asarray([r["mean_neighbor_distance_px"] for r in valid])
        bl = np.asarray([r["mean_neighbor_distance_bl"] for r in fish if r["bl_valid"]])
        duration = sum(b["time_s"]-a["time_s"] for a, b in zip(fish, fish[1:])
                       if a["valid"] and b["valid"] and b["frame_id"] == a["frame_id"]+1
                       and 0 < b["time_s"]-a["time_s"] <= MAX_TIME_GAP_FACTOR/meta["fps"])
        summary = {k: math.nan for k in SUMMARY_DESCRIPTIONS}
        summary.update(track_id=identity, start_frame=fish[0]["frame_id"], end_frame=fish[-1]["frame_id"],
                       observed_frames=len(fish), missing_frames=sum(fish[0]["frame_id"] <= f <= fish[-1]["frame_id"] for f in timeline)-len(fish),
                       valid_frames=len(valid), invalid_frames=len(fish)-len(valid), valid_fraction=len(valid)/len(fish), valid_time_s=duration,
                       reference_body_length_px=fish[0]["reference_body_length_px"], reference_length_sample_count=fish[0]["reference_length_sample_count"],
                       status="ok" if valid else "invalid", reason="" if valid else "no_valid_triangulation",
                       bl_status="ok" if len(bl) else "invalid", bl_reason="" if len(bl) else ("no_valid_body_length" if valid else "no_valid_triangulation"))
        if len(values):
            for name, function in (("mean", np.mean), ("median", np.median), ("std", np.std), ("min", np.min), ("max", np.max)):
                summary[f"{name}_neighbor_distance_px"] = float(function(values))
            summary["mean_neighbor_count"] = float(np.mean([r["neighbor_count"] for r in valid]))
        if len(bl):
            for name, function in (("mean", np.mean), ("median", np.median), ("std", np.std)):
                summary[f"{name}_neighbor_distance_bl"] = float(function(bl))
        output.append(summary)
    return output


def plot_results(rows, summaries, meta, path):
    plt, fig, axes = _plot_setup(len(summaries))
    lookup = {(r["track_id"], r["frame_id"]): r for r in rows}
    for ax, summary in zip(axes, summaries):
        values = [lookup.get((summary["track_id"], f["frame_id"]), {}).get("mean_neighbor_distance_px", math.nan) for f in meta["timeline"]]
        ax.plot([f["time_s"] for f in meta["timeline"]], values, color="#247ba0", linewidth=1.2)
        ax.set(title=f"Fish {summary['track_id']} | valid {summary['valid_frames']}/{summary['observed_frames']}", xlabel="Time (s)", ylabel="Mean Delaunay neighbor distance (px)")
        ax.grid(alpha=.25)
    if not summaries:
        axes[0].text(.5, .5, "No fish observations", ha="center", va="center", transform=axes[0].transAxes)
    fig.suptitle("Individual local aggregation | smaller distance = tighter neighborhood; gaps = unavailable")
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(path, dpi=160); plt.close(fig)


def make_video(records, rows, meta, path):
    lookup = {(r["frame_id"], r["track_id"]): r for r in rows}
    def draw(image, frame, fish):
        positions = {r["track_id"]: (int(round(r["cx"])), int(round(r["cy"]))) for r in fish if r["position_valid"]}
        drawn = set()
        for row in fish:
            identity = row["track_id"]
            if identity not in positions:
                continue
            result = lookup[(frame["frame_id"], identity)]
            for other in result["neighbor_ids"]:
                edge = tuple(sorted((identity, other)))
                if edge not in drawn and other in positions:
                    cv2.line(image, positions[identity], positions[other], (100, 220, 100), 1, cv2.LINE_AA); drawn.add(edge)
            x, y = positions[identity]
            cv2.circle(image, (x, y), 5, _color(identity), -1)
            value = f"{result['mean_neighbor_distance_px']:.1f}px" if result["valid"] else "unavailable"
            cv2.putText(image, f"ID {identity}: {value}", (max(0,x-55), max(20,y-15)), cv2.FONT_HERSHEY_SIMPLEX, .52, _color(identity), 2, cv2.LINE_AA)
        cv2.putText(image, f"Frame {frame['frame_id']}  t={frame['time_s']:.3f}s | smaller distance = tighter neighborhood", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2, cv2.LINE_AA)
    _save_video(path, records, meta, draw)


def run(input_csv=INPUT_CSV, output_dir=OUTPUT_DIR, *, save_video=SAVE_VIDEO):
    """独立函数调用 run(绝对result.csv路径, 绝对输出目录, save_video=True)。

    返回 {'frames':逐帧列表,'summary':每鱼列表}，不写 JSON。四个固定输出：
    aggregation_frames.csv / aggregation_summary.csv / aggregation.png / aggregation.mp4。
    数值列含义见上方两个 *_DESCRIPTIONS 字典；无效值在 CSV 中留空。
    """
    records, meta = load_result(input_csv)
    rows, summaries = calculate(records, meta)
    output = _output_directory(output_dir, ["aggregation_frames.csv", "aggregation_summary.csv", "aggregation.png", "aggregation.mp4"])
    write_csv(output/"aggregation_frames.csv", rows, FRAME_DESCRIPTIONS)
    write_csv(output/"aggregation_summary.csv", summaries, SUMMARY_DESCRIPTIONS)
    plot_results(rows, summaries, meta, output/"aggregation.png")
    if save_video:
        make_video(records, rows, meta, output/"aggregation.mp4")
    print(f"Aggregation: {len(rows)} fish-frame rows, {len(summaries)} fish; saved {output}")
    return {"frames": rows, "summary": summaries}


if __name__ == "__main__":
    run()
