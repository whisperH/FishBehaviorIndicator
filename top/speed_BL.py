"""顶部视角：独立的 BL（Body Length，鱼体长度）距离/速度分析。

依赖同目录已有的 speed.py 与 geometry.py，不需要重新运行 SAM3 分割。
先用 speed.py 原算法计算像素结果，再用每鱼固定参考体长统一换算，
不修改 speed.py 的参数或计算函数，也不改变 main.py 的行为。

1 BL 的定义：该 ID 在本次输入全部有效帧中的中轴线头端到尾端长度中位数。
这是根据分割图像估计的体长，包含中轴线延伸到的尾端，不是人工测量的
解剖学标准体长，也不是外接框宽度或头尾直线距离。各鱼独立估计，整段
视频固定使用一个比例，所有帧步长共用；不同视频会重新估计。
因此不能只用一小段输入估计比例、再把其余部分当作同一次完整分析。
输入应是同一视频、同一视角、同一图像尺度的一批记录，ID 应持续对应同一条鱼。

BL=px/L，BL/s=(px/s)/L，其中 L 为该鱼参考体长（px）。例如 L=200，
移动 -100 px 就是 -0.5 BL；-100 px/s 就是 -0.5 BL/s。
图像坐标 cx/cy 仍是 px、area 仍是 px^2，绘制轨迹时使用原图坐标。
"""
from pathlib import Path

# ---------------------- 用户参数（路径全部为绝对路径） ----------------------
# 这里填已经完成的 SAM3 分割结果目录，不是原图目录，也不是模型权重路径。
SEGMENTATION_DIR = Path(r"E:\Phd\temperature\sam3\top\outputs\20261114_095054_184431")
# 每次运行在该目录下创建时间戳子目录，不覆盖像素版本或以前的 BL 结果。
OUTPUT_DIR = Path(r"E:\Phd\temperature\sam3\top\analysis\speed_BL")
FRAME_STEPS = [1, 5]            # 可改成 [1, 3, 10]；每鱼首帧固定锚点，非滑动窗口
HEAD_HINTS = {}                 # 首次头向人工提示，例如 {0: "left"}；含义同原脚本
USE_FILENAME_TIMESTAMPS = True  # 旧格式优先原图文件名时间；标准输入采用其时间规则
SAVE_VIDEO = True               # False 时仍输出全部数值和图表，仅跳过 MP4
TRAIL_FRAMES = 30               # 视频轨迹长度；只影响显示，不影响计算
# 运动质量阈值和判号容差直接使用 speed.py 的现有设置，不在这里重复配置。
# 体长只使用 valid 为真且 body_length_px 有限、>0 的记录；不额外剔除帧。
# -------------------------------------------------------------------------

import json
import math
from collections import defaultdict, deque
from datetime import datetime
from statistics import median

import cv2
import numpy as np

import speed as pixel_speed


# 明确列出每个需要换算的字段，防止误把图像坐标、面积或时间也除以体长。
# dx/dy：图像 x 向右、y 向下的位移分量；step_distance：窗口首末欧氏距离；
# signed_distance：完整欧氏距离乘以前后向符号，纯侧移为正；speed：距离/dt。
# forward_projection/lateral_distance：诊断投影，不代替主指标的完整位移。
# sign_tolerance：原算法在 px 下判断符号时使用的容差，仅换算展示，不重新判号。
# cumulative_*：按该步长的有效窗口累计；五帧累计不是逐帧累计距离再分组。
# path/signed/forward/backward：汇总无符号、有符号、正向、反向绝对值总距离。
# mean_*：原算法的相应有效总距离/相应有效时间，再除以固定参考体长。
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

# 新增追溯列：参考体长(px)、参与中位数的有效帧数、体长是否可用、不可用原因。
# body_length_valid=1 表示至少有一个合格长度；样本数少时可从 count 直接看到。
# quality_reason/detail 及原 distance_valid/direction_valid 完全保留像素计算结果。
# BL 指标必须查看 bl_distance_valid/bl_direction_valid：原指标有效且体长可用。
# 汇总亦然：用 bl_*_summary_valid 判断 BL 总距离/均值是否可解释。
# 不覆盖原标记，便于区分“运动本身无法计算”和“运动可计算但没有体长标尺”。
# 保留的明细列含义（与 speed.py 一致）：
# frame_step/anchor_frame：采样步长和该鱼首次观测帧；start_frame/frame：本窗口起末帧；
# expected_end_frame：原计划末帧，尾部不足时大于实际末帧；frame_range：实际观测帧列表。
# missing_frame_count：缺失总帧数；missing_frame_ranges：缺帧闭区间[[起,末],...]；
# missing_frames：小缺口逐帧列出，大缺口为空，不能只看这个空列表判定没有缺帧。
# window_complete：窗口长度足够且无缺帧，不保证时间/轮廓/头向也合格。
# start_time_s/time_s/dt_s：起点、终点时间及差值，秒；首行dt为空，起点时间归零规则同原版。
# id/filename：鱼的跟踪ID/当前端点图像文件名；cx/cy/area：当前端点质心(px)/面积(px^2)。
# heading_deg/head_confidence：当前头向角度(度)/头向置信信息，直接复制，不改变速度算法。
# direction_sign：+1前向或侧移，-1后向；quality_reason/detail：原运动无效原因及细节。
# 汇总保留列含义：start_frame/end_frame/frame_range为本鱼观测范围；observations为观测帧数；
# sample_rows为含首行/不足窗口的明细行数；candidate_intervals=sample_rows-1；
# incomplete_intervals为末尾不足窗口数；track_span_s为首末观测时间跨度。
# valid_distance_intervals/valid_signed_intervals为原像素无符号/有符号有效窗口数；
# valid_distance_time_s/valid_signed_time_s为相应有效dt之和；distance_time_coverage/
# signed_time_coverage为相应有效时间除以track_span_s，跨度为0则留空。
REFERENCE_FIELDS = ["reference_body_length_px", "reference_length_sample_count",
                    "body_length_valid", "body_length_reason"]
FRAME_FIELDS = [METRIC_NAMES.get(k, k) for k in pixel_speed.FRAME_FIELDS] + REFERENCE_FIELDS + [
    "bl_distance_valid", "bl_direction_valid"]
SUMMARY_FIELDS = [METRIC_NAMES.get(k, k) for k in pixel_speed.SUMMARY_FIELDS] + REFERENCE_FIELDS + [
    "bl_distance_summary_valid", "bl_direction_summary_valid"]
CURVES = [
    ("speed_bl_s", "unsigned_speed.png", "Unsigned speed (BL/s)", "bl_distance_valid"),
    ("signed_speed_bl_s", "signed_speed.png", "Signed speed (BL/s)", "bl_direction_valid"),
    ("step_distance_bl", "unsigned_distance.png", "Endpoint distance (BL)", "bl_distance_valid"),
    ("signed_distance_bl", "signed_distance.png", "Signed endpoint distance (BL)", "bl_direction_valid"),
    ("cumulative_path_distance_bl", "cumulative_unsigned_distance.png", "Cumulative valid distance (BL)", "bl_distance_valid"),
    ("cumulative_signed_distance_bl", "cumulative_signed_distance.png", "Cumulative signed distance (BL)", "bl_direction_valid"),
]


def estimate_body_lengths(records):
    """返回 {整数鱼ID: 参考体长信息}，不写文件、不修改输入。

    records 为 geometry 生成的记录列表。仅 valid=True 且 body_length_px
    为有限正数的帧参与中位数；NaN、无穷、0、负数及无效几何都不参与。
    无合格帧时参考体长为 NaN、count=0、valid=0，绝不默认 1 px/BL。
    """
    lengths = defaultdict(list)
    for record in records:
        values = lengths[int(record["id"])]
        length = pixel_speed._number(record.get("body_length_px"))
        if pixel_speed._valid(record.get("valid")) and math.isfinite(length) and length > 0:
            values.append(length)
    return {identity: dict(reference_body_length_px=float(median(values)) if values else math.nan,
                           reference_length_sample_count=len(values), body_length_valid=int(bool(values)),
                           body_length_reason="ok" if values else "no_valid_body_length")
            for identity, values in sorted(lengths.items())}


def _convert_rows(rows, references, summary=False):
    """只改单位，复制原运动质量信息；无体长时所有运动量（包括累计值）留空。"""
    converted = []
    flags = ("distance_summary_valid", "direction_summary_valid") if summary else ("distance_valid", "direction_valid")
    for source in rows:
        ref = references[int(source["id"])]
        length = ref["reference_body_length_px"]
        row = {}
        for key, value in source.items():
            if key in METRIC_NAMES:
                number = pixel_speed._number(value)
                row[METRIC_NAMES[key]] = number / length if ref["body_length_valid"] else math.nan
            else:
                row[key] = value
        row.update(ref)
        for flag in flags:
            row["bl_" + flag] = int(pixel_speed._valid(source[flag]) and ref["body_length_valid"])
        converted.append(row)
    return converted


def analyze_records(records, frame_steps=None):
    """纯计算入口：几何记录 -> (BL 区间记录列表, BL 每鱼汇总列表)。

    records：同一视频/视角的 geometry 记录。必需字段含 frame/id/time_s、
      cx/cy/area、valid/head_dx/head_dy，新增换算需要 body_length_px。
      frame/id 为整数，时间为秒，坐标和体长为像素；缺失体长不会假造结果。
    frame_steps：正整数列表；None 使用本脚本 FRAME_STEPS；例如 [1,5]。
    返回 rows：每行一个 (frame_step,id,frame) 窗口；各列见 FRAME_FIELDS。
      frame_range 仅列实际观测帧；首帧、缺帧和末尾不足窗口的处理继承 speed.py。
      无效数值在内存为 NaN，导出 CSV 时为空；有效静止的数值才是有效的0。
      累计值保留以前有效段的和；无有效区间时原累计和0不可当作有效静止。
    返回 summaries：每行一个 (frame_step,id)，各列见 SUMMARY_FIELDS。
      有符号/无符号均值各用其有效总距离除以有效时间，不做简单逐行算术平均。
      valid_*_time、coverage、intervals 保留原像素质量统计，BL有效性另看 bl_*。

    体长由全部 records 估计；1帧和5帧不会重新估计出不同的体长。
    函数不修改输入、不写文件、不修改 speed.py 的全局变量。
    """
    records = list(records)
    steps = pixel_speed._steps(FRAME_STEPS if frame_steps is None else frame_steps)
    # 先在原算法中计算所有运动结果（含判号和汇总），然后只做除法。
    pixel_rows = pixel_speed.calculate_speed(records, steps)
    pixel_summaries = pixel_speed.summarize(pixel_rows)
    references = estimate_body_lengths(records)
    return (_convert_rows(pixel_rows, references),
            _convert_rows(pixel_summaries, references, summary=True))


def plot_curves(rows, output, step):
    """与像素版相同的六类曲线和六面板总图；横轴仍为真实时间(s)。"""
    plt = pixel_speed._plt()
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), constrained_layout=True)
    for ax, (field, filename, label, flag) in zip(axes.ravel(), CURVES):
        pixel_speed._curve_panel(ax, rows, field, label, flag)
        single, single_ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
        pixel_speed._curve_panel(single_ax, rows, field, label, flag)
        single_ax.set_title(f"Step {step}: {label} | fixed non-overlapping windows")
        single.savefig(output / filename, dpi=150)
        plt.close(single)
    fig.suptitle(f"Step {step} | BL per fish: fixed median centerline length; gaps excluded")
    fig.savefig(output / "speed_curves.png", dpi=150)
    plt.close(fig)


def plot_summary(summaries, output, steps, comparison=False):
    """各鱼有/无符号平均速度和总距离；仅画 BL 汇总有效的柱形。"""
    plt = pixel_speed._plt()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    identities = sorted({r["id"] for r in summaries})
    metrics = [("mean_speed_bl_s", "Mean unsigned speed (BL/s)", "bl_distance_summary_valid"),
               ("mean_signed_speed_bl_s", "Mean signed speed (BL/s)", "bl_direction_summary_valid"),
               ("path_distance_bl", "Total valid endpoint distance (BL)", "bl_distance_summary_valid"),
               ("signed_distance_bl", "Total signed endpoint distance (BL)", "bl_direction_summary_valid")]
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
            ax.text(.5, .5, "No valid BL intervals", transform=ax.transAxes, ha="center")
    fig.suptitle("BL frame-step comparison: each step uses its own valid intervals" if comparison
                 else f"Step {steps[0]}: BL valid-interval summaries")
    fig.savefig(output / ("frame_steps_comparison.png" if comparison else "speed_summary.png"), dpi=160)
    plt.close(fig)


def draw_video(rows, records, meta, output, step):
    """保留原帧、质心、头轴及轨迹；采样终点显示带符号 BL/s 和 BL。

    非采样帧不延续上一窗口数值；无体长或方向无效显示 N/A。轨迹清空条件
    仍依据原像素运动质量，体长缺失不会改变图像上的位置或轨迹。
    缺少原图时跳过视频并返回原因；原图可用时输出全部帧，随后解码核对帧数。
    """
    paths = {int(k): v for k, v in (meta.get("image_paths") or {}).items()}
    if not paths:
        return "skipped_no_image_paths"
    fallback = sorted(set(paths) | {int(r["frame"]) for r in records})
    frame_ids = [int(f) for f in meta.get("frame_ids", fallback)]
    if not frame_ids:
        return "skipped_no_frames"
    missing = [f for f in frame_ids if not paths.get(f) or not Path(paths[f]).is_file()]
    if missing:
        return f"skipped_incomplete_image_paths: {len(missing)} missing frames; first frame IDs={missing[:10]}"
    width, height, fps = int(meta["width"]), int(meta["height"]), float(meta["fps"])
    if not math.isfinite(fps) or fps <= 0 or any(b <= a for a, b in zip(frame_ids, frame_ids[1:])):
        raise ValueError("视频 fps 或 frame_ids 无效。")
    if not isinstance(TRAIL_FRAMES, int) or isinstance(TRAIL_FRAMES, bool) or TRAIL_FRAMES < 1:
        raise ValueError("TRAIL_FRAMES 必须是正整数。")
    by_frame = defaultdict(list)
    for record in records:
        by_frame[int(record["frame"])].append(record)
    samples = {(r["frame"], r["id"]): r for r in rows}
    trails, previous = defaultdict(lambda: deque(maxlen=TRAIL_FRAMES)), {}
    path = output / "speed.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("无法创建 BL speed.mp4。")
    try:
        for frame in frame_ids:
            image = cv2.imdecode(np.fromfile(str(paths[frame]), np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise ValueError(f"frame={frame} 原图无法读取或尺寸不一致。")
            for record in by_frame[frame]:
                identity = int(record["id"])
                if not np.isfinite([pixel_speed._number(record.get("cx")), pixel_speed._number(record.get("cy"))]).all():
                    continue
                center = (int(round(record["cx"])), int(round(record["cy"])))
                sample = samples.get((frame, identity))
                trail = trails[identity]
                if previous.get(identity) != frame - 1 or sample is not None and not sample["distance_valid"]:
                    trail.clear()
                trail.append(center)
                previous[identity] = frame
                if len(trail) > 1:
                    cv2.polylines(image, [np.asarray(trail, np.int32)], False, (50, 220, 255), 2, cv2.LINE_AA)
                cv2.circle(image, center, 4, (0, 255, 255), -1)
                axis = np.array([pixel_speed._number(record.get("head_dx")), pixel_speed._number(record.get("head_dy"))])
                if pixel_speed._valid(record.get("valid")) and np.isfinite(axis).all() and np.linalg.norm(axis) > 1e-8:
                    axis /= np.linalg.norm(axis)
                    tip = tuple(np.rint(np.asarray(center) + 45 * axis).astype(int))
                    cv2.arrowedLine(image, center, tip, (255, 200, 30), 2, cv2.LINE_AA, tipLength=.2)
                if sample is None:
                    label = f"ID {identity} | not a step-{step} sample"
                elif sample["bl_direction_valid"]:
                    label = (f"ID {identity} [{sample['start_frame']}->{frame}] "
                             f"v={sample['signed_speed_bl_s']:+.3f} BL/s d={sample['signed_distance_bl']:+.4f} BL")
                else:
                    reason = sample["body_length_reason"] if not sample["body_length_valid"] else sample["quality_reason"]
                    label = f"ID {identity} [{sample['start_frame']}->{frame}] N/A: {reason}"
                # BL 标签更长，按实际文字宽度限制横坐标，避免靠右的鱼文字超出画面。
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .44, 1)
                x = max(2, min(center[0] - 80, width - label_width - 4))
                y = max(75, center[1] - 20)
                # 单次白字加黑底，避免部分 OpenCV 版本不同粗细文字叠画时产生重影。
                cv2.rectangle(image, (x - 2, y - label_height - 3),
                              (min(width - 1, x + label_width + 2), min(height - 1, y + baseline + 2)),
                              (0, 0, 0), -1)
                cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .44, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, f"SPEED BL step {step} | original frame {frame} | v:BL/s d:BL", (12, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "Fixed median body length per ID; numbers only at sample endpoints; lateral motion uses +", (12, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(image)
    finally:
        writer.release()
    capture, decoded = cv2.VideoCapture(str(path)), 0
    try:
        while capture.read()[0]:
            decoded += 1
    finally:
        capture.release()
    if decoded != len(frame_ids):
        raise RuntimeError(f"BL速度视频只有 {decoded}/{len(frame_ids)} 帧。")
    return "saved"


def export_results(records, meta, output: Path, frame_steps=None, save_video=True):
    """完整导出入口，返回与 analyze_records 相同的 (rows,summaries)。

    records：全部几何记录，含 body_length_px；不会被修改。
    meta：geometry 返回的来源信息、时间信息和可选原图路径。视频需要
      image_paths（帧号->原图绝对路径）、frame_ids、width/height、fps。
    output：本次输出的绝对路径，必须不存在或为空；不会覆盖已有结果。
    frame_steps：None 用本脚本设置；save_video：是否额外导出每步长视频。

    根目录：speed_frames_all_steps.csv（所有步长区间明细）、
      speed_summary_all_steps.csv（所有步长每鱼汇总）、frame_steps_comparison.png。
    step_N：speed_frames.csv、speed_summary.csv、六张单指标曲线、
      speed_curves.png 六面板图、speed_summary.png 四面板汇总图、可选 speed.mp4。
    run_info.json：运行状态、来源、单位、体长/样本数、计算规则、视频是否生成。
    CSV 为 UTF-8 BOM；列表列为 JSON 数组；NaN/无穷留空。图表略过无效指标。
    输出文件名与像素版一致，但放在 speed_BL 的独立批次中，单位看列名/标签。
    """
    steps = pixel_speed._steps(FRAME_STEPS if frame_steps is None else frame_steps)
    records = list(records)
    output = Path(output)
    if not output.is_absolute():
        raise ValueError("output 必须是绝对路径。")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"BL结果目录必须为空或不存在：{output}")
    rows, summaries = analyze_records(records, steps)
    references = estimate_body_lengths(records)
    output.mkdir(parents=True, exist_ok=True)
    info = dict(status="running", segmentation_dir=str(meta["segmentation_dir"]) if meta.get("segmentation_dir") else None,
                time_source=meta.get("time_source", "provided record time_s"), unit="BL and BL/s",
                coordinate_unit="pixel", area_unit="pixel^2", frame_steps=steps,
                head_hints=meta.get("head_hints", HEAD_HINTS), centroid_smoothing=False,
                use_filename_timestamps=meta.get("use_filename_timestamps", USE_FILENAME_TIMESTAMPS),
                save_video_requested=bool(save_video), trail_frames=TRAIL_FRAMES, fps=meta.get("fps"),
                frame_ids=[int(f) for f in meta.get("frame_ids", sorted({int(r["frame"]) for r in records} |
                          {int(f) for f in (meta.get("image_paths") or {})}))],
                max_area_change_ratio=pixel_speed.MAX_AREA_CHANGE_RATIO,
                max_adjacent_head_turn_deg=pixel_speed.MAX_HEAD_TURN_DEG,
                sign_dot_tolerance=pixel_speed.SIGN_DOT_TOLERANCE,
                max_explicit_missing_frames=pixel_speed.MAX_EXPLICIT_MISSING_FRAMES,
                body_length_method="Per-ID median of body_length_px from all valid geometry records with finite positive length; fixed across all frame steps",
                body_length_definition="Image-estimated head-to-tail centerline arc length, not bbox width or straight endpoint distance",
                body_lengths={str(k): {field: (None if isinstance(v, float) and not math.isfinite(v) else v)
                                      for field, v in ref.items()} for k, ref in references.items()},
                conversion="BL = pixel / reference_body_length_px; BL/s = (pixel/s) / reference_body_length_px",
                missing_length_policy="All BL quantities blank/NaN; body_length_valid=0; bl_*_valid=0; original pixel quality flags preserved",
                validity_definition="Original distance/direction flags, counts and time coverage refer to pixel motion; BL consumers must use bl_*_valid flags",
                signed_distance_definition="Full Euclidean endpoint distance * sign(dot(displacement, normalized sum of normalized endpoint head axes)); dot>=-tolerance is positive; then divide by reference length",
                forward_projection_definition="Diagnostic only; NOT signed_distance_bl",
                lateral_projection_definition="dot(displacement,[forward_y,-forward_x]) / reference length; positive points up when head points right",
                forward_backward_totals_definition="Positive signed distances and absolute negative signed distances; pure lateral intervals are positive",
                sampling="Fixed first-observation anchor per ID; non-overlapping; missing frames never reset anchor; incomplete tail excluded; entirely unobserved windows omitted",
                frame_range_definition="Actually observed frame IDs; missing_frame_ranges uses inclusive endpoints; large gaps not expanded into missing_frames",
                cumulative_definition="Sum of valid endpoint distances separately per step and ID, divided by fixed reference length; NOT sum of inner one-frame distances",
                average_definition="Corresponding valid total distance divided by corresponding valid time and fixed reference length",
                videos={}, observed_records=len(records), sample_rows=len(rows), summary_rows=len(summaries))
    info_path = output / "run_info.json"
    def save_info():
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    save_info()
    try:
        pixel_speed.save_csv(output / "speed_frames_all_steps.csv", rows, FRAME_FIELDS)
        pixel_speed.save_csv(output / "speed_summary_all_steps.csv", summaries, SUMMARY_FIELDS)
        for step in steps:
            folder = output / f"step_{step}"
            folder.mkdir()
            selected = [r for r in rows if r["frame_step"] == step]
            summary = [r for r in summaries if r["frame_step"] == step]
            pixel_speed.save_csv(folder / "speed_frames.csv", selected, FRAME_FIELDS)
            pixel_speed.save_csv(folder / "speed_summary.csv", summary, SUMMARY_FIELDS)
            plot_curves(selected, folder, step)
            plot_summary(summary, folder, [step])
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
    r"""本地运行入口。修改顶部绝对路径和参数后，用 sam3 环境运行本文件即可。

    Python 函数调用示例（无需运行 main，也无需重新进行模型分割）：

      from pathlib import Path
      from geometry import load_geometry, geometry_from_data
      from speed_BL import analyze_records, export_results

      # 方式1：已经有本项目分割结果目录。
      records, meta = load_geometry(
          Path(r"E:\Phd\temperature\sam3\top\outputs\20261114_095054_184431"),
          head_hints={}, use_filename_timestamps=True)
      rows, summaries = analyze_records(records, frame_steps=[1, 5])

      # 方式2：外部传入标准 {"data": [...]} 分割数据，按已有 geometry
      # 标准接口先提取几何；每项含 track_id/frame_id/segmentation 等。
      # fps/时间信息、RLE、model_info/video_info/view_info 等字段的校验
      # 仍由 geometry_from_data 统一负责，不在本模块再造一种输入格式。
      # records, meta = geometry_from_data(payload, head_hints={}, fps=30.0)
      # rows, summaries = analyze_records(records, frame_steps=[1, 5])

      # 如需同时导出CSV/图表/视频，调用以下函数；输出目录必须新建或为空。
      # rows, summaries = export_results(records, meta,
      #     Path(r"E:\Phd\temperature\sam3\top\analysis\speed_BL\my_new_run"),
      #     frame_steps=[1, 5], save_video=True)

    rows/summaries 是本模块明细/汇总列表，并非 main.analyze_behavior 的
    标准行为指标封装；main.py 仍按原方式返回原指标，本脚本独立使用。
    具体入参、返回字段、单位和无效值规则见以上函数注释与字段注释。
    """
    if not SEGMENTATION_DIR.is_absolute() or not OUTPUT_DIR.is_absolute():
        raise ValueError("SEGMENTATION_DIR 和 OUTPUT_DIR 必须使用绝对路径。")
    from geometry import load_geometry
    print(f"读取分割结果并提取中轴线：{SEGMENTATION_DIR}", flush=True)
    records, meta = load_geometry(SEGMENTATION_DIR, head_hints=HEAD_HINTS,
                                  use_filename_timestamps=USE_FILENAME_TIMESTAMPS)
    output = OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    print(f"按每鱼固定参考体长生成 BL 结果：{output}", flush=True)
    rows, summaries = export_results(records, meta, output, FRAME_STEPS, SAVE_VIDEO)
    print(f"BL 距离/速度分析完成：{output}\n步长={FRAME_STEPS}，采样记录={len(rows)}，汇总={len(summaries)}", flush=True)
    return output


if __name__ == "__main__":
    main()
