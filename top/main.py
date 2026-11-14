"""顶部视角鱼体行为分析的统一 Python 入口。

运行顺序：run_sam3.py 完成整段分割 → 本文件读取标准记录 → 测量几何 → 分析指标。
直接调用 analyze_behavior 时无需加载 SAM3 或读取原图，掩膜来自调用方。
请重点阅读 analyze_behavior 的字段协议、示例和 frame_range/status 约定。
"""
from pathlib import Path

# ==================== 本机直接运行时修改这里 ====================
SEGMENTATION_DIR = Path(r"E:\Phd\temperature\sam3\top\outputs\20261114_095054_184431")
OUTPUT_DIR = Path(r"E:\Phd\temperature\sam3\top\analysis\all")
HEAD_HINTS = {}  # 例如 {0: "left"}，只初始化该鱼首次成功识别的头向。
USE_FILENAME_TIMESTAMPS = True  # 兼容旧分割时，优先解析原图文件名中的 _t 秒数。
SAVE_VIDEO = True  # 本地模式有原图，可生成三个分析模块的视频。
# 帧间隔统一在 speed.py 的 FRAME_STEPS = [1, 5] 中设置。
# Python 调用也可显式传 frame_steps=[1, 5, 10]，仅对这次调用生效。
# ==============================================================

import gzip
import json
import math
from datetime import datetime

import geometry
import speed
import tail
import towards


def _indicator(track_id, frame_id, name, value, unit, frames,
               status="ok", reason="", frame_step=None):
    """创建一条 JSON 可序列化的指标。缺测用 0 + 非 ok 状态，杜绝 NaN/Infinity。"""
    number = float(value)
    if not math.isfinite(number):
        number = 0.0
        if status == "ok":
            status, reason = "invalid", reason or "non_finite_result"
    row = {
        "track_id": int(track_id),
        "frame_id": int(frame_id),
        "behavior_indicator": name,
        "behavior_value": number,
        "unit": unit,
        "frame_range": [int(frame) for frame in frames],
        "status": status,
        "reason": reason,
    }
    if frame_step is not None:
        row["frame_step"] = int(frame_step)
    return row


def _interval_status(row, signed):
    """距离有效性与方向有效性分开，有效的零值保持 status=ok。"""
    if row["direction_valid" if signed else "distance_valid"]:
        return "ok", ""
    reason = row["quality_reason"] or "invalid_interval"
    if reason in ("first_observation", "incomplete_window"):
        return "insufficient_frames", reason
    return "invalid", reason


def _standard_results(records, meta, speed_rows, speed_summary, heading_rows, frequency_rows):
    """把各算法的数值结果转换为一套字段。这里不重新计算任何生物学指标。"""
    output = []
    headings = {(row["frame"], row["id"]): row for row in heading_rows}
    for row in records:
        common = (row["id"], row["frame"])
        frame_range = [row["frame"]]
        for name, key in (("position_x", "cx"), ("position_y", "cy")):
            output.append(_indicator(*common, name, row[key], "pixel", frame_range))
        heading = headings[(row["frame"], row["id"])]
        for name, value, valid, reason in (
            ("heading_tilt", heading["tilt_deg"], heading["valid"], heading["quality_reason"]),
            ("tail_angle", row["tail_angle_deg"], row["valid"], row["quality_reason"]),
        ):
            output.append(_indicator(*common, name, value, "degree", frame_range,
                                     "ok" if valid else "invalid", "" if valid else reason))

    # 每个区间输出两种速度和两种距离。位置/角度记录不带 frame_step。
    for row in speed_rows:
        for name, key, unit, signed in (
            ("speed", "speed_px_s", "pixel/s", False),
            ("velocity", "signed_speed_px_s", "pixel/s", True),
            ("step_distance", "step_distance_px", "pixel", False),
            ("signed_distance", "signed_distance_px", "pixel", True),
        ):
            status, reason = _interval_status(row, signed)
            output.append(_indicator(row["id"], row["frame"], name, row[key], unit,
                                     row["frame_range"], status, reason, row["frame_step"]))

    # 汇总 frame_range 只列真正参与该项统计的有效区间中的已观测帧。
    # 这样有符号与无符号指标的有效时间和帧范围可以各自追溯。
    groups = {}
    for row in speed_rows:
        groups.setdefault((row["frame_step"], row["id"]), []).append(row)
    for summary in speed_summary:
        step, identity = summary["frame_step"], summary["id"]
        group = groups[(step, identity)]
        end_frame = max(row["frame"] for row in group)
        for name, key, unit, signed in (
            ("total_distance", "path_distance_px", "pixel", False),
            ("total_signed_distance", "signed_distance_px", "pixel", True),
            ("mean_speed", "mean_speed_px_s", "pixel/s", False),
            ("mean_velocity", "mean_signed_speed_px_s", "pixel/s", True),
        ):
            valid_rows = [row for row in group if row["direction_valid" if signed else "distance_valid"]]
            if valid_rows:
                frames = sorted({frame for row in valid_rows for frame in row["frame_range"]})
                status, reason = "ok", ""
            else:
                frames = sorted({frame for row in group for frame in row["frame_range"]})
                only_short = all(_interval_status(row, signed)[0] == "insufficient_frames" for row in group)
                status = "insufficient_frames" if only_short else "invalid"
                reason = "no_valid_intervals"
            output.append(_indicator(identity, end_frame, name, summary[key], unit,
                                     frames, status, reason, step))

    for row in frequency_rows:
        frames = row.get("frame_range", [])
        if not frames:
            frames = [record["frame"] for record in records if record["id"] == row["id"]]
        quality = row["frequency_quality"]
        status = "ok" if quality == "ok" else "invalid"
        if quality in ("too_short", "insufficient_complete_cycles"):
            status = "insufficient_frames"
        end_frame = row.get("end_frame")
        if end_frame is None:
            end_frame = frames[-1]
        output.append(_indicator(row["id"], end_frame,
                                 "tail_beat_frequency", row["frequency_hz"], "Hz", frames,
                                 status, "" if status == "ok" else quality))
    return {"data": output}


def _analyze_geometry(records, meta, frame_steps, output_dir, save_video):
    """供标准输入和本地文件入口共用，保证一轮分析只提取一次中轴。"""
    speed_rows = speed.calculate_speed(records, frame_steps=frame_steps)
    speed_summary = speed.summarize(speed_rows)
    heading_rows = towards.prepare_rows(records)
    frequency_rows = tail.analyze_tail(records, meta)
    result = _standard_results(records, meta, speed_rows, speed_summary, heading_rows, frequency_rows)
    if output_dir is not None:
        root = Path(output_dir)
        if not root.is_absolute():
            raise ValueError("output_dir 必须使用绝对路径。")
        # 每次生成新的子目录，保留完整的运行记录便于复核。
        destination = root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination.mkdir(parents=True, exist_ok=False)
        info = {
            "status": "running", "video_id": meta.get("video_id"), "view_info": "top",
            "frame_count": meta["frame_count"], "observations": len(records),
            "frame_steps": list(frame_steps), "time_source": meta["time_source"],
            "segmentation_dir": meta.get("segmentation_dir"),
            "head_hints": meta.get("head_hints", {}),
            "use_filename_timestamps": meta.get("use_filename_timestamps"),
            "signed_distance_definition": "Euclidean distance; sign from mean endpoint head direction; lateral is positive",
            "warnings": meta.get("warnings", []),
        }
        try:
            tail.export_results(records, meta, destination / "tail", save_video=save_video,
                                frequency_rows=frequency_rows)
            towards.export_results(records, meta, destination / "towards", save_video=save_video,
                                   rows=heading_rows)
            speed.export_results(records, meta, destination / "speed", frame_steps=frame_steps,
                                 save_video=save_video)
            with (destination / "behavior_results.json").open("w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            info.update(status="complete", indicator_records=len(result["data"]))
        except BaseException as exc:
            info.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            (destination / "run_info.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"完整分析结果已保存到 {destination}")
    return result


def analyze_behavior(data, *, frame_steps=None, head_hints=None, fps=None,
                     output_dir=None, save_video=False):
    """通过标准数据计算顶部视角鱼体行为，返回 {"data": [标准指标记录, ...]}。

    Python 调用方法（调用方已经持有上游分割结果时）：
        from main import analyze_behavior
        result = analyze_behavior(standard_data, frame_steps=[1, 5])
        for item in result["data"]:
            if item["status"] == "ok":
                print(item["track_id"], item["behavior_indicator"], item["behavior_value"])

    standard_data = {"data": [record1, record2, ...]}，每个 record 是某帧的一条鱼：
      track_id       整数跟踪ID；0也合法。跨帧同ID必须是同一条鱼。
      frame_id       非负整数原始帧号；保留原编号。步长按帧号差计算。
      model_info     模型信息字典，例如 {"name":"SAM3"}，不会进入指标公式。
      video_info     视频信息字典。同一批属于一个视频，字段约定如下：
          video_id          视频标识字符串，建议明确提供以便追溯。
          fps               正帧率；也可通过本函数fps参数提供。两者需一致。
          timestamp_seconds 可选采集秒数。整批全有或全无，同帧时间一致。
                            全有时使用实际时间差，全无时使用帧号差/fps。
          image_path        可选原图绝对路径，仅用于导出叠加视频。
      view_info      本项目使用 "top"；鱼头朝上/朝下以图像方向定义。
      category_id    整数类别，当前鱼类约定1。
      segmentation   {"size":[height,width], "counts":[整数游程,...]}。
                     COCO按列展开，从背景0开始，前景1/背景0轮流计数。
                     所有游程之和等于height*width；首像素为前景则首游程0。
                     也兼容COCO压缩counts字符串，以便接入既有结果。
      area           掩膜前景像素数。分析用解码掩膜的实测面积。
      bbox           [左上角x,左上角y,宽,高]，单位像素。

    普通JSON的data为列表，可乱序，由内部排序。Python大文件适配器还可传入
    按帧有序的记录迭代器，每次只解码一个掩膜，分析保存数值和51点中轴。
    本函数一次分析传入的完整批次。跨两次函数调用不隐式保留历史。

    参数说明：
      frame_steps  帧间隔，例如[1,5]。None采用speed.FRAME_STEPS。
                   每鱼以首次帧为固定锚；5表示0→5、5→10，首区间覆盖6张图。
      head_hints   {track_id:"left/right/up/down"}，用于初次成功判头。
      fps          输入未提供帧率时显式补充，单位帧/秒。
      output_dir   None只返回结果；绝对路径则创建新子目录导出CSV/图/JSON。
      save_video   仅当导出且有全部原图路径时生成视频，默认False。

    每个返回记录有以下字段，所有数值均可严格JSON序列化：
      track_id/frame_id 鱼编号与结果所属帧。区间归末帧，整段汇总归最后观测帧。
      behavior_indicator 指标名：position_x/position_y（像素质心）；heading_tilt
          （左右水平0、向上正、向下负）；tail_angle（有符号头尾夹角）；
          speed（无符号速度）；velocity（完整距离赋正负后的速度）；
          step_distance/signed_distance（区间无符号/有符号距离）；
          total_distance/total_signed_distance（各步长有效区间累计）；
          mean_speed/mean_velocity（各自有效距离和除以有效时间和）；
          tail_beat_frequency（每段连续有效角度的摆尾频率）。
      behavior_value 单一数值。无效/不足时为0，正常计算为0时status仍为ok。
      unit           "pixel"、"pixel/s"、"degree"或"Hz"。
      frame_range    实际参与测量/检查的帧号列表；不足时列累计的已有帧。
                     区间距离使用首末质心，中间帧用于完整性及质量检查。
                     汇总仅列有效区间涉及帧，无有效区间时列已有帧。
      frame_step     仅运动指标提供，表示1帧/5帧等；单帧角度和频率无此字段。
      status         ok / insufficient_frames / invalid，后两者的0为占位值。
      reason         状态原因，ok时为空。统计时必须先筛选status=="ok"。

    符号规则：点积只判断前后，大小始终为完整欧氏距离。前进/纯侧向取正，
    后退取负。前向采用两端归一化头轴的圆平均。缺帧、异常时间或质量问题
    会排除相应区间，不跨缺口补算。5帧端点距离可小于5个1帧距离之和。
    频率保留现有周期和质量规则；帧数/完整周期不足时立即返回占位记录。
    """
    steps = list(speed.FRAME_STEPS if frame_steps is None else frame_steps)
    records, meta = geometry.geometry_from_data(data, head_hints=head_hints, fps=fps)
    return _analyze_geometry(records, meta, steps, output_dir, save_video)


def analyze_segmentation(segmentation_dir, *, frame_steps=None, head_hints=None,
                         output_dir=None, save_video=False, use_filename_timestamps=True):
    """分析一个已完成的本地分割目录，兼容旧native_masks及新标准JSONL。

    示例：
        result = analyze_segmentation(
            r"E:\\Phd\\temperature\\sam3\\top\\outputs\\reference_150frames",
            frame_steps=[1,5],
            output_dir=r"E:\\Phd\\temperature\\sam3\\top\\analysis\\all",
            save_video=True)
    文件入口与analyze_behavior复用同一套测量/公式，返回字段完全一致。
    """
    steps = list(speed.FRAME_STEPS if frame_steps is None else frame_steps)
    records, meta = geometry.load_geometry(Path(segmentation_dir), head_hints,
                                           use_filename_timestamps)
    return _analyze_geometry(records, meta, steps, output_dir, save_video)


def convert_segmentation(segmentation_dir, output_file):
    """旧/新分割目录转标准JSONL(.jsonl或.jsonl.gz)，每行是单帧的{"data":[...]}。

    这是文件传输格式，逐行读取后将各行data合并/顺序迭代即可调用analyze_behavior。
    保留各条鱼独立掩膜，重叠区域也分别保留。输出路径必须绝对且尚不存在。
    """
    destination = Path(output_file)
    if not destination.is_absolute():
        raise ValueError("output_file 必须为绝对路径。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if destination.suffix == ".gz" else open
    stream = geometry.iter_standard_records(Path(segmentation_dir))
    # 使用完整帧清单，空鱼帧也输出{"data":[]}，使文件行数与frames.csv一致。
    # 用一个前瞻实例逐帧写入，内存只保存当前帧，转换同样经过统一RLE校验。
    records = iter(stream)
    current = next(records, None)
    with opener(destination, "xt", encoding="utf-8") as sink:
        for frame in stream.meta["frame_ids"]:
            batch = []
            while current is not None and current["frame_id"] == frame:
                mask = geometry.decode_rle(current["segmentation"])
                current["segmentation"] = {"size": list(mask.shape),
                                           "counts": geometry.encode_uncompressed_rle(mask)}
                current["area"] = float(mask.sum())
                batch.append(current)
                current = next(records, None)
            sink.write(json.dumps({"data": batch}, ensure_ascii=False, allow_nan=False) + "\n")
        if current is not None:
            raise ValueError("标准转换遇到清单之外的帧。")
    return destination


def main():
    """本机入口：只需设置顶部绝对路径和speed.py的FRAME_STEPS，运行即可。"""
    analyze_segmentation(SEGMENTATION_DIR, head_hints=HEAD_HINTS, output_dir=OUTPUT_DIR,
                         save_video=SAVE_VIDEO, use_filename_timestamps=USE_FILENAME_TIMESTAMPS)


if __name__ == "__main__":
    main()
