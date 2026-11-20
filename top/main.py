"""顶部视角综合行为入口：一次计算像素/BL速度距离、朝向和摆尾指标。

输入/输出的完整标准字段及校验规则只定义在 data_format.py。
本文件负责组织测量、选择指标与导出，不运行tracker或加载SAM3。
直接运行使用下方参数；函数调用使用显式入参，示例见analyze_behavior。
"""
from pathlib import Path
# ==================== 本机直接运行时修改这里 ====================
SEGMENTATION_DIR = Path(r"E:\Phd\temperature\sam3\top\output\tracking")
OUTPUT_DIR = Path(r"E:\Phd\temperature\sam3\top\output\behavior")
# 采样步长统一在 behavior/speed.py 顶部设置。综合入口与速度独立入口共用 FRAME_STEP。
HEAD_HINTS = {}  # 例如 {0: "left"}，只初始化该鱼首次成功识别的头向。
USE_FILENAME_TIMESTAMPS = True  # 兼容旧分割时，优先解析原图文件名中的 _t 秒数。
SAVE_VIDEO = True  # 原图齐全时，速度输出像素/BL两段视频，摆尾和朝向各输出一段。
# Python 函数调用可传 frame_step=5，单次覆盖默认步长，同时应用于像素与BL。
# ==============================================================

from data_format import make_behavior_indicator, convert_segmentation, prepare_output_directory
import json
import math

from behavior import geometry, speed, tail, towards



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
            output.append(make_behavior_indicator(*common, name, row[key], "pixel", frame_range))
        heading = headings[(row["frame"], row["id"])]
        for name, value, valid, reason in (
            ("heading_tilt", heading["tilt_deg"], heading["valid"], heading["quality_reason"]),
            ("tail_angle", row["tail_angle_deg"], row["valid"], row["quality_reason"]),
        ):
            output.append(make_behavior_indicator(*common, name, value, "degree", frame_range,
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
            output.append(make_behavior_indicator(row["id"], row["frame"], name, row[key], unit,
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
            output.append(make_behavior_indicator(identity, end_frame, name, summary[key], unit,
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
        output.append(make_behavior_indicator(row["id"], end_frame,
                                 "tail_beat_frequency", row["frequency_hz"], "Hz", frames,
                                 status, "" if status == "ok" else quality))
    return {"data": output}


def _analyze_geometry(records, meta, frame_step, output_dir, save_video):
    """供标准输入和本地文件入口共用，保证一轮分析只提取一次中轴。"""
    # speed中的同一条记录同时有像素列、BL列和各自有效性。
    # 参考体长在速度模块内按每鱼本批有效几何估计，所有单位使用同一批窗口。
    speed_rows, speed_summary = speed.analyze_records(records, frame_step=frame_step)
    heading_rows = towards.prepare_rows(records)
    frequency_rows = tail.analyze_tail(records, meta)
    result = _standard_results(records, meta, speed_rows, speed_summary, heading_rows, frequency_rows)
    # 标准接口仍是一条记录表达一个指标和一个单位。共享明细可构造两种单位，
    # 其中BL额外带参考体长；输入、公式和转换过程不在此重复计算。
    result["data"].extend(_bl_standard_results(speed_rows, speed_summary))
    if output_dir is not None:
        root = Path(output_dir)
        if not root.is_absolute():
            raise ValueError("output_dir 必须使用绝对路径。")
        # 固定目录：main 与单独脚本共用各指标子目录，不再追加 all 或时间戳。
        # 先撤下上一轮综合文件；各模块随后只清理自己的旧输出。
        destination = prepare_output_directory(root, ["behavior_results.json", "run_info.json"])
        info = {
            "status": "running", "video_id": meta.get("video_id"), "view_info": "top",
            "frame_count": meta["frame_count"], "observations": len(records),
            "frame_step": frame_step, "time_source": meta["time_source"],
            "segmentation_dir": meta.get("segmentation_dir"),
            "head_hints": meta.get("head_hints", {}),
            "use_filename_timestamps": meta.get("use_filename_timestamps"),
            "signed_distance_definition": "Euclidean distance; sign from mean endpoint head direction; lateral is positive",
            "warnings": meta.get("warnings", []),
            "motion_units": ["pixel", "pixel/s", "BL", "BL/s"],
            "body_length_method": "Each ID uses its own fixed median valid centerline length, independent of frame_step",
            "indicator_identity": "track_id + frame_id + behavior_indicator + unit + frame_step",
            "behavior_modules": ["speed", "tail", "towards"],
            "speed_csv_layout": "One row per window or fish, with pixel and BL columns side by side",

        }
        try:
            tail.export_results(records, meta, destination / "tail", save_video=save_video,
                                frequency_rows=frequency_rows)
            towards.export_results(records, meta, destination / "towards", save_video=save_video,
                                   rows=heading_rows)
            speed.export_results(records, meta, destination / "speed", frame_step=frame_step,
                                 save_video=save_video, results=(speed_rows, speed_summary))
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


def analyze_behavior(data, *, frame_step=None, head_hints=None, fps=None,
                     output_dir=None, save_video=False, frame_steps=None):
    """通过标准分割数据，返回包含像素/BL运动量、朝向和摆尾的标准结果。

    Python调用：
        from main import analyze_behavior
        result = analyze_behavior(standard_data, frame_step=5)
        valid = [item for item in result["data"] if item["status"] == "ok"]

    data：同一视频/顶部视角的完整标准批次；列表可乱序，也接受按帧有序迭代器。
          字段含义、时间约定、RLE和返回结构统一见data_format.py顶部说明。
    frame_step：一个正整数，单位为帧间隔数。省略时使用behavior/speed.py的FRAME_STEP。
                像素与BL共用本次步长。旧关键字frame_steps只接受单元素序列。
    head_hints：{鱼ID:"left/right/up/down"}，用于首次成功判头；None使用几何默认值。
    fps：输入缺少帧率时补充的正数，单位帧/秒；不得与输入提供的帧率冲突。
    output_dir：None仅返回数值；给定绝对路径时直接导出到该目录，覆盖旧结果。
                speed/tail/towards三个目录分别存放速度、摆尾和朝向结果。
                speed中两张CSV同时包含像素与BL列；同一输出目录重跑会覆盖。
    save_video：导出时是否生成视频，需要全部原图路径，默认False。
                速度生成speed_px.mp4和speed_bl.mp4，分别显示像素和BL单位。

    一次调用分析本批全部记录，不跨调用积累历史；帧/周期不足返回占位结果。
    BL按每条鱼本批有效体长的中位数换算，标尺由本批几何记录决定。
    标准结果含同名的像素/BL指标，务必结合unit与frame_step区分。
    """
    step = speed.resolve_frame_step(frame_step, frame_steps=frame_steps)
    records, meta = geometry.geometry_from_data(data, head_hints=head_hints, fps=fps)
    return _analyze_geometry(records, meta, step, output_dir, save_video)


def analyze_segmentation(segmentation_dir, *, frame_step=None, head_hints=None,
                         output_dir=None, save_video=False, use_filename_timestamps=True,
                         frame_steps=None):
    """分析一个已完成的本地分割目录，兼容旧native_masks及新标准JSONL。

    示例：
        result = analyze_segmentation(
            r"E:\\Phd\\temperature\\sam3\\top\\output\\tracking",
            frame_step=1,
            output_dir=r"E:\\Phd\\temperature\\sam3\\top\\output\\behavior",
            save_video=True)
    文件入口与analyze_behavior复用同一套测量/公式，返回字段完全一致。
    """
    step = speed.resolve_frame_step(frame_step, frame_steps=frame_steps)
    records, meta = geometry.load_geometry(Path(segmentation_dir), head_hints,
                                           use_filename_timestamps)
    return _analyze_geometry(records, meta, step, output_dir, save_video)



def _bl_standard_results(rows, summaries):
    """从合并的明细和汇总中读取BL列，构造单位为BL或BL/s的标准记录。

    同名behavior_indicator用unit区分，记录标识需含unit/frame_step。
    frame_range：区间为实际观测帧，汇总为相应BL有效区间的帧并集。
    无有效体长用0占位+invalid；不足窗口且体长可用为insufficient_frames。
    参考长度无法估计时用JSON null，绝不输出NaN或借用其他鱼的体长。
    """
    output, groups = [], {}
    def status(row, signed):
        if not row["body_length_valid"]:
            return "invalid", "no_valid_body_length"
        return _interval_status(row, signed)
    def add(source, name, key, unit, frames, state, reason, end):
        item = make_behavior_indicator(source["id"], end, name, source[key], unit, frames,
                          state, reason, source["frame_step"], reference=source)
        output.append(item)
    for row in rows:
        groups.setdefault((row["frame_step"], row["id"]), []).append(row)
        for name,key,unit,signed in (
            ("speed","speed_bl_s","BL/s",False),
            ("velocity","signed_speed_bl_s","BL/s",True),
            ("step_distance","step_distance_bl","BL",False),
            ("signed_distance","signed_distance_bl","BL",True),
        ):
            state,reason=status(row,signed)
            add(row,name,key,unit,row["frame_range"],state,reason,row["frame"])
    for summary in summaries:
        group=groups[(summary["frame_step"],summary["id"])]
        end=max(r["frame"] for r in group)
        for name,key,unit,signed in (
            ("total_distance","path_distance_bl","BL",False),
            ("total_signed_distance","signed_distance_bl","BL",True),
            ("mean_speed","mean_speed_bl_s","BL/s",False),
            ("mean_velocity","mean_signed_speed_bl_s","BL/s",True),
        ):
            valid=[r for r in group if r["bl_direction_valid" if signed else "bl_distance_valid"]]
            frames=sorted({f for r in (valid or group) for f in r["frame_range"]})
            if valid:
                state,reason="ok",""
            elif not summary["body_length_valid"]:
                state,reason="invalid","no_valid_body_length"
            else:
                state="insufficient_frames" if all(status(r,signed)[0]=="insufficient_frames" for r in group) else "invalid"
                reason="no_valid_intervals"
            add(summary,name,key,unit,frames,state,reason,end)
    return output


def main():
    """本机入口。路径与显示参数在本文件顶部，步长读取behavior/speed.py的FRAME_STEP。"""
    return analyze_segmentation(SEGMENTATION_DIR, head_hints=HEAD_HINTS, output_dir=OUTPUT_DIR,
                                save_video=SAVE_VIDEO, use_filename_timestamps=USE_FILENAME_TIMESTAMPS)

if __name__ == "__main__":
    main()
