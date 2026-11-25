"""侧面视角模块，维护负责人：chenxi。
所有路径默认指向合并项目；可用命令行参数覆盖。只操作自身 Side 目录。
"""
from __future__ import annotations
from pathlib import Path
import sys, argparse, csv, json, math, os, time, io, zipfile
import numpy as np
import pandas as pd
import cv2
SIDE_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SIDE_ROOT))
from common.segmentation_utils import *
from scipy.signal import butter,find_peaks,savgol_filter,sosfiltfilt
INPUT=Path(r"E:\Phd\FishBehavior\Side\Tracker\Output\gill\result.csv")
OUTPUT=Path(r"E:\Phd\FishBehavior\Side\Behavior\Output\gill")
BAND_LOW_HZ=.60
BAND_HIGH_HZ=1.50
PEAK_PROMINENCE=.05
PEAK_DISTANCE_S=.25
DEFAULT_VIDEO=Path(r"E:\Phd\FishBehavior\Side\Tracker\Input\gill\video.mp4")
DEFAULT_ZIP=Path(r"E:\Phd\FishBehavior\Side\Tracker\Input\gill")
def interpolate(values: np.ndarray) -> np.ndarray:
    """用相邻有效值线性插值补齐 NaN/Inf；有效点少于两个时返回全零。"""
    y = np.asarray(values, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return np.zeros_like(y)
    index = np.arange(len(y))
    return np.interp(index, index[finite], y[finite])

def bandpass(values: np.ndarray, fps: float) -> np.ndarray:
    """先补齐缺失值，再对信号做零相位四阶巴特沃斯带通滤波。"""
    y = interpolate(values)
    sos = butter(
        4,
        [BAND_LOW_HZ, BAND_HIGH_HZ],
        btype="bandpass",
        fs=fps,
        output="sos",
    )
    return sosfiltfilt(sos, y)

def robust_z(values: np.ndarray) -> np.ndarray:
    """使用中位数和 IQR 做稳健标准化，降低异常振幅对归一化的影响。"""
    y = np.asarray(values, dtype=float)
    median = float(np.median(y))
    q25, q75 = np.percentile(y, [25, 75])
    scale = float(q75 - q25) or float(np.std(y)) or 1.0
    return (y - median) / scale

def peak_candidates(values: np.ndarray, fps: float, raw: bool) -> list[dict[str, float | int]]:
    """检测峰值候选，并返回峰值帧索引和显著度。

    raw=True 时直接在插值后的原始曲线上寻找峰；raw=False 时先做带通滤波。
    两种模式都先稳健归一化，再用 Savitzky-Golay 平滑以抑制单帧噪声。
    """
    base = interpolate(values) if raw else bandpass(values, fps)
    z = robust_z(base)
    smooth = savgol_filter(z, 11, 3)
    peaks, properties = find_peaks(
        smooth,
        distance=max(1, int(round(0.10 * fps))),
        prominence=0.001,
    )
    result: list[dict[str, float | int]] = []
    for peak, prominence in zip(peaks, properties["prominences"]):
        index = int(peak)
        if raw:
            # 平滑会轻微移动峰位，因此在原始曲线附近重新定位局部最大值。
            lo = max(0, index - 6)
            hi = min(len(z), index + 7)
            index = lo + int(np.argmax(z[lo:hi]))
        result.append(
            {
                "index": index,
                "prominence": round(float(prominence), 6),
            }
        )
    return result

def selected_peaks(candidates: list[dict[str, float | int]], fps: float) -> list[dict[str, float | int]]:
    """按显著度和最小时间间隔筛选峰值，最终按帧顺序返回。"""
    minimum = max(1, int(round(PEAK_DISTANCE_S * fps)))
    eligible = sorted(
        [item for item in candidates if item["prominence"] >= PEAK_PROMINENCE],
        key=lambda item: item["prominence"],
        reverse=True,
    )
    chosen: list[dict[str, float | int]] = []
    for item in eligible:
        # 从强到弱贪心保留峰值，避免同一鳃动周期被重复计数。
        if all(abs(int(other["index"]) - int(item["index"])) >= minimum for other in chosen):
            chosen.append(item)
    return sorted(chosen, key=lambda item: int(item["index"]))

def nullable(values: np.ndarray, digits: int = 3) -> list[float | None]:
    """把 NumPy 曲线转换为可直接写入 JSON 的列表，非有限值转为 None。"""
    return [
        round(float(value), digits) if np.isfinite(value) else None
        for value in values
    ]

def build_visualization_data(metrics: pd.DataFrame, video_info: dict[str, float | int]) -> dict[str, object]:
    """把逐帧灰度统计整理成可视化页面使用的鱼级曲线和峰值数据。"""
    fps = float(video_info["fps"])
    fish_records: list[dict[str, object]] = []
    for fish_id, group in metrics.groupby("fish_id", sort=True):
        group = group.sort_values("frame").reset_index(drop=True)
        first_frame = int(group.frame.min())
        last_frame = int(group.frame.max())
        mean = np.full(last_frame, np.nan)
        variance = np.full(last_frame, np.nan)
        valid = np.zeros(last_frame, dtype=bool)
        boxes: list[list[float] | None] = [None] * last_frame
        for row in group.itertuples(index=False):
            index = int(row.frame) - 1
            mean[index] = float(row.gray_mean)
            variance[index] = float(row.gray_variance)
            valid[index] = bool(row.valid_tracking)
            boxes[index] = [int(row.x1), int(row.y1), int(row.x2), int(row.y2), bool(row.valid_tracking)]

        # 同时保留原始曲线、带通曲线和平滑曲线，便于页面排查峰值来源。
        filtered_mean = robust_z(bandpass(mean, fps))
        filtered_variance = robust_z(bandpass(variance, fps))
        smoothed_mean = savgol_filter(filtered_mean, 11, 3)
        smoothed_variance = savgol_filter(filtered_variance, 11, 3)
        mean_candidates = peak_candidates(mean, fps, raw=False)
        variance_candidates = peak_candidates(variance, fps, raw=False)
        mean_raw = peak_candidates(mean, fps, raw=True)
        variance_raw = peak_candidates(variance, fps, raw=True)
        mean_peaks = selected_peaks(mean_candidates, fps)
        variance_peaks = selected_peaks(variance_candidates, fps)
        mean_valid = mean[np.isfinite(mean)]
        variance_valid = variance[np.isfinite(variance)]
        fish_records.append(
            {
                "id": int(fish_id),
                "sourceTrackId": int(fish_id),
                "firstFrame": first_frame,
                "lastFrame": last_frame,
                "validFrames": int(valid.sum()),
                "baselineMean": float(np.median(mean_valid)),
                "baselineVariance": float(np.median(variance_valid)),
                "minMean": float(np.min(mean_valid)),
                "maxMean": float(np.max(mean_valid)),
                "minVariance": float(np.min(variance_valid)),
                "maxVariance": float(np.max(variance_valid)),
                "mean": nullable(mean),
                "variance": nullable(variance),
                "filteredMean": nullable(filtered_mean),
                "filteredVariance": nullable(filtered_variance),
                "smoothedMean": nullable(smoothed_mean),
                "smoothedVariance": nullable(smoothed_variance),
                "meanPeakCandidates": mean_candidates,
                "variancePeakCandidates": variance_candidates,
                "meanRawPeakCandidates": mean_raw,
                "varianceRawPeakCandidates": variance_raw,
                "meanPeakIndices": [item["index"] for item in mean_peaks],
                "variancePeakIndices": [item["index"] for item in variance_peaks],
                "meanPeakCount": len(mean_peaks),
                "variancePeakCount": len(variance_peaks),
                "boxes": boxes,
                "valid": valid.tolist(),
            }
        )
    return {
        "meta": {
            **video_info,
            "frameCount": int(video_info["frame_count"]),
            "fps": fps,
            "bandLowHz": BAND_LOW_HZ,
            "bandHighHz": BAND_HIGH_HZ,
            "regionType": "box",
            "source": str(DEFAULT_VIDEO) + " + " + str(DEFAULT_ZIP),
        },
        "fish": fish_records,
    }

def export_peak_tables(data: dict[str, object], output_dir: Path) -> pd.DataFrame:
    """导出每条鱼、每条曲线的峰值数量和峰值发生位置。"""
    fps = float(data["meta"]["fps"])
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for fish in data["fish"]:
        duration = (int(fish["lastFrame"]) - int(fish["firstFrame"]) + 1) / fps
        curves = (
            ("raw_mean", "均值", "meanRawPeakCandidates"),
            ("raw_variance", "方差", "varianceRawPeakCandidates"),
            ("filtered_mean", "带通均值", "meanPeakCandidates"),
        )
        for curve, label, key in curves:
            peaks = selected_peaks(fish[key], fps)
            # 汇总表按“鱼 + 曲线”聚合，明细表保留每一个峰的时间与显著度。
            summary_rows.append(
                {
                    "fish_id": int(fish["id"]),
                    "curve": curve,
                    "metric_label": label,
                    "duration_s": duration,
                    "peak_count": len(peaks),
                    "peak_rate_per_min": len(peaks) * 60.0 / duration,
                    "peak_prominence_min": PEAK_PROMINENCE,
                    "peak_distance_min_s": PEAK_DISTANCE_S,
                }
            )
            for number, peak in enumerate(peaks, start=1):
                index = int(peak["index"])
                detail_rows.append(
                    {
                        "fish_id": int(fish["id"]),
                        "curve": curve,
                        "metric_label": label,
                        "peak_number": number,
                        "frame": index + 1,
                        "time_s": index / fps,
                        "prominence": float(peak["prominence"]),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)
    summary.to_csv(
        output_dir / "peak_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    detail.to_csv(
        output_dir / "peak_details.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary

def _frequency_from_window(
    values_by_frame: dict[int, float],
    start_frame: int,
    end_frame: int,
    fps: float,
    min_distance_seconds: float,
    prominence: float,
    minimum_coverage: float,
) -> tuple[float, int, list[int]]:
    """在指定帧窗口内计算鳃动频率，并返回覆盖率与实际使用的帧号。"""
    # 先检查窗口覆盖率和最小样本数，避免稀疏数据产生伪峰值。
    window_frames = end_frame - start_frame + 1
    frame_ids = sorted(
        frame_id
        for frame_id in values_by_frame
        if start_frame <= frame_id <= end_frame
    )
    coverage = len(frame_ids) / max(1, window_frames)
    if coverage < minimum_coverage or len(frame_ids) < 3:
        return 0.0, int(round(coverage * 100)), frame_ids

    array = np.full(window_frames, np.nan, dtype=float)
    for frame_id in frame_ids:
        array[frame_id - start_frame] = values_by_frame[frame_id]
    finite = np.isfinite(array)
    if finite.sum() < 3:
        return 0.0, int(round(coverage * 100)), frame_ids
    indexes = np.arange(window_frames)
    # 缺失帧用线性插值补齐，只在已有帧覆盖率达标后执行。
    filled = np.interp(indexes, indexes[finite], array[finite])
    # 鳃动有效频段固定在 0.60–1.50 Hz，滤除慢漂移和高频噪声。
    sos = butter(
        4,
        [BAND_LOW_HZ, BAND_HIGH_HZ],
        btype="bandpass",
        fs=fps,
        output="sos",
    )
    try:
        filtered = sosfiltfilt(sos, filled)
    except ValueError:
        # 窗口过短时滤波器不满足稳定条件，按“结果不可用”处理。
        return 0.0, int(round(coverage * 100)), frame_ids
    # Savitzky-Golay 窗口必须是奇数，且不能超过当前分析窗口长度。
    smooth_window = min(
        11,
        window_frames if window_frames % 2 == 1 else window_frames - 1,
    )
    smoothed = (
        savgol_filter(filtered, smooth_window, 3)
        if smooth_window >= 5
        else filtered
    )
    q25, q75 = np.percentile(smoothed, [25, 75])
    scale = float(q75 - q25) or float(np.std(smoothed)) or 1.0
    # 使用中位数和 IQR 做稳健归一化，降低个别帧振幅变化的影响。
    normalized = (smoothed - float(np.median(smoothed))) / scale
    peaks, _ = find_peaks(
        normalized,
        distance=max(1, int(round(min_distance_seconds * fps))),
        prominence=prominence,
    )
    # 峰数换算为每分钟次数。
    frequency = len(peaks) * 60.0 / (window_frames / fps)
    return frequency, int(round(coverage * 100)), frame_ids


def calculate_gill_frequency(data,fps,window_seconds=10.0,min_distance_seconds=.25,prominence=.05,minimum_coverage=.80):
    """标准行为结果适配器：data是已提取灰度的基础记录，不接收原视频。
    每条记录含track_id,frame_id,gray_mean,valid_tracking。返回每鱼最新窗口，
    格式为data数组；窗口不足保留原接口的0占位，不等于确定没有鳃动。
    原segmentation输入先在数据准备阶段正确解码并提取灰度，不能在此用整鱼掩膜代替鳃区。
    """
    if fps<=0 or window_seconds<=0 or not 0<minimum_coverage<=1:raise ValueError('Invalid frequency parameters')
    grouped={}
    for r in data:
        if flag(r.get('valid_tracking',True)) and math.isfinite(float(r.get('gray_mean','nan'))):grouped.setdefault(int(r['track_id']),{})[int(r['frame_id'])]=float(r['gray_mean'])
    output=[];window=max(15,round(window_seconds*fps))
    for ident,values in sorted(grouped.items()):
        ids=sorted(values);end=ids[-1]
        if end-ids[0]+1<window:value=0.;used=ids
        else:value,_,used=_frequency_from_window(values,end-window+1,end,fps,min_distance_seconds,prominence,minimum_coverage)
        output.append(dict(track_id=ident,frame_id=end,behavior_indicator='gill_frequency',behavior_value=round(value,3),unit='times/min',frame_range=used))
    return {'data':output}

def run(input_csv=INPUT,output=OUTPUT):
    """仅从鳃区result.csv做滤波、峰检测和统计。保留原离线三曲线算法和输出。
    框内均值/方差依赖拍摄光照、运动和区域质量；峰率不是直接观测到的每次呼吸。
    """
    rows,meta=read_table(input_csv,['gray_mean','gray_variance','valid_tracking']);records=[]
    for r in observations(rows):
        item=dict(frame=int(r['frame_id']),fish_id=int(r['track_id']),time_s=float(r['time_s']),valid_tracking=flag(r['valid_tracking']))
        for k in ['confidence','x1','y1','x2','y2','gray_mean','gray_variance']:item[k]=float(r[k] or 'nan')
        for k in ['x1','y1','x2','y2']:item[k]=int(item[k])
        records.append(item)
    if not records:raise ValueError('No gill observations')
    fps=float(meta['fps'])
    if fps<=2*BAND_HIGH_HZ:raise ValueError('FPS insufficient for frequency band')
    metrics=pd.DataFrame(records).sort_values(['fish_id','frame']).reset_index(drop=True)
    for ident,group in metrics.groupby('fish_id'):
        if len(group)<30 or not group['gray_mean'].notna().any():raise ValueError(f'Fish {ident}: insufficient gill signal for original offline filter')
    info=dict(frame_count=int(meta['frame_count']),width=int(meta['width']),height=int(meta['height']),fps=fps,duration=int(meta['frame_count'])/fps)
    data=build_visualization_data(metrics,info);data['meta']['source']=str(Path(input_csv).resolve());data['meta']['video']=meta['video_path'];data['meta']['run_id']=meta['run_id'];data['meta']['peakProminence']=PEAK_PROMINENCE;data['meta']['peakDistanceS']=PEAK_DISTANCE_S
    output=Path(output);analysis=output/'analysis';analysis.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(analysis/'frame_gray_metrics.csv',index=False,encoding='utf-8-sig')
    (output/'gill_data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')
    summary=export_peak_tables(data,analysis);print(f'Gill: {len(records)} observations, {len(summary)} curve summaries -> {output}',flush=True)
    return data,summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default=str(INPUT));p.add_argument('--output',default=str(OUTPUT));a=p.parse_args();run(a.input,a.output)
