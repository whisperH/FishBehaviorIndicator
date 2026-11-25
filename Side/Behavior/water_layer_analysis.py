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
INPUT=Path(r"E:\Phd\FishBehavior\Side\Tracker\Output\body\result.csv")
OUTPUT=Path(r"E:\Phd\FishBehavior\Side\Behavior\Output\water_layer")
SURFACE_Y=232.0
BOTTOM_Y=553.0
WINDOW_SECONDS=5.0
LAYER_NAMES=('top','middle','bottom')
DEPTH_MEAN='water_depth_mean'
LAYER_TOP='water_layer_top_ratio'
LAYER_MIDDLE='water_layer_middle_ratio'
LAYER_BOTTOM='water_layer_bottom_ratio'
def _view_supports_depth(view_info: object) -> bool:
    """判断视角是否能由图像 y 轴表示水深；俯视和底视不满足该前提。"""
    return str(view_info or "").strip().lower() not in {"top", "bottom"}

def _segmentation_center(record: dict) -> tuple[float, float]:
    """从标准记录的 RLE 掩膜计算前景像素算术中心 (x, y)。"""
    segmentation = record.get("segmentation")
    if not isinstance(segmentation, dict):
        raise ValueError("SAM segmentation is required for this metric")
    return mask_centroid(rle_to_mask(segmentation))

def _layer_index(depth_ratio: float) -> int:
    """按三等分水深返回层索引：0=表层，1=中层，2=底层。"""
    if depth_ratio < 1.0 / 3.0:
        return 0
    if depth_ratio < 2.0 / 3.0:
        return 1
    return 2

def extract_water_layer_series(
    data: list[dict],
    surface_y: float,
    bottom_y: float,
) -> dict[int, list[dict[str, float | int]]]:
    """把标准记录转换为按鱼分组、按帧排序的归一化水深观测。

    必需字段为 track_id、frame_id 和 segmentation。model_info、video_info、
    category_id、bbox、area 等元数据不参与计算。view_info 为 top/bottom 时
    该记录会被跳过，因为此类视角的图像 y 轴不能代表真实水深。

    归一化公式：
    ``(center_y - surface_y) / (bottom_y - surface_y)``。
    结果裁剪到 0–1，0 表示水面，1 表示水底。

    Returns:
    {track_id: [observation, ...]}，观测按 frame_id 升序排列。
    """
    if not isinstance(data, list) or not data:
        raise ValueError("data must be a non-empty standard record array")
    if bottom_y <= surface_y:
        raise ValueError("bottom_y must be greater than surface_y")

    series: dict[int, list[dict[str, float | int]]] = {}
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise TypeError(f"data[{index}] must be an object")
        if "track_id" not in record or "frame_id" not in record:
            raise ValueError(f"data[{index}] requires track_id and frame_id")
        if not _view_supports_depth(record.get("view_info")):
            continue
        # 使用 SAM 掩膜全部前景像素的质心，而不是 bbox 中心。
        center_x, center_y = _segmentation_center(record)
        # 将绝对像素 y 映射到水面到水底之间的 0–1 深度比例。
        ratio = float(
            np.clip(
                (center_y - surface_y) / (bottom_y - surface_y),
                0.0,
                1.0,
            )
        )
        track_id = int(record["track_id"])
        bbox = record.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            # bbox 只用于展示；缺失时退化为以质心为中心的零尺寸框。
            bbox = [center_x, center_y, 0.0, 0.0]
        series.setdefault(track_id, []).append(
            {
                "frame_id": int(record["frame_id"]),
                "depth_ratio": ratio,
                "layer_index": _layer_index(ratio),
                "center_x": center_x,
                "center_y": center_y,
                "bbox": [float(value) for value in bbox[:4]],
            }
        )
    for observations in series.values():
        # 后续窗口计算隐含依赖时间顺序，因此在返回前统一排序。
        observations.sort(key=lambda item: int(item["frame_id"]))
    return series

def _distribution_rows(
    track_id: int,
    observations: list[dict[str, float | int]],
    window_frames: int,
    minimum_coverage: float,
) -> list[dict[str, object]]:
    """为单条鱼生成最新瞬时指标和最近窗口的分布指标。"""
    frame_ids = [int(item["frame_id"]) for item in observations]
    last_frame = frame_ids[-1]
    start_frame = last_frame - window_frames + 1
    window = [
        item for item in observations if int(item["frame_id"]) >= start_frame
    ]
    used_frames = [int(item["frame_id"]) for item in window]
    # 窗口类指标只有在帧跨度和覆盖率同时满足时才计算；覆盖率同时能识别
    # 跟踪中断和重复/缺失帧造成的稀疏观测。
    coverage = len(window) / max(1, window_frames)
    ready = (
        last_frame - frame_ids[0] + 1 >= window_frames
        and coverage >= minimum_coverage
    )

    if ready:
        depths = np.asarray(
            [float(item["depth_ratio"]) for item in window],
            dtype=float,
        )
        layers = np.asarray(
            [int(item["layer_index"]) for item in window],
            dtype=int,
        )
        # 平均深度是连续比例；三层占比统计窗口内落在各层的帧比例。
        mean_depth = float(depths.mean())
        ratios = [float(np.mean(layers == index)) for index in range(3)]
    else:
        mean_depth = 0.0
        ratios = [0.0, 0.0, 0.0]

    # 瞬时指标始终取最新一帧，不受窗口是否已满影响。
    latest = observations[-1]
    latest_center_x = float(latest["center_x"])
    latest_center_y = float(latest["center_y"])
    latest_depth = float(latest["depth_ratio"])
    latest_layer = int(latest["layer_index"])
    base = {
        "track_id": int(track_id),
        "frame_id": int(last_frame),
        "frame_range": used_frames,
    }
    return [
        {
            **base,
            "behavior_indicator": "segmentation_center_x",
            "behavior_value": round(latest_center_x, 3),
            "unit": "pixel",
        },
        {
            **base,
            "behavior_indicator": "segmentation_center_y",
            "behavior_value": round(latest_center_y, 3),
            "unit": "pixel",
        },
        {
            **base,
            "behavior_indicator": "water_depth_percent",
            "behavior_value": round(latest_depth * 100.0, 3),
            "unit": "percent",
        },
        {
            **base,
            "behavior_indicator": "water_layer_index",
            "behavior_value": latest_layer,
            "unit": "layer",
        },
        {
            **base,
            "behavior_indicator": DEPTH_MEAN,
            "behavior_value": round(mean_depth, 6),
            "unit": "ratio",
        },
        {
            **base,
            "behavior_indicator": LAYER_TOP,
            "behavior_value": round(ratios[0], 6),
            "unit": "ratio",
        },
        {
            **base,
            "behavior_indicator": LAYER_MIDDLE,
            "behavior_value": round(ratios[1], 6),
            "unit": "ratio",
        },
        {
            **base,
            "behavior_indicator": LAYER_BOTTOM,
            "behavior_value": round(ratios[2], 6),
            "unit": "ratio",
        },
    ]

def calculate_water_layer_distribution(
    data: list[dict],
    surface_y: float = 232.0,
    bottom_y: float = 553.0,
    window_frames: int = 150,
    minimum_coverage: float = 0.80,
) -> dict[str, list[dict[str, object]]]:
    """计算每条鱼最新的水层位置和时间窗口内的水层分布。

    输入 data 使用统一标准记录格式，每条记录必需包含 track_id、frame_id 和
    SAM segmentation RLE。surface_y 与 bottom_y 是图像中水面和水底的 y
    像素坐标，且必须满足 bottom_y > surface_y。

    每条鱼输出 8 个标准行为指标：
    - segmentation_center_x/y: 最新掩膜质心，单位 pixel。
    - water_depth_percent: 最新归一化水深百分比，范围 0–100。
    - water_layer_index: 最新层索引，0=表层，1=中层，2=底层。
    - water_depth_mean: 窗口平均归一化水深，0=水面，1=水底。
    - water_layer_top/middle/bottom_ratio: 窗口内各层帧数占比。

    后四个窗口指标在窗口未满或覆盖率不足时返回 0；前四个瞬时指标始终取
    最新有效帧。所有记录都使用 frame_range 返回本次实际使用的帧号。
    """
    if window_frames < 1:
        raise ValueError("window_frames must be at least 1")
    series = extract_water_layer_series(data, surface_y, bottom_y)
    output: list[dict[str, object]] = []
    for track_id, observations in sorted(series.items()):
        output.extend(
            _distribution_rows(
                track_id,
                observations,
                window_frames,
                minimum_coverage,
            )
        )
    return {"data": output}

def run_analysis(
    series: dict,
    output: Path,
    surface_y: float,
    bottom_y: float,
    window_frames: int,
) -> None:
    """计算并导出逐帧水层、鱼级汇总以及最新窗口分布结果。"""
    output.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for track_id, observations in sorted(series.items()):
        for item in observations:
            frame_rows.append(
                {
                    "track_id": track_id,
                    "frame_id": int(item["frame_id"]),
                    "depth_ratio": round(float(item["depth_ratio"]), 6),
                    "layer_index": int(item["layer_index"]),
                    "layer_name": LAYER_NAMES[int(item["layer_index"])],
                    "center_x": round(float(item["center_x"]), 3),
                    "center_y": round(float(item["center_y"]), 3),
                    "bbox_x": round(float(item["bbox"][0]), 3),
                    "bbox_y": round(float(item["bbox"][1]), 3),
                    "bbox_w": round(float(item["bbox"][2]), 3),
                    "bbox_h": round(float(item["bbox"][3]), 3),
                }
            )
        depth = np.asarray(
            [float(item["depth_ratio"]) for item in observations],
            dtype=float,
        )
        layers = np.asarray(
            [int(item["layer_index"]) for item in observations],
            dtype=int,
        )
        summary_rows.append(
            {
                "track_id": track_id,
                "samples": len(observations),
                "mean_depth_ratio": round(float(depth.mean()), 6),
                "top_ratio": round(float(np.mean(layers == 0)), 6),
                "middle_ratio": round(float(np.mean(layers == 1)), 6),
                "bottom_ratio": round(float(np.mean(layers == 2)), 6),
            }
        )
    pd.DataFrame(frame_rows).sort_values(
        ["frame_id", "track_id"]
    ).to_csv(
        output / "frame_water_layer.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(summary_rows).to_csv(
        output / "track_water_layer_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    distribution = {"data": [metric for track_id, group in sorted(series.items()) for metric in _distribution_rows(track_id,group,window_frames,.80)]}
    (output / "water_layer_distribution.json").write_text(
        json.dumps(distribution, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(input_csv=INPUT,output=OUTPUT,surface_y=SURFACE_Y,bottom_y=BOTTOM_Y,window_seconds=WINDOW_SECONDS):
    """从质心计算归一化水深；三层比率按观测帧数计算，不是实际dt加权停留秒数。"""
    if not math.isfinite(surface_y) or not math.isfinite(bottom_y) or bottom_y<=surface_y:raise ValueError('Invalid water bounds')
    if not math.isfinite(window_seconds) or window_seconds<=0:raise ValueError('Invalid window')
    rows,meta=read_body(input_csv);series={}
    for r in observations(rows):
        if not flag(r['position_valid']):continue
        x,y=float(r['center_x']),float(r['center_y']);depth=float(np.clip((y-surface_y)/(bottom_y-surface_y),0,1))
        series.setdefault(int(r['track_id']),[]).append(dict(frame_id=int(r['frame_id']),center_x=x,center_y=y,depth_ratio=depth,layer_index=_layer_index(depth),bbox=[float(r[k]) for k in ['bbox_x','bbox_y','bbox_w','bbox_h']]))
    if not series:raise ValueError('No valid body positions')
    for group in series.values():group.sort(key=lambda r:r['frame_id'])
    window=max(1,round(window_seconds*float(meta['fps'])))
    run_analysis(series,Path(output),surface_y,bottom_y,window)
    info=dict(meta,video=meta['video_path'],surface_y=surface_y,bottom_y=bottom_y,window_frames=window,input_csv=str(Path(input_csv).resolve()))
    tracks=pd.read_csv(Path(output)/'track_water_layer_summary.csv').to_dict('records')
    (Path(output)/'water_layer_summary.json').write_text(json.dumps({'meta':info,'tracks':tracks},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Water layer: {sum(map(len,series.values()))} observations -> {output}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default=str(INPUT));p.add_argument('--output',default=str(OUTPUT));p.add_argument('--surface-y',type=float,default=SURFACE_Y);p.add_argument('--bottom-y',type=float,default=BOTTOM_Y);p.add_argument('--window-seconds',type=float,default=WINDOW_SECONDS);a=p.parse_args();run(a.input,a.output,a.surface_y,a.bottom_y,a.window_seconds)
