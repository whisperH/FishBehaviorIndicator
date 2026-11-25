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
OUTPUT=Path(r"E:\Phd\FishBehavior\Side\Behavior\Output\angle")
def calculate_fish_body_angle(data: list[dict]) -> dict[str, list[dict[str, object]]]:
    """从标准 SAM 记录计算每一帧的鱼体角度。

    输入记录必需包含 track_id、frame_id 和 segmentation；model_info、
    video_info、view_info、category_id、area、bbox 等元数据不会参与计算。
    segmentation 是 COCO 风格 RLE，size=[height, width]。

    每帧返回一条标准行为记录：
    - behavior_indicator 固定为 fish_body_angle。
    - behavior_value 为鱼体主轴与水平线的无符号夹角，范围 0–90°。
      图像坐标中 0° 向右、90° 向下、180° 向左、270° 向上。
    - frame_range 只包含当前帧。

    鱼头端通过主轴方向的宽度轮廓判断：将掩膜投影到主轴上分成 20 段，
    比较两端各 30% 范围的平均宽度，较宽的一端作为鱼头。

    掩膜缺失、为空、像素不足或解码失败时，behavior_value 返回 0。
    """
    if not isinstance(data, list) or not data:
        raise ValueError("data must be a non-empty standard record array")
    output: list[dict[str, object]] = []
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise TypeError(f"data[{index}] must be an object")
        if "track_id" not in record or "frame_id" not in record:
            raise ValueError(f"data[{index}] requires track_id and frame_id")
        frame_id = int(record["frame_id"])
        angle = 0.0
        segmentation = record.get("segmentation")
        if isinstance(segmentation, dict):
            try:
                orientation = mask_orientation(rle_to_mask(segmentation))
                if orientation is not None:
                    directed_angle = orientation[2]
                    angle = float(directed_angle)
            except (ValueError, KeyError, TypeError):
                # 单帧分割异常不阻断整批数据，按无效观测返回 0。
                angle = 0.0
        result = {
            "track_id": int(record["track_id"]),
            "frame_id": frame_id,
            "behavior_indicator": "fish_body_angle",
            "behavior_value": round(angle, 6),
            "unit": "degree",
            "frame_range": [frame_id],
        }
        output.append(result)
    return {"data": output}

def summarize_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    """按鱼汇总有效帧的方向角均值、中位数、标准差和极值。"""
    valid = dataframe[dataframe["valid"]].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "track_id",
                "valid_frames",
                "mean_angle_deg",
                "median_angle_deg",
                "std_angle_deg",
                "min_angle_deg",
                "max_angle_deg",
            ]
        )
    rows: list[dict[str, object]] = []
    for track_id, group in valid.groupby("track_id", sort=True):
        angles = group["angle_horizontal_deg"].to_numpy(dtype=float)
        rows.append(
            {
                "track_id": int(track_id),
                "valid_frames": int(len(angles)),
                "mean_angle_deg": float(np.mean(angles)),
                "median_angle_deg": float(np.median(angles)),
                "std_angle_deg": float(np.std(angles)),
                "min_angle_deg": float(np.min(angles)),
                "max_angle_deg": float(np.max(angles)),
            }
        )
    return pd.DataFrame(rows)


def run(input_csv=INPUT,output=OUTPUT):
    """读取geometry已测出的0~90度主轴角，整理逐帧/每鱼统计，不加载模型。
    主轴角无上/下正负，head/tail方向只供展示。无效行保留valid=False。
    """
    rows,meta=read_body(input_csv);result=[]
    fields='frame time_s track_id score box_iou valid mask_area center_x center_y angle_horizontal_deg orientation_deg directed_angle_deg elongation head_x head_y tail_x tail_y head_width tail_width direction_source head_match_score head_match_overlap'.split()
    strings={'direction_source'}
    for r in observations(rows):
        item=dict(frame=int(r['frame_id']),time_s=float(r['time_s']),track_id=int(r['track_id']),valid=flag(r['geometry_valid']),mask_area=float(r['area'] or 'nan'),center_x=float(r['angle_center_x'] or 'nan'),center_y=float(r['angle_center_y'] or 'nan'))
        for k in fields:
            if k not in item:item[k]=r.get(k,'') if k in strings else float(r.get(k) or 'nan')
        result.append(item)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);write_table(output/'frame_angles.csv',result,fields)
    summary=summarize_results(pd.DataFrame(result,columns=fields));summary.to_csv(output/'track_angle_summary.csv',index=False,encoding='utf-8-sig')
    (output/'angle_summary.json').write_text(json.dumps({'meta':dict(meta,video=meta['video_path'],input_csv=str(Path(input_csv).resolve()),angle_definition='Unsigned PCA axis angle 0-90 degrees'),'tracks':summary.to_dict('records')},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Angle: {len(result)} observations -> {output}',flush=True)
    return {'frames':result,'summary':summary.to_dict('records')}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default=str(INPUT));p.add_argument('--output',default=str(OUTPUT));a=p.parse_args();run(a.input,a.output)
