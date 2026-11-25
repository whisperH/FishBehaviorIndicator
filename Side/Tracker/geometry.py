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

BODY_FRAMES=Path(r"E:\Phd\FishBehavior\Side\Tracker\Output\body\frames.csv")
BODY_OUTPUT=Path(r"E:\Phd\FishBehavior\Side\Tracker\Output\body")
HEAD_MASKS=Path(r"E:\Phd\FishBehavior\Side\Tracker\Input\body\head_mask_shapes.json")
GILL_VIDEO=Path(r"E:\Phd\FishBehavior\Side\Tracker\Input\gill\video.mp4")
GILL_MOT=Path(r"E:\Phd\FishBehavior\Side\Tracker\Input\gill")
GILL_OUTPUT=Path(r"E:\Phd\FishBehavior\Side\Tracker\Output\gill")
def load_mot(zip_path: Path) -> pd.DataFrame:
    """从 MOT 目录或 zip 包读取 gt.txt，并返回标准列名的跟踪表。"""
    if zip_path.is_dir():
        frame = pd.read_csv(
            zip_path / "gt" / "gt.txt",
            header=None,
            names=["frame", "fish_id", "x", "y", "w", "h", "mark", "class", "conf"],
        )
    else:
        with zipfile.ZipFile(zip_path) as archive:
            frame = pd.read_csv(
                io.BytesIO(archive.read("gt/gt.txt")),
                header=None,
                names=["frame", "fish_id", "x", "y", "w", "h", "mark", "class", "conf"],
            )
    for column in ("frame", "fish_id", "mark", "class"):
        frame[column] = frame[column].astype(int)
    return frame

def extract_metrics(video_path: Path, tracking: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """按 MOT 跟踪框提取逐帧灰度均值和方差，供离线可视化分析使用。

    跟踪框会先裁剪到视频边界；只有置信度达标且框几乎完整位于画面内时，
    该帧才标记为有效，避免越界目标污染曲线。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    by_frame = {
        int(frame): group.sort_values("fish_id")
        for frame, group in tracking.groupby("frame", sort=False)
    }
    rows: list[dict[str, float | int | bool]] = []
    for frame_index in range(min(frame_count, int(tracking.frame.max()))):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Video ended at frame {frame_index + 1}")
        frame_no = frame_index + 1
        tracks = by_frame.get(frame_no)
        if tracks is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for row in tracks.itertuples(index=False):
            # MOT 的 x/y/w/h 是浮点框，这里转换成左闭右开的像素边界。
            x1 = max(0, min(width - 1, int(math.floor(row.x))))
            y1 = max(0, min(height - 1, int(math.floor(row.y))))
            x2 = max(x1 + 1, min(width, int(math.ceil(row.x + row.w))))
            y2 = max(y1 + 1, min(height, int(math.ceil(row.y + row.h))))
            roi = gray[y1:y2, x1:x2].astype(np.float64)
            # 裁剪后面积与原框面积的比值用于识别严重出界的跟踪框。
            visible = roi.size / max(1.0, float(row.w) * float(row.h))
            valid = bool(float(row.conf) >= 0.5 and visible >= 0.98 and roi.size)
            rows.append(
                {
                    "frame": frame_no,
                    "time_s": frame_index / fps,
                    "fish_id": int(row.fish_id),
                    "valid_tracking": valid,
                    "confidence": float(row.conf),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "gray_mean": float(roi.mean()) if valid else np.nan,
                    "gray_variance": float(roi.var()) if valid else np.nan,
                }
            )
    cap.release()
    metrics = pd.DataFrame(rows).sort_values(["fish_id", "frame"]).reset_index(drop=True)
    return metrics, {
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "duration": frame_count / fps,
    }


def prepare_body(input_csv=BODY_FRAMES,output=BODY_OUTPUT,head_masks=HEAD_MASKS):
    """只计算基础量：像素质心、PCA主轴角度/伸长率、头尾端点/宽度与有效性。
    不计算速度、频率或水层。保留原项目的两种质心口径：angle_center来自原始
    清理掩膜，center来自简化轮廓重建（原speed/water实际使用此口径）。
    """
    rows,meta=read_table(input_csv,['mask_rle','polygons']);results=[];samples=[]
    width,height=int(meta['width']),int(meta['height'])
    keys='angle_center_x angle_center_y angle_horizontal_deg directed_angle_deg elongation tail_x tail_y head_x head_y head_width tail_width'.split()
    for raw in rows:
        r={k:raw.get(k,'') for k in BODY_FIELDS};r.update(position_valid=False,geometry_valid=False,quality_reason='empty_frame' if raw['record_type']=='empty_frame' else 'missing_mask')
        if raw['record_type']=='observation' and raw['mask_rle']:
            m=rle_to_mask({'size':[height,width],'counts':raw['mask_rle']});polygons=json.loads(raw['polygons'] or '[]');poly=fill_polygons(polygons,width,height)
            try:
                r['center_x'],r['center_y']=mask_centroid(poly if poly.any() else m);r['position_valid']=True;r['area']=int(m.sum())
                orient=mask_orientation(m)
                if orient is not None:
                    r.update(dict(zip(keys,orient)));r['orientation_deg']=r['directed_angle_deg'];r['geometry_valid']=True;r['quality_reason']='ok'
                    # 旧轮廓导入可以保留其原始全掩膜几何种子，避免把轮廓简化误当新测量。
                    seed=json.loads(raw.get('source_geometry') or '{}')
                    for k,v in seed.items():
                        if k in ['center_x','center_y']:r['angle_'+k]=v
                        elif k in r:r[k]=v
                    r['frame']=int(raw['frame_id']);r['valid']=True
                    samples.append(dict(frame=r['frame'],track_id=int(r['track_id']),polygons=polygons))
                else:r['quality_reason']='too_small_or_short_mask'
            except ValueError as e:r['quality_reason']=str(e)
        results.append(r)
    head_payload=json.loads(Path(head_masks).read_text('utf-8-sig')) if head_masks and Path(head_masks).is_file() else None
    if head_payload:
        hm=head_payload.get('meta',{})
        for k in ['width','height']:
            if k in hm and int(hm[k])!=int(meta[k]):raise ValueError('Head masks size differs')
    apply_head_direction(results,samples,head_payload,width,height)
    write_table(Path(output)/'result.csv',results,BODY_FIELDS)
    print(f'Body geometry: {len(results)} rows -> {output}',flush=True)
    return Path(output)/'result.csv'

def prepare_gill(video=GILL_VIDEO,mot=GILL_MOT,output=GILL_OUTPUT):
    """按原MOT框提取灰度均值、方差。conf>=0.5且可见面积>=98%才有效。
    这是基础信号提取，不做滤波和数峰；无需鱼体SAM3。空帧也保存时间轴。
    """
    meta=video_metadata(video,'mot_gill_box');tracking=load_mot(Path(mot));metrics,info=extract_metrics(Path(video),tracking);by_frame={}
    for item in metrics.to_dict('records'):by_frame.setdefault(int(item['frame']),[]).append(item)
    rows=[]
    for frame in range(1,int(meta['frame_count'])+1):
        for item in by_frame.get(frame,[]):
            r=base_row(meta,frame,int(item['fish_id']));r.update({k:item[k] for k in ['valid_tracking','confidence','x1','y1','x2','y2','gray_mean','gray_variance']});r['quality_reason']='ok' if item['valid_tracking'] else 'low_confidence_or_clipped_box';r['annotation_path']=str(Path(mot).resolve());rows.append(r)
        if frame not in by_frame:rows.append(base_row(meta,frame))
    write_table(Path(output)/'result.csv',rows,GILL_FIELDS);print(f'Gill geometry: {len(rows)} rows -> {output}',flush=True)
    return Path(output)/'result.csv'

def run(task='body',input_csv=None,output=None,**kwargs):
    """run(task='body'|'gill',...):按任务选择基础准备，无跨视角或跨行为调用。"""
    if task=='body':return prepare_body(input_csv or BODY_FRAMES,output or BODY_OUTPUT,kwargs.get('head_masks',HEAD_MASKS))
    if task=='gill':return prepare_gill(kwargs.get('video',GILL_VIDEO),kwargs.get('mot',GILL_MOT),output or GILL_OUTPUT)
    raise ValueError('Unknown task')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['body','gill'],default='body');p.add_argument('--input');p.add_argument('--output');p.add_argument('--video',default=str(GILL_VIDEO));p.add_argument('--mot',default=str(GILL_MOT));p.add_argument('--head-masks',default=str(HEAD_MASKS))
    a=p.parse_args();run(a.task,a.input,a.output,video=a.video,mot=a.mot,head_masks=a.head_masks)
