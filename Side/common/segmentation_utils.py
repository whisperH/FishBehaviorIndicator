"""chenxi：侧面内部的CSV协议、COCO掩膜编解码和PCA测量工具；不供顶部引用。
CSV保留完整时间轴；一行是一帧的一条鱼/鳃部区域，空帧用empty_frame表示。
无效浮点数写为空白；valid字段单独表示可测性，空白不能解释成零。
"""
from __future__ import annotations
from pathlib import Path
import csv,json,math,os
from uuid import uuid4
import numpy as np
import cv2
from pycocotools import mask as mask_utils
META_FIELDS='schema_version,run_id,record_type,video_path,video_size_bytes,video_mtime_ns,fps,width,height,frame_count,frame_id,time_s,track_id,source_kind'.split(',')
FRAME_FIELDS=META_FIELDS+'score,box_iou,bbox_x,bbox_y,bbox_w,bbox_h,mask_rle,polygons,source_geometry,annotation_path'.split(',')
BODY_FIELDS=META_FIELDS+'score,box_iou,bbox_x,bbox_y,bbox_w,bbox_h,center_x,center_y,angle_center_x,angle_center_y,area,angle_horizontal_deg,orientation_deg,directed_angle_deg,elongation,head_x,head_y,tail_x,tail_y,head_width,tail_width,direction_source,head_match_score,head_match_overlap,position_valid,geometry_valid,quality_reason,annotation_path'.split(',')
GILL_FIELDS=META_FIELDS+'valid_tracking,confidence,x1,y1,x2,y2,gray_mean,gray_variance,quality_reason,annotation_path'.split(',')

def rle_to_mask(segmentation):
    """严格COCO标准：[高,宽]，按列展开。压缩字符串和普通游程数组得到同一掩膜。
    不把普通游程数组误当累计边界，不用默认按行reshape。修复原接口的坐标偏移。
    """
    h,w=map(int,segmentation['size'])
    if h<=0 or w<=0:raise ValueError('Invalid mask size')
    counts=segmentation['counts']
    if isinstance(counts,str):r={'size':[h,w],'counts':counts.encode('ascii')}
    elif isinstance(counts,bytes):r={'size':[h,w],'counts':counts}
    else:
        raw=np.asarray(counts)
        if raw.ndim!=1 or not np.all(np.isfinite(raw)) or np.any(raw<0) or np.any(raw!=np.floor(raw)) or sum(map(int,raw))!=h*w:raise ValueError('Invalid COCO RLE lengths')
        r=mask_utils.frPyObjects({'size':[h,w],'counts':list(map(int,raw))},h,w)
    m=np.asarray(mask_utils.decode(r),np.uint8)
    if m.shape!=(h,w):raise ValueError('Invalid decoded mask shape')
    return m

def encode_mask(mask):
    return mask_utils.encode(np.asfortranarray(mask,dtype=np.uint8))['counts'].decode('ascii')

def mask_centroid(mask):
    ys,xs=np.nonzero(mask)
    if not len(xs):raise ValueError('Empty foreground')
    return float(xs.mean()),float(ys.mean())

def flag(value):
    if value in (True,1,'1','True','true'):return True
    if value in (False,0,'0','False','false',''):return False
    raise ValueError('Invalid boolean: '+str(value))

def video_metadata(path,source_kind):
    path=Path(path).resolve();cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise FileNotFoundError(path)
    fps=float(cap.get(cv2.CAP_PROP_FPS));width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    if not math.isfinite(fps) or fps<=0 or min(width,height,count)<=0:raise ValueError('Invalid video metadata')
    st=path.stat()
    return dict(schema_version=1,run_id=uuid4().hex,video_path=str(path),video_size_bytes=st.st_size,video_mtime_ns=st.st_mtime_ns,fps=fps,width=width,height=height,frame_count=count,source_kind=source_kind)

def base_row(meta,frame_id,track_id=None):
    return dict(meta,record_type='empty_frame' if track_id is None else 'observation',frame_id=int(frame_id),time_s=(int(frame_id)-1)/float(meta['fps']),track_id=track_id)

def write_table(path,rows,fields):
    """原子替换本次CSV，避免失败时留下半张可误用的表。UTF-8 BOM兼容Excel。"""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    def value(v):
        if v is None:return ''
        if isinstance(v,(float,np.floating)) and not math.isfinite(float(v)):return ''
        if isinstance(v,(dict,list,tuple)):return json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False)
        return v
    try:
        with temp.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
            for r in rows:writer.writerow({k:value(r.get(k)) for k in fields})
        temp.replace(path)
    finally:
        if temp.exists():temp.unlink()

def read_table(path,required=()):
    """校验同一视频/运行批次、唯一鱼帧和时间。返回原始字符串记录和元数据。
    读取完整批次；禁止把同ID不同视频拼接。source_kind明确fresh SAM3或旧轮廓导入。
    """
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        missing=set(META_FIELDS+list(required))-set(reader.fieldnames or [])
        if missing:raise ValueError('Missing columns: '+','.join(sorted(missing)))
        rows=list(reader)
    if not rows:raise ValueError('Empty CSV')
    meta={k:rows[0][k] for k in META_FIELDS if k not in ['record_type','frame_id','time_s','track_id']}
    if meta['schema_version']!='1':raise ValueError('Unsupported schema')
    for k in ['fps','width','height','frame_count','video_size_bytes','video_mtime_ns']:
        n=float(meta[k])
        if not math.isfinite(n) or n<=0:raise ValueError('Invalid '+k)
    if not Path(meta['video_path']).is_absolute():raise ValueError('Video path must be absolute')
    seen=set();times={};empty=set();observed=set()
    for r in rows:
        if None in r or any(v is None for v in r.values()):raise ValueError('Malformed CSV row')
        if any(r[k]!=v for k,v in meta.items()):raise ValueError('Mixed video/run metadata')
        frame=int(r['frame_id']);t=float(r['time_s'])
        if frame<1 or frame>int(meta['frame_count']) or not math.isfinite(t) or abs(t-(frame-1)/float(meta['fps']))>1e-6:raise ValueError('Invalid frame/time')
        if frame in times and times[frame]!=t:raise ValueError('Inconsistent frame time')
        times[frame]=t
        if r['record_type']=='observation':
            ident=int(r['track_id']);key=(frame,ident)
            if key in seen:raise ValueError('Duplicate fish/frame')
            seen.add(key);observed.add(frame)
        elif r['record_type']=='empty_frame':
            if r['track_id'] or frame in empty:raise ValueError('Malformed empty frame')
            empty.add(frame)
        else:raise ValueError('Invalid record_type')
    if empty&observed:raise ValueError('Empty frame also contains fish')
    return rows,meta

def observations(rows):return [r for r in rows if r['record_type']=='observation']

def read_body(path):
    rows,meta=read_table(path,['center_x','center_y','position_valid','geometry_valid','angle_horizontal_deg'])
    for r in observations(rows):
        for k in ['position_valid','geometry_valid']:flag(r[k])
        if flag(r['position_valid']):
            if not all(math.isfinite(float(r[k])) for k in ['center_x','center_y']):raise ValueError('Invalid centroid')
    return rows,meta

def mask_polygons(mask: np.ndarray) -> list[list[list[int]]]:
    """把二值掩膜转换为适合 JSON 存储的外轮廓多边形。"""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    total_area = max(1, int(mask.sum()))
    polygons: list[list[list[int]]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        # 过滤分割噪点，并控制轮廓点数量以免输出文件过大。
        if area < max(12.0, total_area * 0.008):
            continue
        epsilon = max(1.2, 0.003 * cv2.arcLength(contour, True))
        approximated = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approximated) >= 3:
            polygons.append(approximated.astype(int).tolist())
    return polygons

def normalize_angle_deg(value: float) -> float:
    """把角度规范化到 [0, 360)，并把浮点误差产生的 360° 转为 0°。"""
    normalized = float(value) % 360.0
    if math.isclose(normalized, 0.0, abs_tol=1e-9) or math.isclose(
        normalized, 360.0, abs_tol=1e-9
    ):
        return 0.0
    return normalized

def horizontal_angle(directed: float) -> float:
    """把带方向角折叠为鱼体主轴与水平线的 0–90° 无符号夹角。"""
    half_turn = directed % 180.0
    return min(half_turn, 180.0 - half_turn)

def fill_polygons(
    polygons: list[list[list[int]]],
    width: int,
    height: int,
) -> np.ndarray:
    """把 JSON 多边形恢复为指定视频尺寸的二值掩膜。"""
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
    return mask

def body_mask_axis(body_mask: np.ndarray) -> dict[str, object] | None:
    """计算头尾匹配所需的 PCA 主轴、端点和两端宽度。"""
    ys, xs = np.nonzero(body_mask)
    if len(xs) < 30:
        return None
    points = np.column_stack((xs, ys)).astype(np.float64)
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major_vector = eigenvectors[:, order[0]]
    minor_vector = eigenvectors[:, order[1]]
    projected = centered @ major_vector
    perpendicular = centered @ minor_vector
    low = float(np.percentile(projected, 2.0))
    high = float(np.percentile(projected, 98.0))
    span = high - low
    if span < 2.0:
        return None
    normalized = np.clip((projected - low) / span, 0.0, 0.999999)
    bins = 20
    bin_index = (normalized * bins).astype(int)
    widths = np.full(bins, np.nan, dtype=np.float64)
    for index in range(bins):
        selected = perpendicular[bin_index == index]
        if selected.size >= 3:
            low_width, high_width = np.percentile(selected, [10.0, 90.0])
            widths[index] = float(high_width - low_width)
    left_width = (
        float(np.nanmean(widths[:6]))
        if np.isfinite(widths[:6]).any()
        else 0.0
    )
    right_width = (
        float(np.nanmean(widths[-6:]))
        if np.isfinite(widths[-6:]).any()
        else 0.0
    )
    return {
        "center": center,
        "major": major_vector,
        "low": low,
        "high": high,
        "span": span,
        "low_point": center + major_vector * low,
        "high_point": center + major_vector * high,
        "left_width": left_width,
        "right_width": right_width,
    }

def choose_head_candidate(
    body_mask: np.ndarray,
    axis: dict[str, object],
    head_samples: list[dict],
    width: int,
    height: int,
) -> dict[str, object] | None:
    """从同帧鱼头掩膜中选择与鱼体一个端点最匹配的候选。"""
    center = np.asarray(axis["center"], dtype=np.float64)
    major = np.asarray(axis["major"], dtype=np.float64)
    span = float(axis["span"])
    best: tuple[float, dict[str, object]] | None = None
    for sample in head_samples:
        head_mask = fill_polygons(sample.get("polygons", []), width, height)
        head_area = int(head_mask.sum())
        if head_area < 30:
            continue
        intersection = int(np.logical_and(body_mask, head_mask).sum())
        overlap = intersection / head_area
        if overlap < 0.25:
            continue
        ys, xs = np.nonzero(head_mask)
        head_center = np.asarray([xs.mean(), ys.mean()], dtype=np.float64)
        projection = float((head_center - center) @ major)
        endpoint_distance = min(
            abs(projection - float(axis["low"])),
            abs(projection - float(axis["high"])),
        )
        endpoint_score = float(
            np.clip(1.0 - endpoint_distance / max(1.0, span * 0.45), 0.0, 1.0)
        )
        score = (
            0.50 * overlap
            + 0.30 * endpoint_score
            + 0.20 * float(sample.get("score", 0.0))
        )
        candidate = {
            "sample": sample,
            "score": score,
            "overlap": overlap,
            "head_is_high": projection >= 0.0,
            "head_center_x": float(head_center[0]),
            "head_center_y": float(head_center[1]),
        }
        if best is None or score > best[0]:
            best = (score, candidate)
    return best[1] if best else None

def apply_head_direction(
    rows: list[dict[str, object]],
    mask_samples: list[dict[str, object]],
    head_payload: dict | None,
    width: int,
    height: int,
) -> dict[str, int]:
    """用预计算的 SAM3 鱼头掩膜修正头尾方向，失败时保留宽度轮廓结果。"""
    row_by_key = {
        (int(row["frame"]), int(row["track_id"])): row
        for row in rows
        if str(row.get("valid", "")).lower() == "true"
    }
    for row in row_by_key.values():
        row["direction_source"] = "width_profile_fallback"
    for sample in mask_samples:
        sample["direction_source"] = "width_profile_fallback"

    valid_observations = len(row_by_key)
    matched = 0
    changed = 0
    if head_payload is None:
        return {
            "valid_observations": valid_observations,
            "matched": 0,
            "fallback": valid_observations,
            "changed": 0,
        }

    head_samples_by_frame: dict[int, list[dict]] = {}
    for sample in head_payload.get("samples", []):
        head_samples_by_frame.setdefault(int(sample["frame"]), []).append(sample)

    for sample in mask_samples:
        key = (int(sample["frame"]), int(sample["track_id"]))
        row = row_by_key.get(key)
        if row is None:
            continue
        body_mask = fill_polygons(sample.get("polygons", []), width, height)
        if not body_mask.any():
            continue
        axis = body_mask_axis(body_mask)
        if axis is None:
            continue
        match = choose_head_candidate(
            body_mask,
            axis,
            head_samples_by_frame.get(key[0], []),
            width,
            height,
        )
        if match is None:
            continue

        if bool(match["head_is_high"]):
            head_point = np.asarray(axis["high_point"], dtype=np.float64)
            selected_tail_point = np.asarray(axis["low_point"], dtype=np.float64)
            head_vector = np.asarray(axis["major"], dtype=np.float64)
            head_width = float(axis["right_width"])
            tail_width = float(axis["left_width"])
        else:
            head_point = np.asarray(axis["low_point"], dtype=np.float64)
            selected_tail_point = np.asarray(axis["high_point"], dtype=np.float64)
            head_vector = -np.asarray(axis["major"], dtype=np.float64)
            head_width = float(axis["left_width"])
            tail_width = float(axis["right_width"])

        directed = normalize_angle_deg(
            math.degrees(math.atan2(float(head_vector[1]), float(head_vector[0])))
        )
        previous = float(
            row.get("directed_angle_deg")
            or row.get("orientation_deg")
            or 0.0
        )
        if abs(((directed - previous + 180.0) % 360.0) - 180.0) > 90.0:
            changed += 1
        matched += 1
        updates: dict[str, object] = {
            "angle_horizontal_deg": horizontal_angle(directed),
            "orientation_deg": directed,
            "directed_angle_deg": directed,
            "head_x": float(head_point[0]),
            "head_y": float(head_point[1]),
            "tail_x": float(selected_tail_point[0]),
            "tail_y": float(selected_tail_point[1]),
            "head_width": head_width,
            "tail_width": tail_width,
            "direction_source": "sam3_head_mask",
            "head_match_score": float(match["score"]),
            "head_match_overlap": float(match["overlap"]),
        }
        row.update(updates)
        sample.update(updates)

    return {
        "valid_observations": valid_observations,
        "matched": matched,
        "fallback": valid_observations - matched,
        "changed": changed,
    }

def mask_orientation(
    mask: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
] | None:
    """计算鱼体中心、头尾方向角、伸长率和头尾端点。

    先通过 PCA 得到鱼体主轴，再把掩膜投影到主轴上并按位置分箱统计
    鱼体宽度。鱼头一侧通常更宽，因此宽度均值较大的端点判为头端。
    指标角度为鱼体主轴与水平线的无符号夹角；头尾方向另用于箭头显示。

    Returns:
    - center_x, center_y: 掩膜质心。
    - horizontal_angle: 鱼体主轴与水平线的无符号夹角，范围 0–90°。
    - directed_angle: 从尾端指向头端的方向角，范围 0–360°。
    - elongation: 主次特征值之比的平方根，越大表示鱼体越细长。
    - tail_x, tail_y, head_x, head_y: 尾端和头端坐标。
    - head_width, tail_width: 两端附近宽度轮廓的均值。
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return None
    points = np.column_stack((xs, ys)).astype(np.float64)
    center = points.mean(axis=0)
    centered = points - center
    # PCA：最大特征值对应的特征向量就是鱼体像素分布的主方向。
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # eigh 返回升序特征值，反转后第一个特征向量即鱼体主轴。
    order = np.argsort(eigenvalues)[::-1]
    major_value = max(float(eigenvalues[order[0]]), 1e-9)
    minor_value = max(float(eigenvalues[order[1]]), 1e-9)
    major_vector = eigenvectors[:, order[0]]
    minor_vector = eigenvectors[:, order[1]]
    elongation = math.sqrt(major_value / minor_value)

    # 将掩膜点投影到主轴和副轴，用于定位两端并计算局部宽度轮廓。
    projected = centered @ major_vector
    perpendicular = centered @ minor_vector
    low = float(np.percentile(projected, 2.0))
    high = float(np.percentile(projected, 98.0))
    span = high - low
    if span < 2.0:
        return None

    normalized = (projected - low) / span
    profile_bins = 20
    bin_index = np.clip(
        (normalized * profile_bins).astype(int),
        0,
        profile_bins - 1,
    )
    widths = np.full(profile_bins, np.nan, dtype=np.float64)
    counts = np.zeros(profile_bins, dtype=np.int64)
    for index in range(profile_bins):
        selected = perpendicular[bin_index == index]
        counts[index] = selected.size
        if selected.size >= 3:
            low_width, high_width = np.percentile(selected, [10.0, 90.0])
            widths[index] = float(high_width - low_width)

    end_bins = max(2, int(round(profile_bins * 0.30)))
    left_widths = widths[:end_bins]
    right_widths = widths[-end_bins:]
    left_width = (
        float(np.nanmean(left_widths))
        if np.isfinite(left_widths).any()
        else 0.0
    )
    right_width = (
        float(np.nanmean(right_widths))
        if np.isfinite(right_widths).any()
        else 0.0
    )
    left_count = int(counts[:end_bins].sum())
    right_count = int(counts[-end_bins:].sum())
    head_is_right = (right_width, right_count) >= (left_width, left_count)

    low_point = center + major_vector * low
    high_point = center + major_vector * high
    if head_is_right:
        head_x, head_y = high_point
        tail_x, tail_y = low_point
        head_vector = major_vector
        head_width = right_width
        tail_width = left_width
    else:
        head_x, head_y = low_point
        tail_x, tail_y = high_point
        head_vector = -major_vector
        head_width = left_width
        tail_width = right_width

    # 图像坐标 y 轴向下：0° 向右、90° 向下、180° 向左、270° 向上。
    directed_angle = normalize_angle_deg(
        math.degrees(math.atan2(float(head_vector[1]), float(head_vector[0])))
    )
    half_turn_angle = directed_angle % 180.0
    horizontal_angle = min(half_turn_angle, 180.0 - half_turn_angle)
    return (
        float(center[0]),
        float(center[1]),
        horizontal_angle,
        directed_angle,
        elongation,
        float(tail_x),
        float(tail_y),
        float(head_x),
        float(head_y),
        float(head_width),
        float(tail_width),
    )

