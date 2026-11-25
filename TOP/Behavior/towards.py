# 顶部视角维护负责人：YHC。此脚本仅属于 FishBehavior/Top，不依赖 Side 子项目。
"""独立朝向分析：只读取result.csv，保留原项目朝向计算定义。

tilt_deg=atan2(-head_dy,abs(head_dx))，范围[-90,90]，水平左右均0、向上正。
raw_heading_deg=atan2(-head_dy,head_dx)，保留左右象限，范围[-180,180]。
展开角只用于连续有效段绘图，不能用来填补缺帧；不根据移动方向判头。
API: run(input_csv=绝对CSV,output_dir=绝对目录,save_video=True)；
返回{'frames':每条鱼逐帧朝向字典列表}，不调用其它项目脚本。
"""
from pathlib import Path
INPUT_CSV=Path(r"E:\Phd\FishBehavior\Top\Tracker\Output\Information\result.csv")
OUTPUT_DIR=Path(r"E:\Phd\FishBehavior\Top\Behavior\Output\towards")
HORIZONTAL_TOLERANCE_DEG=10.0  # ±10度内记作水平，用于direction文字分类。
SAVE_VIDEO=True

import csv
import json
import math
from collections import defaultdict
import numpy as np
import cv2


def read_result(input_csv):
    """只读取基础测量CSV；不导入其它项目脚本，不读取其它指标输出。

    一行一帧一鱼；empty_frame只加入时间清单。空数值保持NaN，不能改成有效零。
    数据坐标为原图像素，x向右、y向下。几何有效与位置有效分别使用。
    """
    path = Path(input_csv)
    if not path.is_absolute():
        raise ValueError('输入CSV必须是绝对路径。')
    required = {'schema_version','run_id','record_type','video_id','frame_id','time_s','fps','time_source',
                'image_path','image_width','image_height','track_id','cx','cy','area_px',
                'position_valid','geometry_valid','head_valid','head_dx','head_dy',
                'tail_dx','tail_dy','heading_deg','tail_angle_deg','body_length_px',
                'reference_body_length_px','reference_length_sample_count','centerline',
                'head_confidence','quality_reason'}
    numeric = ['cx','cy','area_px','head_dx','head_dy','tail_dx','tail_dy','heading_deg',
               'tail_angle_deg','body_length_px','reference_body_length_px','head_confidence']
    records, timeline, keys, context = [], {}, set(), None
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames and len(reader.fieldnames)!=len(set(reader.fieldnames)):
            raise ValueError('result.csv含重复列标题')
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('result.csv缺少字段: '+str(sorted(required-set(reader.fieldnames or []))))
        for line_number, raw in enumerate(reader,2):
            try:
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError('该行列数与表头不一致')
                frame, time_s, fps = int(raw['frame_id']), float(raw['time_s']), float(raw['fps'])
                width, height = int(raw['image_width']), int(raw['image_height'])
                if raw['schema_version']!='1' or frame<0 or not math.isfinite(time_s) or time_s<0 or not math.isfinite(fps) or fps<=0 or min(width,height)<=0:
                    raise ValueError('无效版本、帧号、时间、帧率或尺寸')
                if not Path(raw['image_path']).is_absolute():
                    raise ValueError('原图路径必须为绝对路径')
                if not raw['run_id'].strip() or not raw['video_id'].strip() or not raw['time_source'].strip():
                    raise ValueError('运行、视频和时间来源标识不能为空')
                signature=(raw['run_id'],raw['video_id'],fps,width,height,raw['time_source'])
                if context is not None and signature!=context:
                    raise ValueError('不能混合多个运行、视频、帧率或图像尺寸')
                context=signature
                frame_meta={'frame_id':frame,'time_s':time_s,'fps':fps,'image_path':raw['image_path'],
                            'image_width':width,'image_height':height}
                if frame in timeline and timeline[frame]!=frame_meta:
                    raise ValueError('同帧元信息不一致')
                timeline[frame]=frame_meta
                if raw['record_type']=='empty_frame':
                    if raw['track_id'].strip(): raise ValueError('空帧不能含鱼ID')
                    key=(frame,None)
                    if key in keys or any(k[0]==frame for k in keys): raise ValueError('重复或冲突的空帧')
                    keys.add(key)
                    continue
                if raw['record_type']!='observation': raise ValueError('未知记录类型')
                identity=int(raw['track_id'])
                key=(frame,identity)
                if identity<0 or key in keys or (frame,None) in keys: raise ValueError('重复或冲突的鱼记录')
                keys.add(key)
                record=dict(raw,frame=frame,id=identity,frame_id=frame,track_id=identity,time_s=time_s,
                            fps=fps,image_width=width,image_height=height,filename=Path(raw['image_path']).name)
                for name in numeric:
                    record[name]=float(raw[name]) if raw[name].strip() else math.nan
                    if raw[name].strip() and not math.isfinite(record[name]): raise ValueError(name+'必须为有限数或空值')
                for name in ('position_valid','geometry_valid','head_valid'):
                    if raw[name] not in ('0','1'): raise ValueError(name+'必须为0/1')
                    record[name]=int(raw[name])
                record['reference_length_sample_count']=int(raw['reference_length_sample_count'] or 0)
                if record['reference_length_sample_count']<0: raise ValueError('体长样本数不能为负')
                if record['position_valid'] and not (np.isfinite([record['cx'],record['cy'],record['area_px']]).all() and 0<=record['cx']<width and 0<=record['cy']<height and record['area_px']>0):
                    raise ValueError('有效位置与坐标/面积不一致')
                record['valid']=bool(record['geometry_valid'])
                record['area']=record['area_px']
                line=json.loads(raw['centerline']) if raw['centerline'].strip() else None
                record['centerline']=np.asarray(line,dtype=float) if line is not None else None
                if line is not None and (record['centerline'].ndim!=2 or record['centerline'].shape[1]!=2 or not np.isfinite(record['centerline']).all()):
                    raise ValueError('中轴线必须为有限坐标的二维点数组')
                if record['geometry_valid'] and (line is None or not np.isfinite([record['tail_angle_deg'],record['body_length_px']]).all() or record['body_length_px']<=0):
                    raise ValueError('有效几何缺少中轴、角度或体长')
                if record['head_valid'] and (not np.isfinite([record['head_dx'],record['head_dy']]).all() or math.hypot(record['head_dx'],record['head_dy'])<1e-8):
                    raise ValueError('有效头向缺少方向向量')
                records.append(record)
            except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
                raise ValueError(f'result.csv第{line_number}行: {exc}') from exc
    frames=[timeline[f] for f in sorted(timeline)]
    if not frames: raise ValueError('result.csv没有完整帧信息')
    if any(b['time_s']<=a['time_s'] for a,b in zip(frames,frames[1:])):
        raise ValueError('帧时间必须随帧号严格递增')
    by_id=defaultdict(list)
    for row in records: by_id[row['id']].append(row)
    for identity,fish in by_id.items():
        lengths=[r['body_length_px'] for r in fish if r['geometry_valid'] and math.isfinite(r['body_length_px']) and r['body_length_px']>0]
        expected=float(np.median(lengths)) if lengths else math.nan
        for row in fish:
            actual=row['reference_body_length_px']
            if row['reference_length_sample_count']!=len(lengths) or (lengths and not (math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-9))) or (not lengths and math.isfinite(actual)):
                raise ValueError(f'鱼{identity}参考体长与本批有效几何不一致；不要截取result.csv后沿用整段标尺')
    return sorted(records,key=lambda r:(r['frame'],r['id'])), frames


def write_csv(path, rows, fields):
    """UTF-8 BOM便于Excel打开；不可测数值写空白单元，不伪造零。"""
    destination=Path(path)
    temporary=destination.with_name(destination.name+'.tmp')
    try:
        with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                item={}
                for key in fields:
                    value=row.get(key)
                    if isinstance(value,(list,tuple,dict)):
                        value=json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)
                    elif value is None or isinstance(value,(float,np.floating)) and not math.isfinite(value): value=''
                    item[key]=value
                writer.writerow(item)
        temporary.replace(destination)
    finally:
        if temporary.exists(): temporary.unlink()


def prepare_output(output_dir, names):
    """只覆盖本模块明确列出的输出；禁用视频时也移除旧视频避免混用。"""
    root=Path(output_dir)
    if not root.is_absolute(): raise ValueError('输出目录必须为绝对路径')
    root.mkdir(parents=True,exist_ok=True)
    for name in names:
        file=root/name
        if file.is_file(): file.unlink()
    return root


def get_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def read_image(path):
    image=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError('无法读取原图: '+str(path))
    return image


def write_video(path, frames, paint):
    """按完整源帧顺序写视频；播放帧率采用fps，计算始终采用实际time_s。"""
    width,height=frames[0]['image_width'],frames[0]['image_height']
    temporary=Path(path).with_name(Path(path).stem+'.partial.mp4')
    writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),frames[0]['fps'],(width,height))
    if not writer.isOpened(): raise RuntimeError('无法创建MP4视频')
    try:
        for frame in frames:
            image=read_image(frame['image_path'])
            if image.shape[:2]!=(height,width): raise ValueError('原图尺寸与result.csv不一致')
            paint(image,frame)
            writer.write(image)
    except BaseException:
        writer.release()
        temporary.unlink(missing_ok=True)
        raise
    writer.release()
    cap=cv2.VideoCapture(str(temporary))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));ok=cap.isOpened();cap.release()
    if not ok or count!=len(frames):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f'视频帧数不完整: {count}/{len(frames)}')
    temporary.replace(path)


def fish_color(identity):
    return tuple(int(v) for v in np.random.default_rng(int(identity)).integers(75,235,3))


def panel_grid(count):
    plt=get_plot()
    columns=2 if count>1 else 1
    rows=max(1,math.ceil(count/columns))
    fig,axes=plt.subplots(rows,columns,figsize=(7*columns,3.5*rows),squeeze=False)
    for ax in axes.flat[count:]: ax.set_visible(False)
    return plt,fig,list(axes.flat)

def heading_degrees(dx, dy):
    """图像坐标 x 向右、y 向下；头朝右=0，上=+90，下=-90。"""
    if not (math.isfinite(dx) and math.isfinite(dy)) or math.hypot(dx, dy) < 1e-9:
        return math.nan
    return math.degrees(math.atan2(-dy, dx))

def tilt_degrees(dx, dy):
    """相对水平线的倾角：左右水平均为 0，朝上正、朝下负，范围 [-90, 90]。"""
    if not (math.isfinite(dx) and math.isfinite(dy)) or math.hypot(dx, dy) < 1e-9:
        return math.nan
    return math.degrees(math.atan2(-dy, abs(dx)))

def circular_statistics(angles):
    """单位向量平均，正确处理 +179/-179；合向量为零时均值无定义。"""
    values = np.asarray(angles, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    radians = np.deg2rad(values)
    x, y = float(np.cos(radians).mean()), float(np.sin(radians).mean())
    strength = min(1.0, math.hypot(x, y))
    mean = math.degrees(math.atan2(y, x)) if strength > 1e-8 else math.nan
    return mean, strength

def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def prepare_rows(records):
    """纯计算：从共享几何记录计算倾角、完整航向与连续段展开角，不改输入。

    输入包含frame/time_s/id/valid/head_dx/head_dy；方向采用图像坐标x右y下。
    tilt_deg=atan2(-dy,abs(dx))，左右水平均0，上正下负；raw_heading_deg
    保留左/右象限，unwrapped_heading_deg仅用于连续有效段曲线。invalid
    的角度始终NaN。帧号可从1开始；缺整数帧、无效帧和时间倒退都切段。
    不需要image_paths，不加载geometry/SAM3，不写文件；空输入返回[]。
    """
    rows, previous, segment_numbers = [], {}, defaultdict(int)
    for record in sorted(records, key=lambda r: (int(r["frame"]), int(r["id"]))):
        identity, frame = int(record["id"]), int(record["frame"])
        time_s = float(record["time_s"])
        dx, dy = record.get("head_dx"), record.get("head_dy")
        angle = heading_degrees(float(dx), float(dy)) if finite(dx) and finite(dy) else math.nan
        valid = bool(record.get("valid", False)) and finite(angle) and finite(time_s)
        reason = str(record.get("quality_reason", ""))
        if not valid and not reason:
            reason = "invalid_geometry_or_head_direction"
        row = dict(frame=frame, filename=record.get("filename", ""), time_seconds=time_s,
                   id=identity, cx=record.get("cx"), cy=record.get("cy"), area=record.get("area"),
                   tilt_deg=math.nan, raw_heading_deg=math.nan, unwrapped_heading_deg=math.nan,
                   segment_id=None, direction="invalid", valid=valid, quality_reason=reason,
                   head_confidence=record.get("head_confidence"))
        if valid:
            prev = previous.get(identity)
            continuous = prev is not None and frame == prev["frame"] + 1 and time_s > prev["time_seconds"]
            if continuous:
                # 选择相邻角度的最短弧；这不填补缺帧，也不能证明帧间没有快速整圈旋转。
                delta = math.degrees(math.atan2(math.sin(math.radians(angle - prev["raw_heading_deg"])),
                                                math.cos(math.radians(angle - prev["raw_heading_deg"]))))
                unwrapped = prev["unwrapped_heading_deg"] + delta
            else:
                segment_numbers[identity] += 1
                unwrapped = angle
            vertical = math.sin(math.radians(angle))
            horizontal = abs(vertical) <= math.sin(math.radians(HORIZONTAL_TOLERANCE_DEG)) + 1e-12
            row.update(tilt_deg=tilt_degrees(float(dx), float(dy)),
                       raw_heading_deg=angle, unwrapped_heading_deg=unwrapped,
                       segment_id=segment_numbers[identity],
                       direction="horizontal" if horizontal else ("up" if vertical > 0 else "down"))
            previous[identity] = row
        else:
            previous.pop(identity, None)
        rows.append(row)
    return rows

FIELDS=['frame_id','track_id','time_s','cx','cy','tilt_deg','raw_heading_deg','unwrapped_heading_deg',
        'segment_id','direction','head_confidence','valid','quality_reason']


def calculate(records):
    """原算法输入别名由本文件读取器创建；输出用统一frame_id/track_id。"""
    result=prepare_rows([dict(r,valid=bool(r['head_valid'])) for r in records])
    return [dict(r,frame_id=r['frame'],track_id=r['id'],time_s=r['time_seconds'],valid=int(r['valid'])) for r in result]


def plot(rows,frames,path):
    ids=sorted({r['track_id'] for r in rows});plt,fig,axes=panel_grid(len(ids));times=[r['time_s'] for r in frames]
    for identity,ax in zip(ids,axes):
        lookup={r['frame_id']:r for r in rows if r['track_id']==identity}
        ax.plot(times,[lookup.get(f['frame_id'],{}).get('tilt_deg',math.nan) for f in frames],lw=1.3)
        ax.axhspan(-HORIZONTAL_TOLERANCE_DEG,HORIZONTAL_TOLERANCE_DEG,color='gray',alpha=.15)
        ax.axhline(0,color='gray',lw=.6);ax.set(xlabel='Time (s)',ylabel='Tilt (deg)',ylim=(-95,95),title=f'Fish {identity} | up + / down -')
        ax.grid(alpha=.2)
    if not ids: axes[0].set_visible(True);axes[0].text(.5,.5,'No fish observations',ha='center')
    fig.suptitle('Head-axis tilt relative to horizontal');fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def video(records,rows,frames,path):
    values={(r['frame_id'],r['track_id']):r for r in rows};groups=defaultdict(list)
    for r in records:groups[r['frame']].append(r)
    def paint(image,frame):
        cv2.putText(image,f"Heading | frame {frame['frame_id']} | up + / down -",(20,32),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        for r in groups[frame['frame_id']]:
            if not r['position_valid']:continue
            row=values[(r['frame'],r['id'])];color=fish_color(r['id']);point=(int(r['cx']),int(r['cy']))
            text='N/A'
            if row['valid']:
                head=np.asarray([r['head_dx'],r['head_dy']]);head/=np.linalg.norm(head)
                tip=tuple(np.rint(np.asarray(point)+head*60).astype(int));cv2.arrowedLine(image,point,tip,color,3,tipLength=.25)
                text=f"{row['tilt_deg']:+.1f} deg"
            cv2.circle(image,point,3,color,-1)
            cv2.putText(image,f"ID {r['id']} {text}",(point[0]-40,point[1]-20),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
    write_video(path,frames,paint)


def run(input_csv=INPUT_CSV,output_dir=OUTPUT_DIR,*,save_video=SAVE_VIDEO):
    records,frames=read_result(input_csv);rows=calculate(records)
    output=prepare_output(output_dir,['towards_frames.csv','towards.png','towards.mp4'])
    write_csv(output/'towards_frames.csv',rows,FIELDS);plot(rows,frames,output/'towards.png')
    if save_video:video(records,rows,frames,output/'towards.mp4')
    print(f'Heading: {len(rows)} observations -> {output}')
    return {'frames':rows}


if __name__=='__main__':run()

# ---------------- 输出CSV逐列中文解释 ----------------
# frame_id：原始帧号。
# track_id：鱼ID。
# time_s：当前帧相对第一帧时间，秒。
# cx：原始掩膜质心x，像素，向右正。
# cy：原始掩膜质心y，像素，向下正。
# tilt_deg：相对水平线的倾角，范围[-90,90]度；左右水平均为0，向上正、向下负。
# raw_heading_deg：完整头朝向，右0、上+90、下-90、左±180度，保留左右象限。
# unwrapped_heading_deg：连续有效段内展开后的完整头朝向，用于跨±180度时连续显示；可超出±180度。
# segment_id：连续有效朝向段编号；缺帧和无效方向切断，不跨段展开角度。
# direction：朝向分类：up、down、horizontal或invalid；默认距水平±10度内算horizontal。
# head_confidence：基础测量中的启发式头向评分，不是准确率概率。
# valid：朝向是否有效，1是、0否。
# quality_reason：基础几何质量来源或异常原因，成功时也可能非空；应先看valid。
