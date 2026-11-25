# 顶部视角维护负责人：YHC。此脚本仅属于 FishBehavior/Top，不依赖 Side 子项目。
"""独立停留区域分析：逐鱼按实际有效时间统计区域占用。

默认把原图等分2x2格；这不是自动检测水槽边界。ROI_BOUNDS可限定水槽区域。
采用零阶保持：连续有效帧t→t+1的dt归给起点帧质心所在区域；缺帧/无效
位置/过长时间间隔不累计，末帧不凭空增加一个帧周期。有效时间、未知时间
和观察跨度分别输出。静止与否不影响区域占用：鱼在游动也会经过某个区域。
API: run(input_csv=绝对路径,output_dir=绝对目录,grid_rows=2,grid_cols=2,roi=None)。
返回{'summary':逐鱼逐区域字典列表}，不读取其它行为模块。
"""
from pathlib import Path
INPUT_CSV=Path(r"E:\Phd\FishBehavior\Top\Tracker\Output\Information\result.csv")
OUTPUT_DIR=Path(r"E:\Phd\FishBehavior\Top\Behavior\Output\area")
GRID_ROWS=2
GRID_COLS=2
ROI_BOUNDS=None  # None=整幅图；或(x_min,y_min,x_max,y_max)，原图像素坐标。
MAX_TIME_GAP_FACTOR=1.8  # 相邻帧dt不得超过此系数/fps。

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


FIELDS=['track_id','region_id','region_name','grid_row','grid_col','x_min','y_min','x_max','y_max',
        'start_frame','end_frame','observed_frames','valid_intervals','region_intervals','entries',
        'dwell_time_s','occupancy_fraction','track_span_s','valid_time_s','unknown_time_s','coverage',
        'summary_valid','quality_reason']


def regions(width,height,grid_rows,grid_cols,roi):
    if any(isinstance(v,bool) or int(v)!=v or v<1 for v in [grid_rows,grid_cols]):raise ValueError('网格行列必须为正整数')
    x0,y0,x1,y1=(0.,0.,float(width),float(height)) if roi is None else map(float,roi)
    if not np.isfinite([x0,y0,x1,y1]).all() or not (0<=x0<x1<=width and 0<=y0<y1<=height):raise ValueError('ROI超出原图或边界无效')
    output=[]
    for row in range(int(grid_rows)):
        for col in range(int(grid_cols)):
            name=['top_left','top_right','bottom_left','bottom_right'][row*2+col] if grid_rows==grid_cols==2 else f'row_{row+1}_col_{col+1}'
            output.append(dict(region_id=row*grid_cols+col,region_name=name,grid_row=row,grid_col=col,
                x_min=x0+(x1-x0)*col/grid_cols,x_max=x0+(x1-x0)*(col+1)/grid_cols,
                y_min=y0+(y1-y0)*row/grid_rows,y_max=y0+(y1-y0)*(row+1)/grid_rows))
    return output


def locate(row,cells):
    for cell in cells:
        if cell['x_min']<=row['cx']<cell['x_max'] and cell['y_min']<=row['cy']<cell['y_max']:
            return cell['region_id']
    return -1


def calculate(records,frames,grid_rows=GRID_ROWS,grid_cols=GRID_COLS,roi=ROI_BOUNDS):
    cells=regions(frames[0]['image_width'],frames[0]['image_height'],grid_rows,grid_cols,roi)
    groups=defaultdict(list);out=[]
    for row in records:groups[row['id']].append(row)
    for identity,fish in sorted(groups.items()):
        fish=sorted(fish,key=lambda r:r['frame']);duration=defaultdict(float);counts=defaultdict(int);entries=defaultdict(int)
        valid_time=0.;valid_count=0;prior_region=None
        for a,b in zip(fish,fish[1:]):
            dt=b['time_s']-a['time_s']
            good=a['position_valid'] and b['position_valid'] and b['frame']==a['frame']+1 and 0<dt<MAX_TIME_GAP_FACTOR/a['fps']
            if not good:prior_region=None;continue
            region=locate(a,cells);duration[region]+=dt;counts[region]+=1;valid_time+=dt;valid_count+=1
            if prior_region!=region:entries[region]+=1
            prior_region=region
        span=fish[-1]['time_s']-fish[0]['time_s'];valid=valid_time>0
        # ROI外也是可确定的位置，单列outside_roi，避免内部各区占比被无声放大。
        regions_out=cells+[dict(region_id=-1,region_name='outside_roi',grid_row=None,grid_col=None,x_min=None,y_min=None,x_max=None,y_max=None)]
        for cell in regions_out:
            k=cell['region_id']
            out.append(dict(cell,track_id=identity,start_frame=fish[0]['frame'],end_frame=fish[-1]['frame'],observed_frames=len(fish),
                valid_intervals=valid_count,region_intervals=counts[k],entries=entries[k],dwell_time_s=duration[k] if valid else math.nan,
                occupancy_fraction=duration[k]/valid_time if valid else math.nan,track_span_s=span,valid_time_s=valid_time,
                unknown_time_s=max(0.,span-valid_time),coverage=valid_time/span if span>0 else math.nan,
                summary_valid=int(valid),quality_reason='' if valid else 'no_valid_intervals'))
    return out


def plot(summary,records,frames,path):
    from matplotlib.patches import Rectangle
    ids=sorted({r['track_id'] for r in summary});plt,fig,axes=panel_grid(len(ids));cmap=plt.get_cmap('YlOrRd')
    width,height=frames[0]['image_width'],frames[0]['image_height']
    for identity,ax in zip(ids,axes):
        fish=[r for r in records if r['id']==identity and r['position_valid']]
        ax.scatter([r['cx'] for r in fish],[r['cy'] for r in fish],s=4,color='gray',alpha=.45)
        values=[r for r in summary if r['track_id']==identity]
        for row in values:
            if row['region_id']<0:continue
            fraction=row['occupancy_fraction'];color=cmap(fraction) if math.isfinite(fraction) else '.9'
            x,y=row['x_min'],row['y_min'];w,h=row['x_max']-x,row['y_max']-y
            ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='black',alpha=.55,lw=.8))
            text=f"{row['dwell_time_s']:.2f}s ({fraction:.1%})" if math.isfinite(fraction) else 'N/A'
            ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
        ax.set(xlim=(0,width),ylim=(height,0),xlabel='Image x (pixel)',ylabel='Image y (pixel)',title=f"Fish {identity} | unknown={values[0]['unknown_time_s']:.3f}s")
        ax.set_aspect('equal')
    if not ids:axes[0].set_visible(True);axes[0].text(.5,.5,'No fish observations',ha='center')
    fig.suptitle('Region occupancy | interval assigned to starting position | percentage of valid time')
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def run(input_csv=INPUT_CSV,output_dir=OUTPUT_DIR,*,grid_rows=GRID_ROWS,grid_cols=GRID_COLS,roi=ROI_BOUNDS):
    records,frames=read_result(input_csv);summary=calculate(records,frames,grid_rows,grid_cols,roi)
    output=prepare_output(output_dir,['area_summary.csv','area.png'])
    write_csv(output/'area_summary.csv',summary,FIELDS);plot(summary,records,frames,output/'area.png')
    print(f'Area: {len(summary)} fish-region rows -> {output}')
    return {'summary':summary}


if __name__=='__main__':run()

# ---------------- 输出CSV逐列中文解释 ----------------
# track_id：鱼ID。
# region_id：区域编号，从0开始，逐行由左到右；-1表示ROI外。
# region_name：区域名称：默认top_left、top_right、bottom_left、bottom_right和outside_roi。
# grid_row：网格行索引，从0开始；ROI外为空。
# grid_col：网格列索引，从0开始；ROI外为空。
# x_min：区域左边界，原图像素，包含边界。
# y_min：区域上边界，原图像素，包含边界。
# x_max：区域右边界，原图像素，不包含边界。
# y_max：区域下边界，原图像素，不包含边界。
# start_frame：该鱼首观测帧。
# end_frame：该鱼末观测帧。
# observed_frames：该鱼实际观测帧数。
# valid_intervals：该鱼可判断区域的有效相邻帧区间总数。
# region_intervals：归属本区域的有效时间区间数。
# entries：本区域连续可判定占用片段的开始次数，包含首次观测/缺失后重新观测；不等同于已确认的跨边界次数。
# dwell_time_s：起点质心位于本区域的有效相邻帧时间差之和，秒；无有效统计时为空。
# occupancy_fraction：本区域停留时间除以该鱼全部有效区域判断时间，0~1；不是百分数，乘100后为百分比。
# track_span_s：该鱼首末观测时间差，秒；不外推至视频起止边界。
# valid_time_s：所有可判定区域的时间之和，含ROI外，秒。
# unknown_time_s：观察跨度减去有效区域判断时间，秒；缺帧等归入未知。
# coverage：有效区域判断时间/观察跨度；零跨度为空。
# summary_valid：是否至少存在一个有效时间区间，1是、0否。
# quality_reason：正常为空；无有效区间为no_valid_intervals。
