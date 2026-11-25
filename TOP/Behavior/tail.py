# 顶部视角维护负责人：YHC。此脚本仅属于 FishBehavior/Top，不依赖 Side 子项目。
"""独立摆尾分析：result.csv -> 逐帧角度/窗口频率/窗口幅度、总图、视频。

本文件不调用其它项目脚本。频率沿用原项目同相位峰间周期中位数的倒数，
新增逐帧回看窗口；窗口内必须为同ID连续有效几何，缺帧即切断。
幅度=窗口内各完整同相位周期的(最大角度-最小角度)/2的中位数，单位度。
角度采用插值、5点Savitzky-Golay平滑后去线性趋势的波形；半峰峰值与
旧算法95/5百分位半差不同，后者仅作为robust_amplitude_deg诊断列保留。
API: run(input_csv=绝对路径, output_dir=绝对目录, window_frames=150, save_video=True)。
run返回{'frames':逐帧字典列表}；纯数值调用calculate(records,window_frames=150)。
"""
from pathlib import Path
INPUT_CSV=Path(r"E:\Phd\FishBehavior\Top\Tracker\Output\Information\result.csv")
OUTPUT_DIR=Path(r"E:\Phd\FishBehavior\Top\Behavior\Output\tail")
WINDOW_FRAMES=150  # 最大回看样本数，含当前帧；30fps时首末跨度149/30秒。
MIN_FREQUENCY_HZ=0.4
MAX_FREQUENCY_HZ=6.0
MIN_AMPLITUDE_DEG=3.0
MIN_PEAK_PROMINENCE_DEG=3.0
MIN_COMPLETE_CYCLES=2
MAX_PERIOD_CV=0.35
SAVE_VIDEO=True
from scipy.signal import detrend, find_peaks, periodogram, savgol_filter

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

def valid_segments(records, fps):
    """将同一ID分成连续有效段，不跨缺帧、无效几何或时间断点。

    records应来自geometry接口，至少含frame/time_s/valid/tail_angle_deg。
    fps是标称采样率；两帧时间差须在(0, 1.8/fps)内。帧号可从1开始，
    但缺少一个整数帧号会切段，绝不跨缺帧插值后计算频率。空输入返回[]。
    """
    segments, current = [], []
    for row in sorted(records, key=lambda r: r["frame"]):
        good = row["valid"] and math.isfinite(row["tail_angle_deg"]) and math.isfinite(row["time_s"])
        if current and (not good or row["frame"] != current[-1]["frame"] + 1
                        or not 0 < row["time_s"] - current[-1]["time_s"] < 1.8 / fps):
            segments.append(current); current = []
        if good:
            current.append(row)
    if current:
        segments.append(current)
    return segments

def estimate_frequency(segment, fps):
    """估计一个连续有效段的摆尾频率，不读文件、不修改输入。

    主频率取同相位正峰间隔倒数的中位周期；频谱仅作独立核对，避免
    对abs(angle)计算导致频率翻倍。smoothed_angles与输入段逐行对应。
    frequency_quality='ok'才表示frequency_hz可用；其余状态保留NaN，
    包括too_short、amplitude_too_small、insufficient_complete_cycles等。
    空段/单帧返回too_short。算法及默认阈值保留原项目的实现。
    """
    result = {"frequency_hz": float("nan"), "spectral_peak_hz": float("nan"),
              "amplitude_deg": float("nan"), "cycles": 0, "period_cv": float("nan"),
              "frequency_quality": "too_short", "peak_times_s": [], "smoothed_angles": []}
    if len(segment) < 7:
        return result
    t = np.asarray([r["time_s"] for r in segment])
    angle = np.degrees(np.unwrap(np.radians([r["tail_angle_deg"] for r in segment])))
    sample_rate = (len(t) - 1) / (t[-1] - t[0])
    # 统一到近等间距，处理采集时间戳33/34毫秒的正常抖动。
    grid = np.linspace(t[0], t[-1], len(t))
    smooth = savgol_filter(np.interp(grid, t, angle), 5, 2)
    wave = detrend(smooth)
    amplitude = (np.percentile(wave, 95) - np.percentile(wave, 5)) / 2
    result.update(amplitude_deg=float(amplitude), smoothed_angles=np.interp(t, grid, smooth).tolist())
    if amplitude < MIN_AMPLITUDE_DEG:
        result["frequency_quality"] = "amplitude_too_small"
        return result
    if t[-1] - t[0] < MIN_COMPLETE_CYCLES / MAX_FREQUENCY_HZ:
        return result
    # 至少每周期5个样本；30fps上限6Hz。频谱仍查看更高频段，以拒绝不稳定信号。
    upper = min(MAX_FREQUENCY_HZ, sample_rate / 5)
    frequency, power = periodogram(wave, fs=(len(grid)-1)/(grid[-1]-grid[0]), window="hann",
                                  nfft=max(256, 4 * len(wave)))
    band = (frequency >= MIN_FREQUENCY_HZ) & (frequency < sample_rate / 2)
    if band.any():
        spectral = float(frequency[band][np.argmax(power[band])])
        result["spectral_peak_hz"] = spectral
        if spectral > upper * 1.001:
            result["frequency_quality"] = "frequency_above_safe_limit"
            return result
    peaks, _ = find_peaks(wave, prominence=max(MIN_PEAK_PROMINENCE_DEG, amplitude * .25),
                         distance=max(1, int(math.floor(sample_rate / upper))))
    result["peak_times_s"] = grid[peaks].tolist()
    intervals = np.diff(grid[peaks])
    result["cycles"] = int(len(intervals))
    if len(intervals) < MIN_COMPLETE_CYCLES:
        result["frequency_quality"] = "insufficient_complete_cycles"
        return result
    estimates = 1 / intervals
    cv = float(np.std(intervals) / np.mean(intervals))
    result["period_cv"] = cv
    if np.any((estimates < MIN_FREQUENCY_HZ) | (estimates > upper * 1.001)):
        result["frequency_quality"] = "period_outside_frequency_range"
    elif cv > MAX_PERIOD_CV:
        result["frequency_quality"] = "irregular_cycles"
    else:
        value = float(1 / np.median(intervals))
        spectral = result["spectral_peak_hz"]
        if math.isfinite(spectral) and abs(spectral - value) > max(.35, .3 * value):
            result["frequency_quality"] = "spectrum_peak_disagrees"
        else:
            result.update(frequency_hz=value, frequency_quality="ok")
    return result

FIELDS=['frame_id','track_id','time_s','window_start_frame','window_end_frame','window_samples',
        'window_complete','frame_range','segment_id','tail_angle_deg','smoothed_angle_deg',
        'frequency_hz','amplitude_deg','peak_to_peak_deg','robust_amplitude_deg','cycles',
        'period_cv','spectral_peak_hz','geometry_valid','frequency_valid','amplitude_valid',
        'frequency_quality','amplitude_quality']


def cycle_amplitude(segment, result):
    """完整周期由相邻正峰界定；对每个周期取半峰峰值，再取中位数。"""
    peaks=result['peak_times_s']
    if len(peaks)<2: return math.nan,'insufficient_complete_cycle'
    t=np.asarray([r['time_s'] for r in segment]);grid=np.linspace(t[0],t[-1],len(t))
    angle=np.degrees(np.unwrap(np.radians([r['tail_angle_deg'] for r in segment])))
    wave=detrend(savgol_filter(np.interp(grid,t,angle),5,2))
    amplitudes=[]
    for first,last in zip(peaks,peaks[1:]):
        part=wave[(grid>=first-1e-10)&(grid<=last+1e-10)]
        if len(part)>=3: amplitudes.append(float(np.ptp(part)/2))
    return (float(np.median(amplitudes)),'ok') if amplitudes else (math.nan,'insufficient_complete_cycle')


def calculate(records,window_frames=WINDOW_FRAMES):
    """每帧结果只用当前及过去帧；不足窗口可提前估计，但必须通过周期质量检查。

    window_complete只表示已达到最大回看长度，不替代frequency_valid。
    频率至少2个完整周期，幅度至少1个完整周期；两者有效性分别保存。
    """
    if isinstance(window_frames,bool) or int(window_frames)!=window_frames or window_frames<7:
        raise ValueError('WINDOW_FRAMES必须是>=7的整数')
    groups=defaultdict(list);output=[]
    for r in records: groups[r['id']].append(r)
    for identity,fish in sorted(groups.items()):
        previous=None;history=[];segment_number=0
        for record in sorted(fish,key=lambda r:r['frame']):
            good=bool(record['geometry_valid']) and math.isfinite(record['tail_angle_deg'])
            continuous=previous is not None and record['frame']==previous['frame']+1 and 0<record['time_s']-previous['time_s']<1.8/record['fps']
            if not good or not continuous:
                history=[]
                if good: segment_number+=1
            if good: history.append(record);history=history[-int(window_frames):]
            result=estimate_frequency(history,record['fps']) if good else dict(frequency_hz=math.nan,
                amplitude_deg=math.nan,cycles=0,period_cv=math.nan,spectral_peak_hz=math.nan,
                frequency_quality='invalid_geometry',smoothed_angles=[],peak_times_s=[])
            amplitude,amplitude_quality=cycle_amplitude(history,result) if good else (math.nan,'invalid_geometry')
            output.append(dict(frame_id=record['frame'],track_id=identity,time_s=record['time_s'],
                window_start_frame=history[0]['frame'] if history else record['frame'],window_end_frame=record['frame'],
                window_samples=len(history),window_complete=int(len(history)==window_frames),
                frame_range=[r['frame'] for r in history],segment_id=segment_number if good else None,
                tail_angle_deg=record['tail_angle_deg'] if good else math.nan,
                smoothed_angle_deg=result['smoothed_angles'][-1] if result['smoothed_angles'] else math.nan,
                frequency_hz=result['frequency_hz'],amplitude_deg=amplitude,peak_to_peak_deg=2*amplitude,
                robust_amplitude_deg=result['amplitude_deg'],cycles=result['cycles'],period_cv=result['period_cv'],
                spectral_peak_hz=result['spectral_peak_hz'],geometry_valid=int(good),
                frequency_valid=int(result['frequency_quality']=='ok'),amplitude_valid=int(amplitude_quality=='ok'),
                frequency_quality=result['frequency_quality'],amplitude_quality=amplitude_quality))
            previous=record if good else None
    return sorted(output,key=lambda r:(r['frame_id'],r['track_id']))


def plot(rows,frames,path):
    ids=sorted({r['track_id'] for r in rows});plt,fig,axes=panel_grid(len(ids))
    times=[f['time_s'] for f in frames]
    angles=[abs(r['tail_angle_deg']) for r in rows if math.isfinite(r['tail_angle_deg'])]
    angle_limit=max(10.,math.ceil(max(angles,default=45.)/10)*10)
    for identity,ax in zip(ids,axes):
        lookup={r['frame_id']:r for r in rows if r['track_id']==identity}
        series=lambda field:[lookup.get(f['frame_id'],{}).get(field,math.nan) for f in frames]
        ax.plot(times,series('tail_angle_deg'),color='.7',lw=.7,label='Raw angle')
        ax.plot(times,series('smoothed_angle_deg'),lw=1,label='Smoothed endpoint angle')
        ax.plot(times,series('amplitude_deg'),color='green',ls='--',lw=1,label='Cycle half peak-to-peak')
        second=ax.twinx();second.plot(times,series('frequency_hz'),color='darkorange',lw=1.2,label='Frequency')
        second.set_ylabel('Frequency (Hz)',color='darkorange');second.set_ylim(0,MAX_FREQUENCY_HZ*1.08)
        ax.set(xlabel='Time (s)',ylabel='Angle / amplitude (deg)',ylim=(-angle_limit,angle_limit),title=f'Fish {identity}')
        ax.grid(alpha=.2);ax.legend(fontsize=7,loc='upper left')
    if not ids: axes[0].set_visible(True);axes[0].text(.5,.5,'No fish observations',ha='center')
    fig.suptitle('Tail analysis | trailing window; gaps / blank frequency = unavailable')
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def video(records,rows,frames,path):
    by_frame=defaultdict(list);values={(r['frame_id'],r['track_id']):r for r in rows}
    for r in records: by_frame[r['frame']].append(r)
    def paint(image,frame):
        cv2.putText(image,f"Tail | frame {frame['frame_id']} | t={frame['time_s']:.3f}s",(20,32),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        for record in by_frame[frame['frame_id']]:
            if not record['position_valid']: continue
            row=values[(record['frame'],record['id'])];color=fish_color(record['id']);point=(int(record['cx']),int(record['cy']))
            if record['centerline'] is not None and record['geometry_valid']:
                line=np.rint(record['centerline']).astype(np.int32)
                cv2.polylines(image,[line],False,color,2);cv2.circle(image,tuple(line[0]),5,(0,0,255),-1)
                cv2.circle(image,tuple(line[-1]),4,(255,160,20),-1)
                for index,dx,dy,axis_color in [(14,record['head_dx'],record['head_dy'],(0,0,255)),(40,record['tail_dx'],record['tail_dy'],(255,160,20))]:
                    start=record['centerline'][index];vector=np.asarray([dx,dy])*30
                    cv2.arrowedLine(image,tuple(np.rint(start-vector).astype(int)),tuple(np.rint(start+vector).astype(int)),axis_color,2,tipLength=.2)
            f=f"{row['frequency_hz']:.2f}" if row['frequency_valid'] else 'N/A'
            a=f"{row['amplitude_deg']:.1f}" if row['amplitude_valid'] else 'N/A'
            label=f"ID {record['id']} f={f} Hz A={a} deg"
            label_width=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.52,2)[0][0]
            label_x=max(5,min(point[0]-55,image.shape[1]-label_width-5));label_y=max(55,min(point[1]-20,image.shape[0]-15))
            cv2.putText(image,label,(label_x,label_y),cv2.FONT_HERSHEY_SIMPLEX,.52,color,2)
    write_video(path,frames,paint)


def run(input_csv=INPUT_CSV,output_dir=OUTPUT_DIR,*,window_frames=WINDOW_FRAMES,save_video=SAVE_VIDEO):
    records,frames=read_result(input_csv);rows=calculate(records,window_frames)
    output=prepare_output(output_dir,['tail_frames.csv','tail.png','tail.mp4'])
    write_csv(output/'tail_frames.csv',rows,FIELDS);plot(rows,frames,output/'tail.png')
    if save_video: video(records,rows,frames,output/'tail.mp4')
    print(f'Tail: {len(rows)} observations -> {output}')
    return {'frames':rows}


if __name__=='__main__':
    run()

# ---------------- 输出CSV逐列中文解释 ----------------
# frame_id：当前结果所属的原始帧号，窗口终点；从0开始。
# track_id：鱼的跟踪ID，0是合法ID。
# time_s：当前帧相对第一帧的实际时间，秒。
# window_start_frame：本次实际参与计算的连续窗口起点；不是整段视频起点。
# window_end_frame：实际窗口终点，即当前帧。
# window_samples：窗口中实际有效几何样本数；无效当前帧为0。
# window_complete：是否达到配置的最大回看样本数；1是、0否。未满窗口也可能已有足够周期，因此它不替代频率有效性。
# frame_range：本次实际使用的全部帧号JSON数组。
# segment_id：该鱼连续有效几何段编号；缺帧、时间断点或无效几何会重新分段。
# tail_angle_deg：当前帧头轴和尾轴的有符号夹角，度；几何无效时为空。
# smoothed_angle_deg：在当前窗口内插值并作5点平滑后的终点角度，度；不是整段双向平滑的终点值。
# frequency_hz：同相位正峰周期中位数的倒数，Hz；只在frequency_valid=1时有值。
# amplitude_deg：完整同相位周期的半峰峰值中位数，度；去趋势后计算；至少一个完整周期。
# peak_to_peak_deg：幅度的两倍，即完整周期峰峰值中位数，度。
# robust_amplitude_deg：旧频率算法的去趋势波形95/5百分位半差，度；用于质量诊断，不是本次定义的周期摆尾幅度。
# cycles：窗口中检出的完整同相位峰间周期数量；频率至少需要2个。
# period_cv：周期时长总体标准差除以平均周期，无量纲；越大说明周期越不规则。
# spectral_peak_hz：频谱最大功率峰的频率，Hz；仅用于与峰间周期估计交叉检查。
# geometry_valid：当前帧中轴及头尾测量是否有效，1是、0否。
# frequency_valid：频率是否通过全部质量检查，1是、0否。
# amplitude_valid：周期幅度是否可测，1是、0否；可以与频率有效性不同。
# frequency_quality：频率质量状态，ok为可用，其他值说明帧数、周期数、幅度、频段或规则性问题。
# amplitude_quality：幅度质量状态：ok、invalid_geometry或insufficient_complete_cycle。
