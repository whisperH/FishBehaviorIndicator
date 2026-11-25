"""侧面 鱼体角度 可视化，维护负责人：chenxi。

独立运行本脚本后打开终端中的本机网址；按 Ctrl+C 停止。
这里只读取 Tracker 和 Behavior 的当前结果，生成页面数据缓存并提供视频流。
不会运行 SAM3、调用行为计算脚本或改写任何分析 CSV。缺少结果会明确报错。
HTML 模板不携带旧演示数据；每次启动重新载入结果，运行期间是固定快照。
"""
from __future__ import annotations
import argparse
import base64
import csv
import io
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

# 本文件可单独运行，全部默认路径为交付位置的绝对路径。
# 迁移项目时修改 SIDE_ROOT，或使用命令行 --side-root 指向迁移后的 Side。
SIDE_ROOT=Path(r"E:\Phd\FishBehavior\Side")
NAME="angle"
PORT=8770
_HTML_BYTES=b""
_FILES={}
_VIDEO_PATH=Path()
_RESULTS_DIR=Path()

def read_csv(path):
    """只读 UTF-8 BOM CSV，返回字典行；避免把空表当有效展示结果。"""
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}。请先在主入口运行相应侧面任务。")
    with path.open(encoding="utf-8-sig",newline="") as handle:
        reader=csv.DictReader(handle)
        if not reader.fieldnames:raise ValueError(f"CSV没有列标题：{path}")
        rows=list(reader)
    if not rows:raise ValueError(f"CSV没有可展示的记录：{path}")
    return rows

def required(path):
    """启动前核对依赖，绝不自动启动指标分析。"""
    if not path.is_file():raise FileNotFoundError(f"缺少 {path}。请先运行对应的数据准备/指标分析。")
    return path

def replace_script(html, element_id, content):
    """注入当前结果，替换而非追加；JSON转义<避免闭合script标签。"""
    pattern=r'(<script[^>]*id="'+re.escape(element_id)+r'"[^>]*>).*?(</script>)'
    html,count=re.subn(pattern,lambda m:m[1]+content+m[2],html,flags=re.S)
    if count!=1:raise ValueError(f"页面模板缺少唯一数据槽 {element_id}")
    return html

def page_csv(path):
    # 页面只按既有指标显示；CSV未满窗口的状态与数值保持原样。
    return required(path).read_text(encoding="utf-8-sig")

def csv_metadata(path):
    """核对基础CSV全表属于同一准备批次，避免一张页面拼接旧、新跟踪。

    run_id是一次Tracker准备任务的批次标记，不是鱼编号。
    视频路径、大小和修改时间共同用于检查输入是否被替换；FPS/尺寸/帧数
    则检查视频坐标系与时间轴。只做来源核验，不修改数据或重算指标。
    """
    fields=["run_id","video_path","video_size_bytes","video_mtime_ns",
            "fps","width","height","frame_count"]
    baseline=None
    with required(path).open(encoding="utf-8-sig",newline="") as handle:
        reader=csv.DictReader(handle)
        missing=set(fields)-set(reader.fieldnames or [])
        if missing:raise ValueError(f"{path} 缺少来源字段 {sorted(missing)}，请重跑对应Tracker。")
        for row in reader:
            current={key:row.get(key) for key in fields}
            if any(value in (None,"") for value in current.values()):
                raise ValueError(f"{path} 存在空的来源信息，请重跑对应Tracker。")
            if baseline is None:baseline=current
            elif current!=baseline:
                raise ValueError(f"{path} 混合了不同批次/视频的记录，请重跑对应Tracker。")
    if baseline is None:raise ValueError(f"{path} 没有数据，请先完成对应Tracker。")
    return baseline

def same_path(left,right):
    # Windows Path的比较同时处理斜杠、相对成分和大小写。
    return Path(left).resolve()==Path(right).resolve()

def validate_base_video(base,video_path,video_meta):
    """拒绝将旧基础数据叠加到同名但已被替换的视频。"""
    stat=video_path.stat()
    if not same_path(base["video_path"],video_path):
        raise ValueError("Tracker记录的视频路径与当前输入不同，请重跑对应Tracker。")
    if int(base["video_size_bytes"])!=stat.st_size or int(base["video_mtime_ns"])!=stat.st_mtime_ns:
        raise ValueError("当前视频已改变，Tracker结果已过期；请先重跑Tracker，再重跑指标。")
    for key in ["width","height","frame_count"]:
        if int(base[key])!=int(video_meta[key]):
            raise ValueError(f"Tracker的{key}与当前视频不一致，请重跑Tracker和指标。")
    if abs(float(base["fps"])-video_meta["fps"])>1e-6:
        raise ValueError("Tracker帧率与当前视频不一致，请重跑Tracker和指标。")

def validate_behavior(metadata,base,base_path,used_files):
    """只展示当前基础结果对应的指标，不让旧CSV和新掩膜混用。

    指标JSON的meta记录来源批次及视频。CSV没有另加run_id列时，再检查
    它的修改时间不早于基础CSV。单纯复制保留原mtime的完整项目可正常运行。
    这属于来源核验，不会替用户自动执行分析。
    """
    script={"angle":"angle_analysis.py","speed":"speed_analysis.py",
            "water_layer":"water_layer_analysis.py","gill":"gill_frequency.py"}[NAME]
    def stale(detail):
        raise ValueError(f"{detail}；请先重新运行 Side/Behavior/{script}，再启动本页面。")
    if not isinstance(metadata,dict) or metadata.get("run_id")!=base["run_id"]:
        stale("行为结果与当前Tracker批次不一致或缺少run_id")
    source_video=metadata.get("video_path",metadata.get("video"))
    if not source_video or not same_path(source_video,base["video_path"]):
        stale("行为结果来自其他视频或缺少视频路径")
    source_csv=metadata.get("input_csv",metadata.get("center_source",metadata.get("source")))
    if not source_csv or not same_path(source_csv,base_path):
        stale("行为结果对应的基础CSV路径不一致")
    for key in ["width","height","frame_count"]:
        value=metadata.get(key,metadata.get("frameCount") if key=="frame_count" else None)
        if value is None or int(value)!=int(base[key]):stale(f"行为结果的{key}与Tracker不一致")
    fps=metadata.get("fps")
    if fps is None or abs(float(fps)-float(base["fps"]))>1e-6:
        stale("行为结果的FPS与Tracker不一致")
    # 若指标记录了输入文件指纹，同样应匹配；鳃动JSON使用run_id连接基础CSV。
    for key in ["video_size_bytes","video_mtime_ns"]:
        if key in metadata and int(metadata[key])!=int(base[key]):stale(f"行为结果的{key}与Tracker不一致")
    base_time=base_path.stat().st_mtime_ns
    for path in used_files:
        if required(path).stat().st_mtime_ns<base_time:
            stale(f"{path.name} 早于当前基础CSV，可能仍是旧结果")

def masks_for_page(frames_path, geometry_path, video_meta):
    """将当前分割轮廓和基础测量转换为原网页的展示结构，不重新分割。"""
    observations=read_csv(frames_path)
    geometry=read_csv(geometry_path)
    lookup={(int(r["frame_id"]),int(r["track_id"])):r for r in geometry if r.get("track_id") not in (None,"")}
    samples=[]
    for row in observations:
        if row.get("track_id") in (None,"") or not row.get("polygons"):continue
        polygons=json.loads(row["polygons"])
        if not polygons:continue
        key=(int(row["frame_id"]),int(row["track_id"]))
        sample={"frame":key[0],"track_id":key[1],"polygons":polygons,
                "box":[float(row[k]) for k in ["bbox_x","bbox_y","bbox_w","bbox_h"]]}
        for k in ["center_x","center_y","head_x","head_y","tail_x","tail_y","angle_horizontal_deg","directed_angle_deg","orientation_deg","head_width","tail_width","elongation"]:
            value=lookup.get(key,{}).get(k)
            if value not in (None,""):sample[k]=float(value)
        samples.append(sample)
    return {"meta":video_meta,"samples":samples}

def prepare(side_root, html_path=None, cache_dir=None):
    """读取当前结果并准备可视化文件；不导入任何 Behavior 或 Tracker 脚本。"""
    import cv2
    side_root=Path(side_root).resolve()
    cache=Path(cache_dir).resolve() if cache_dir else side_root/"Visualization"/"data"/NAME
    template=Path(html_path).resolve() if html_path else side_root/"Visualization"/f"{NAME}_viewer.html"
    result_dir=side_root/"Behavior"/"Output"/NAME
    input_dir=side_root/"Tracker"/"Input"/("gill" if NAME=="gill" else "body")
    video_path=required(input_dir/("video.mp4" if NAME=="gill" else "angle.mp4"))
    html=required(template).read_text("utf-8")
    capture=cv2.VideoCapture(str(video_path))
    if not capture.isOpened():raise RuntimeError(f"无法读取视频：{video_path}")
    try:
        meta={"video_name":video_path.name,"video":str(video_path),"width":int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
              "height":int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),"fps":float(capture.get(cv2.CAP_PROP_FPS)),
              "frame_count":int(capture.get(cv2.CAP_PROP_FRAME_COUNT))}
        ok,poster=capture.read()
    finally:capture.release()
    if not ok or meta["fps"]<=0 or meta["frame_count"]<1:raise ValueError("视频时间或首帧不可用")
    files={}
    if NAME=="gill":
        base_path=required(side_root/"Tracker"/"Output"/"gill"/"result.csv")
        base=csv_metadata(base_path)
        validate_base_video(base,video_path,meta)
        data_path=required(result_dir/"gill_data.json")
        data=json.loads(data_path.read_text("utf-8-sig"))
        validate_behavior(data.get("meta"),base,base_path,[data_path])
        if not data.get("fish"):raise ValueError("鳃动结果没有鱼，请先检查分析结果")
        if abs(float(data["meta"]["fps"])-meta["fps"])>1e-6 or int(data["meta"]["frameCount"])!=meta["frame_count"]:
            raise ValueError("鳃动结果与当前视频的帧率/长度不匹配，请重跑分析")
        files["/data.json"]=data_path
    else:
        body=side_root/"Tracker"/"Output"/"body"
        frames_path=required(body/"frames.csv")
        geometry_path=required(body/"result.csv")
        frame_meta=csv_metadata(frames_path)
        geometry_meta=csv_metadata(geometry_path)
        if frame_meta!=geometry_meta:
            raise ValueError("鱼体frames.csv与result.csv不是同一批次，请先重跑Tracker/geometry.py，再重跑指标。")
        validate_base_video(geometry_meta,video_path,meta)
        summary_path=required(result_dir/f"{NAME}_summary.json")
        summary=json.loads(summary_path.read_text("utf-8-sig"))
        frame_names={"angle":"frame_angles.csv","water_layer":"frame_water_layer.csv","speed":"frame_speed.csv"}
        used_files=[summary_path,result_dir/frame_names[NAME]]
        if NAME=="speed":used_files.append(result_dir/"track_speed_summary.csv")
        validate_behavior(summary.get("meta"),geometry_meta,geometry_path,used_files)
        masks=masks_for_page(frames_path,geometry_path,meta)
        if not masks["samples"]:raise ValueError("没有可展示的鱼体分割，请先运行鱼体数据准备")
        if NAME=="angle":
            frame_csv=result_dir/"frame_angles.csv";read_csv(frame_csv)
            html=replace_script(html,"angle-data",base64.b64encode(page_csv(frame_csv).encode()).decode())
            # MOT展示框由当前分割行生成，确保与掩膜处于相同分辨率，不读取旧内嵌标注。
            stream=io.StringIO();writer=csv.writer(stream)
            for s in masks["samples"]:writer.writerow([s["frame"],s["track_id"],*s["box"],1,1,1])
            html=replace_script(html,"mot-data",base64.b64encode(stream.getvalue().encode()).decode())
        elif NAME=="water_layer":
            frame_csv=result_dir/"frame_water_layer.csv";rows=read_csv(frame_csv)
            html=replace_script(html,"depth-data",base64.b64encode(page_csv(frame_csv).encode()).decode())
            metadata_path=result_dir/"water_layer_summary.json"
            if metadata_path.is_file():
                metadata=json.loads(metadata_path.read_text("utf-8-sig"));settings=metadata.get("meta",metadata)
            else:settings=rows[0]
            if not all(k in settings for k in ["surface_y","bottom_y"]):
                raise ValueError("水层结果未记录水面/水底参数，请使用当前 water_layer_analysis.py 重新生成")
            meta.update(surface_y=float(settings["surface_y"]),bottom_y=float(settings["bottom_y"]))
        elif NAME=="speed":
            read_csv(result_dir/"frame_speed.csv");read_csv(result_dir/"track_speed_summary.csv")
            required(result_dir/"speed_summary.json")
        if NAME in {"angle","water_layer"}:
            html=replace_script(html,"mask-data",base64.b64encode(json.dumps(masks,ensure_ascii=False).encode()).decode())
        # 缓存只在需要页面时创建；每次启动覆盖当前展示快照。
        cache.mkdir(parents=True,exist_ok=True)
        mask_path=cache/"mask_shapes.json"
        mask_path.write_text(json.dumps(masks,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
        files["/mask-shapes"]=mask_path
    cache.mkdir(parents=True,exist_ok=True)
    poster_path=cache/"video_poster.jpg"
    ok,encoded=cv2.imencode(".jpg",poster)
    if not ok:raise RuntimeError("无法生成视频预览图")
    poster_path.write_bytes(encoded.tobytes())
    files["/poster"]=poster_path
    files["/results/angle/video_poster.jpg"]=poster_path
    html=replace_script(html,"viewer-meta",json.dumps(meta,ensure_ascii=False).replace("<","\\u003c"))
    # 渲染后的页面保存在可视化缓存，源HTML始终是无旧数据的模板。
    (cache/"current.html").write_text(html,encoding="utf-8")
    return html.encode("utf-8"),files,video_path,result_dir

class ViewerHandler(BaseHTTPRequestHandler):
    """仅提供当前页面、白名单数据文件和本机视频分段读取。"""
    def log_message(self,*args):pass
    def send_content(self,data,mime,head=False):
        self.send_response(200);self.send_header("Content-Type",mime)
        self.send_header("Content-Length",str(len(data)));self.send_header("Cache-Control","no-store");self.end_headers()
        if not head:self.wfile.write(data)
    def send_file(self,path,head=False):
        if not path.is_file():self.send_error(404);return
        self.send_content(path.read_bytes(),mimetypes.guess_type(path.name)[0] or "application/octet-stream",head)
    def video(self,head=False):
        size=_VIDEO_PATH.stat().st_size;start=0;end=size-1;status=200
        header=self.headers.get("Range","")
        if header:
            match=re.fullmatch(r"bytes=(\d*)-(\d*)",header.strip())
            if not match or not any(match.groups()):self.send_error(416);return
            a,b=match.groups()
            if a:start=int(a);end=min(int(b),size-1) if b else size-1
            else:start=max(0,size-int(b))
            if start>=size or start>end:self.send_error(416);return
            status=206
        self.send_response(status);self.send_header("Content-Type","video/mp4")
        self.send_header("Content-Length",str(end-start+1));self.send_header("Accept-Ranges","bytes")
        if status==206:self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head:return
        with _VIDEO_PATH.open("rb") as handle:
            handle.seek(start);remaining=end-start+1
            while remaining:
                block=handle.read(min(1024*1024,remaining))
                if not block:break
                try:self.wfile.write(block)
                except (BrokenPipeError,ConnectionResetError):break
                remaining-=len(block)
    def dispatch(self,head=False):
        path=unquote(urlsplit(self.path).path)
        if path=="/health":self.send_content(b"ok","text/plain",head)
        elif path in {"/",f"/{NAME}_viewer.html"}:self.send_content(_HTML_BYTES,"text/html; charset=utf-8",head)
        elif path=="/video":self.video(head)
        elif path in _FILES:self.send_file(_FILES[path],head)
        elif path.startswith(f"/results/{NAME}/"):
            target=(_RESULTS_DIR/path.removeprefix(f"/results/{NAME}/")).resolve()
            if target.is_relative_to(_RESULTS_DIR):self.send_file(target,head)
            else:self.send_error(404)
        elif path=="/favicon.ico":self.send_response(204);self.end_headers()
        else:self.send_error(404)
    def do_GET(self):self.dispatch(False)
    def do_HEAD(self):self.dispatch(True)

def main():
    """不进入总入口也可启动；--side-root/--cache-dir便于迁移和测试。"""
    global _HTML_BYTES,_FILES,_VIDEO_PATH,_RESULTS_DIR
    parser=argparse.ArgumentParser(description="chenxi侧面 鱼体角度 可视化（只读分析结果）")
    parser.add_argument("--side-root",type=Path,default=SIDE_ROOT)
    parser.add_argument("--html",type=Path,default=None)
    parser.add_argument("--cache-dir",type=Path,default=None)
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=PORT)
    args=parser.parse_args()
    _HTML_BYTES,_FILES,_VIDEO_PATH,_RESULTS_DIR=prepare(args.side_root,args.html,args.cache_dir)
    server=ThreadingHTTPServer((args.host,args.port),ViewerHandler)
    print(f"鱼体角度：http://{args.host}:{args.port}/ （Ctrl+C停止；分析更新后重启）",flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=="__main__":main()
