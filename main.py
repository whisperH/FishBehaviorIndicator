"""FishBehavior 统一入口：只调度任务，不计算行为指标。

顶部负责人 YHC；侧面负责人 chenxi。各子项目不相互导入。
直接运行后依次选择视角和指标；缺少可用基础表时，询问是否先准备数据。
子脚本使用当前 Python 解释器在独立进程中运行，避免两份 sam3/common 相互覆盖。
所有相对名称最终转换成 PROJECT_ROOT 下的绝对路径；原项目不参与运行。

自动调用示例：
    python main.py --view top --metric tail --prepare no
    python main.py --view side --metric speed --prepare yes
    python main.py --view side --metric gill --check-only
--prepare yes 只在数据缺失/失效时准备，不强制重复分割；no 则返回错误。
--check-only 只验证所需基础数据，不加载模型、不写分析结果。
"""
from pathlib import Path
import argparse
import ast
import csv
import os
import re
import subprocess
import sys

# 维护总入口时通常只需扩充下面的指标表。子项目算法/阈值仍由各自负责人维护。
PROJECT_ROOT = Path(r"E:\Phd\FishBehavior")
PYTHON = Path(sys.executable).resolve()  # 推荐使用 D:\Anaconda\envs\sam3\python.exe 启动本入口。
TOP = PROJECT_ROOT / "Top"
SIDE = PROJECT_ROOT / "Side"
TOP_FRAME_FIELDS = set("schema_version,run_id,record_type,video_id,frame_id,time_s,fps,time_source,image_path,image_width,image_height,track_id,category_id,score,area_px,bbox_x,bbox_y,bbox_width,bbox_height,mask_height,mask_width,mask_rle".split(","))
TOP_RESULT_FIELDS = set("schema_version,run_id,record_type,video_id,frame_id,time_s,fps,time_source,image_path,image_width,image_height,track_id,category_id,score,cx,cy,area_px,bbox_x,bbox_y,bbox_width,bbox_height,centerline,head_dx,head_dy,tail_dx,tail_dy,heading_deg,tail_angle_deg,body_length_px,reference_body_length_px,reference_length_sample_count,position_valid,geometry_valid,head_valid,head_confidence,quality_reason".split(","))
SIDE_META_FIELDS = set("schema_version,run_id,record_type,video_path,video_size_bytes,video_mtime_ns,fps,width,height,frame_count,frame_id,time_s,track_id,source_kind".split(","))
SIDE_FRAME_FIELDS = SIDE_META_FIELDS | set("score,box_iou,bbox_x,bbox_y,bbox_w,bbox_h,mask_rle,polygons,source_geometry,annotation_path".split(","))
SIDE_BODY_FIELDS = SIDE_META_FIELDS | set("score,box_iou,bbox_x,bbox_y,bbox_w,bbox_h,center_x,center_y,angle_center_x,angle_center_y,area,angle_horizontal_deg,orientation_deg,directed_angle_deg,elongation,head_x,head_y,tail_x,tail_y,head_width,tail_width,direction_source,head_match_score,head_match_overlap,position_valid,geometry_valid,quality_reason,annotation_path".split(","))
SIDE_GILL_FIELDS = SIDE_META_FIELDS | set("valid_tracking,confidence,x1,y1,x2,y2,gray_mean,gray_variance,quality_reason,annotation_path".split(","))
TOP_METRICS = {
    "speed": ("速度、平均速度及加速度", "speed.py"),
    "distance": ("距离", "distance.py"),
    "tail": ("摆尾频率及摆尾幅度", "tail.py"),
    "towards": ("游动朝向", "towards.py"),
    "area": ("停留区域", "area.py"),
    "resting_time": ("静止时间", "resting_time.py"),
    "trajectory_entanglement": ("轨迹纠缠度", "trajectory_entanglement.py"),
    "aggregation": ("聚集度", "aggregation.py"),
}
SIDE_METRICS = {
    "gill": ("鳃动频率", "gill_frequency.py"),
    "speed": ("游泳速度", "speed_analysis.py"),
    "water_layer": ("水层分布", "water_layer_analysis.py"),
    "angle": ("鱼体与水平面角度", "angle_analysis.py"),
}


def _same_path(left, right):
    """Windows 路径大小写不同不代表不同视频；不接受空路径。"""
    return bool(left) and os.path.normcase(str(Path(left).resolve())) == os.path.normcase(str(Path(right).resolve()))


def _check_csv(path, required, *, source=None, image_dir=None, expected_images=None, upstream=()):
    """检查是否可复用基础表，不把“文件存在”直接等同于“数据可用”。

    验证表头、非空数据、重复表头/损坏行、来源路径及原图存在性，并检查基础表
    是否早于输入文件。具体数值有效性、缺帧与质量规则仍由指标脚本严格校验。
    全部鱼都未检出时可以保留 empty_frame，不伪造观察值，也不因零指标重跑模型。
    返回 (是否可用, 中文原因)，没有副作用。
    """
    if not path.is_file():
        return False, f"缺少 {path}"
    try:
        count = 0
        run_ids, sources = set(), set()
        observed_images = set()
        image_indexes = {str(p.resolve()).casefold(): i for i, p in enumerate(expected_images or [])}
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames or []
            missing = set(required) - set(headers)
            if missing:
                return False, f"{path.name} 缺少字段：{', '.join(sorted(missing))}"
            if len(headers) != len(set(headers)):
                return False, f"{path.name} 存在重复列名"
            for row in reader:
                count += 1
                if None in row or any(value is None for value in row.values()):
                    return False, f"{path.name} 第 {count + 1} 行列数不正确"
                if row.get("frame_id") == "frame_id":
                    return False, f"{path.name} 中间包含重复表头"
                if row.get("schema_version") != "1" or not row.get("run_id", "").strip():
                    return False, f"{path.name} 的格式版本不是1，或缺少本批次run_id"
                if row.get("run_id"):
                    run_ids.add(row["run_id"])
                if image_dir is not None:
                    if not _same_path(row.get("video_id"), image_dir):
                        return False, "顶部基础表不属于当前输入图片目录"
                    image = Path(row.get("image_path", ""))
                    if not image.is_absolute() or image.parent.resolve() != image_dir.resolve() or not image.is_file():
                        return False, f"基础表引用的原图不存在或位置不符：{image}"
                    image_key = str(image.resolve()).casefold()
                    observed_images.add(image_key)
                    if expected_images is not None and (image_key not in image_indexes or int(row["frame_id"]) != image_indexes[image_key]):
                        return False, "图片集合或自然排序已改变，需要重新生成顶部基础数据"
                if source is not None:
                    recorded = row.get("video_path") or row.get("source_video") or row.get("video_id")
                    if not _same_path(recorded, source):
                        return False, "侧面基础表不属于当前输入视频"
                    if not source.is_file():
                        return False, f"侧面输入视频不存在：{source}"
                    stat = source.stat()
                    if int(row.get("video_size_bytes", "-1")) != stat.st_size or int(row.get("video_mtime_ns", "-1")) != stat.st_mtime_ns:
                        return False, "输入视频大小或修改时间已改变，需要重新准备侧面基础数据"
                    sources.add(recorded)
        if not count:
            return False, f"{path.name} 只有表头，没有记录"
        if len(run_ids) > 1 or len(sources) > 1:
            return False, f"{path.name} 混合了多批数据"
        if expected_images is not None and observed_images != set(image_indexes):
            return False, "基础表未完整覆盖当前选定图片序列，需要重新准备数据"
        for dependency in upstream:
            if not dependency.is_file():
                return False, f"基础结果的输入文件不存在：{dependency}"
            # 1秒容差避免不同文件系统时间精度造成误判。
            if dependency.stat().st_mtime > path.stat().st_mtime + 1.0:
                return False, f"{dependency.name} 已更新，需要重新生成 {path.name}"
        return True, f"基础数据可用：{path}"
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        return False, f"不能读取 {path}：{exc}"


def check_data(task, *, frames_only=False):
    """task 为 top、body 或 gill；body/gill 是侧面两条独立输入链。"""
    if task == "top":
        folder = TOP / "Tracker/Output/Information"
        images = TOP / "Tracker/Input/Images"
        upstream = [p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}] if images.is_dir() else []
        upstream.sort(key=lambda p: tuple((1, int(t)) if t.isdigit() else (0, t.casefold()) for t in re.split(r"(\d+)", p.name)))
        # 不导入会加载模型的Tracker，仅读取MAX_FRAMES字面量，使试跑帧数与入口检查一致。
        try:
            module = ast.parse((TOP / "Tracker/sam3.py").read_text("utf-8-sig"))
            maximum = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MAX_FRAMES" for t in n.targets))
            if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 0:
                raise ValueError("MAX_FRAMES需要非负整数")
            if maximum:
                upstream = upstream[:maximum]
        except (OSError, SyntaxError, ValueError, StopIteration) as exc:
            return False, f"不能读取顶部Tracker的MAX_FRAMES配置：{exc}"
        if not upstream:
            return False, f"顶部输入目录没有图片：{images}"
        if frames_only:
            return _check_csv(folder / "frames.csv", TOP_FRAME_FIELDS, image_dir=images, expected_images=upstream, upstream=upstream)
        frames = folder / "frames.csv"
        status = _check_csv(folder / "result.csv", TOP_RESULT_FIELDS, image_dir=images, expected_images=upstream, upstream=upstream + ([frames] if frames.is_file() else []))
        if status[0] and frames.is_file():
            try:
                with frames.open(encoding="utf-8-sig", newline="") as f, (folder / "result.csv").open(encoding="utf-8-sig", newline="") as g:
                    a, b = next(csv.DictReader(f)), next(csv.DictReader(g))
                for key in ("run_id", "video_id", "fps", "image_width", "image_height", "time_source"):
                    if a.get(key) != b.get(key):
                        return False, f"frames.csv与result.csv批次或元信息不一致：{key}"
            except (OSError, StopIteration, csv.Error) as exc:
                return False, f"不能核对顶部基础表批次：{exc}"
        return status
    if task not in {"body", "gill"}:
        raise ValueError(f"未知基础数据类型：{task}")
    source = SIDE / "Tracker/Input" / task / ("angle.mp4" if task == "body" else "video.mp4")
    gt = SIDE / "Tracker/Input" / task / "gt/gt.txt"
    folder = SIDE / "Tracker/Output" / task
    upstream = [source, gt]
    if frames_only:
        return _check_csv(folder / "frames.csv", SIDE_FRAME_FIELDS, source=source, upstream=upstream)
    if task == "body":
        required = SIDE_BODY_FIELDS
        frames = folder / "frames.csv"
        if frames.is_file():
            upstream.append(frames)
        heads = SIDE / "Tracker/Input/body/head_mask_shapes.json"
        if heads.is_file():
            upstream.append(heads)
    else:
        required = SIDE_GILL_FIELDS
    status = _check_csv(folder / "result.csv", required, source=source, upstream=upstream)
    if task == "body" and status[0] and (folder / "frames.csv").is_file():
        try:
            with (folder / "frames.csv").open(encoding="utf-8-sig", newline="") as f, (folder / "result.csv").open(encoding="utf-8-sig", newline="") as g:
                a, b = next(csv.DictReader(f)), next(csv.DictReader(g))
            for key in ("run_id", "video_path", "video_size_bytes", "video_mtime_ns", "fps", "width", "height", "frame_count"):
                if a.get(key) != b.get(key):
                    return False, f"侧面frames.csv与result.csv批次或元信息不一致：{key}"
        except (OSError, StopIteration, csv.Error) as exc:
            return False, f"不能核对侧面基础表批次：{exc}"
    return status


def run_script(script, *arguments):
    """启动且等待唯一指定脚本；成功返回 True，错误不会继续触发下游任务。"""
    if not script.is_file():
        print(f"找不到脚本：{script}")
        return False
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    print(f"\n正在运行：{script}\n", flush=True)
    try:
        completed = subprocess.run([str(PYTHON), "-B", "-u", "-X", "utf8", str(script), *map(str, arguments)], cwd=str(script.parent), env=env, check=False)
    except (OSError, KeyboardInterrupt) as exc:
        print(f"任务未完成：{exc}")
        return False
    if completed.returncode != 0:
        print(f"任务失败（退出码 {completed.returncode}），已停止后续步骤。请查看上方错误。")
        return False
    return True


def _confirm_preparation(task, mode):
    if mode == "yes":
        return True
    if mode == "no":
        print("本次不准备基础数据，未运行指标分析。")
        return False
    label = {"top": "顶部的分割与基础几何测量", "body": "侧面的鱼体分割与基础测量", "gill": "侧面的鳃部灰度提取（不运行整鱼SAM3）"}[task]
    print(f"需要先完成{label}。")
    while True:
        try:
            answer = input("是否现在执行？[1=是，2=返回，默认返回]：").strip().lower()
        except EOFError:
            return False
        if answer in {"1", "y", "yes", "是"}:
            return True
        if answer in {"", "2", "n", "no", "否"}:
            return False
        print("请输入 1 或 2。")


def ensure_data(task, prepare="ask"):
    ready, reason = check_data(task)
    print(reason)
    if ready:
        return True
    if not _confirm_preparation(task, prepare):
        return False
    if task == "top":
        frames_ready, _ = check_data(task, frames_only=True)
        if not frames_ready and not run_script(TOP / "Tracker/sam3.py"):
            return False
        if frames_ready:
            print("已有可用分割表，仅运行几何测量。")
        success = run_script(TOP / "Tracker/geometry.py")
    elif task == "body":
        frames_ready, _ = check_data(task, frames_only=True)
        if not frames_ready and not run_script(SIDE / "Tracker/sam3.py"):
            return False
        if frames_ready:
            print("已有可用分割表，仅运行鱼体基础测量。")
        success = run_script(SIDE / "Tracker/geometry.py", "--task", "body")
    else:
        success = run_script(SIDE / "Tracker/geometry.py", "--task", "gill")
    if not success:
        return False
    ready, reason = check_data(task)
    print(reason)
    if not ready:
        print("准备任务没有生成可用基础数据，未运行指标分析。")
    return ready


def run_selected(view, metric, prepare="ask", check_only=False):
    """主入口的可调用接口。调用示例：run_selected('top','tail',prepare='no')。"""
    registry = TOP_METRICS if view == "top" else SIDE_METRICS if view == "side" else None
    if registry is None or metric not in registry:
        raise ValueError(f"不支持的视角或指标：{view}/{metric}")
    task = "top" if view == "top" else "gill" if metric == "gill" else "body"
    if check_only:
        ready, reason = check_data(task)
        print(reason)
        return ready
    if not ensure_data(task, prepare):
        return False
    label, filename = registry[metric]
    project = TOP if view == "top" else SIDE
    if run_script(project / "Behavior" / filename):
        print(f"\n{label}分析完成。输出位于：{project / 'Behavior/Output'}")
        return True
    return False


def _choose(title, options):
    print(f"\n{title}")
    for index, (_, label) in enumerate(options, 1):
        print(f"{index}. {label}")
    print("0. 返回/退出")
    while True:
        try:
            answer = input("请输入编号：").strip()
        except EOFError:
            return None
        if answer == "0":
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
        print("请输入菜单中的有效编号。")


def main():
    parser = argparse.ArgumentParser(description="FishBehavior：顶部YHC / 侧面chenxi")
    parser.add_argument("--view", choices=["top", "side"])
    parser.add_argument("--metric", help="指标英文标识；无参数运行可查看交互菜单")
    parser.add_argument("--prepare", choices=["ask", "yes", "no"], default="ask")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if bool(args.view) != bool(args.metric):
        parser.error("--view 和 --metric 必须一起提供")
    print(f"FishBehavior | 顶部：YHC | 侧面：chenxi\n当前环境：{PYTHON}")
    if args.view:
        try:
            return 0 if run_selected(args.view, args.metric, args.prepare, args.check_only) else 1
        except ValueError as exc:
            parser.error(str(exc))
    if args.check_only:
        parser.error("--check-only 需要同时指定 --view 和 --metric")
    try:
        while True:
            view = _choose("请选择视角", [("top", "顶部视角（YHC）"), ("side", "侧面视角（chenxi）")])
            if view is None:
                return 0
            # 顶部所有指标共用一张基础表，可在指标菜单前检查。
            if view == "top" and not ensure_data("top", args.prepare):
                continue
            registry = TOP_METRICS if view == "top" else SIDE_METRICS
            metric = _choose("请选择分析指标", [(key, value[0]) for key, value in registry.items()])
            if metric is not None:
                run_selected(view, metric, args.prepare)
    except KeyboardInterrupt:
        print("\n已退出。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
