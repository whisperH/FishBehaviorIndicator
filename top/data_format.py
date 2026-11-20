"""本项目唯一的标准数据格式定义、构造、校验与读写实现。

跟踪模块和行为模块均依赖本文件；本文件不导入tracker、behavior、torch或SAM3，
也不计算头向、中轴线、速度、频率。两个阶段的数据结构不同，但在此统一管理。
业务模块只调用构造/校验函数；CSV列属于各指标的明细报告，定义仍在各指标脚本。

一、分割输入（跟踪模块输出、行为模块输入）
    {"data": [record1, record2, ...]}
每条record对应某帧中的一条鱼，各字段为：
    track_id       非负整数鱼ID，0合法；跨帧同ID必须是同一条鱼，不重编号。
    frame_id       非负整数原始帧号；速度步长按原帧号差计算。
    model_info     模型来源字典，不参与指标公式；默认{}。
    video_info     视频信息字典，字段如下：
        video_id          视频标识，非空字符串或非负整数，默认"unspecified"。
        fps               正帧率；也可由分析函数fps参数或目录元数据提供。
        timestamp_seconds 可选采集秒数，整批全部有或全部无；同帧一致，跨帧递增。
                          有时采用实际时间差，无时采用原帧号差/fps。
        image_path        可选原图绝对路径，只在输出叠加视频时需要。
    view_info      视角，当前只支持"top"，缺省即top，不能混入侧视。
    category_id    非负整数类别，当前鱼类约定1，默认1。
    segmentation   {"size":[height,width], "counts":[整数游程,...]}，必需。
                   COCO按列展开，先从上到下再向右；背景0/前景1长度交替。
                   首像素为前景则首游程0，游程和恰好为height*width。
                   兼容COCO压缩counts字符串，旧字段rle也可读取。
    area           可选注释面积，像素；行为分析以解码掩膜的真实面积为准。
    bbox           可选[左上x,左上y,宽,高]，像素；必须非负且不越界。
    filename       可选字符串，缺省取image_path文件名或由frame_id生成。

一批必须同一视频、视角、帧率、图像尺寸；同帧时间/图片路径一致。
list可乱序，会按frame_id/track_id排序；其他迭代器必须按帧有序。
重复(frame_id,track_id)、损坏RLE、异常时间会报错；不凭空补帧或猜测帧率。
面积/框与掩膜不一致但数值合法时，用掩膜实测值并记录警告。
目录输入保留空鱼帧清单，文件逐行读，每次仅解码一个实例掩膜。
每次分析处理传入的完整批次，不跨调用保存历史；空输入可返回空记录与可用元信息。

二、标准行为输出
    {"data": [indicator1, indicator2, ...]}
每条记录由make_behavior_indicator统一构造：
    track_id/frame_id 鱼ID/结果所属帧，区间归终点帧，整段汇总归最后观测帧。
    behavior_indicator 指标名称，含义如下：
        position_x/position_y：像素质心；heading_tilt：左右水平0、朝上正、朝下负；
        tail_angle：有符号头尾夹角；tail_beat_frequency：连续有效片段摆尾频率；
        speed/velocity：无符号/有符号速度；step_distance/signed_distance：区间距离；
        total_distance/total_signed_distance：各步长有效区间累计距离；
        mean_speed/mean_velocity：对应有效总距离除以对应有效总时间。
    behavior_value 单个数值；不足或无效时为0占位，有效静止的0仍标为ok。
    unit           "pixel"、"pixel/s"、"BL"、"BL/s"、"degree"、"Hz"。
    frame_range    实际涉及的帧ID；区间距离取首末质心，中间帧用于质量检查。
                   汇总列相应有效区间涉及帧，无有效区间时列已有帧。
    frame_step     仅运动指标具有此字段，如1、5；不用于位置、角度、频率。
    status         "ok" / "insufficient_frames" / "invalid"。
    reason         不足/无效的具体原因；正常时为空。统计必须先筛选status=="ok"。
BL记录额外包含：
    reference_body_length_px   此鱼固定参考像素体长，不可用时为JSON null。
    reference_length_sample_count 参与该鱼参考体长估计的有效帧数。
    body_length_valid         体长是否可用，整数0/1。

BL值由每条鱼自己的参考体长换算：本批全部有效中轴长度的中位数，所有步长共用。
无有效体长时BL值为0占位、status=invalid、reason=no_valid_body_length。
同名运动指标同时有像素与BL单位；唯一标识需包含
    track_id + frame_id + behavior_indicator + unit + frame_step
不能只按鱼ID、帧号和指标名去重。返回记录不含NaN或Infinity。

文件说明：standard_data.jsonl.gz每行是一个帧的{"data":[...]}，空鱼帧为[]。
frames.csv和run_info.json保留完整帧清单及采集信息，属于实验结果读取所需元数据。
map_standard_data完成输入校验与小型记录映射，convert_segmentation负责文件转换。

"""
from pathlib import Path
import csv,gzip,json,math,numbers,re
from collections.abc import Iterable
import numpy as np
DEFAULT_SEGMENTATION_DIR = Path(r"E:\Phd\temperature\sam3\top\output\tracking")
MAX_MASK_PIXELS = 64_000_000


def prepare_output_directory(directory, patterns, *, invalidate_combined=False):
    """为覆盖输出准备固定目录，返回其绝对Path；不创建时间戳子目录。

    directory：调用方明确给定的绝对结果路径。
    patterns：本模块生成文件的相对名称/通配模式；只删除匹配的文件。
              不整目录删除，手动放入的其他文件和其他模块结果均保留。
              旧步长、旧鱼ID图和禁用后的视频也在清理范围，避免新旧混用。
    invalidate_combined：行为模块为True。当父目录有main生成的综合结果时，
              撤下behavior_results.json及其run_info.json；它们必须由main统一
              重建，不能在某个模块重算后继续表示全部最新结果。各模块自身的
              CSV、图表、视频仍分别保留。main会在本轮结束时重写综合文件。

    所有待删路径先统一校验再删除：限定在输出目录内，拒绝符号链接/Windows
    junction等重解析点；只移除清理后为空的下级目录，不删除结果根目录。
    """
    output = Path(directory)
    if not output.is_absolute():
        raise ValueError("输出目录必须使用绝对路径。")

    def reject_links(path):
        # lstat不跟随最终链接，Windows属性0x400表示重解析点。
        for part in (path, *path.parents):
            if part.is_symlink() or (part.exists() and
                    getattr(part.lstat(), "st_file_attributes", 0) & 0x400):
                raise ValueError(f"输出清理不允许链接或junction路径：{part}")

    reject_links(output)
    output = output.resolve()
    if output == Path(output.anchor):
        raise ValueError("输出目录不能是磁盘根目录。")
    output.mkdir(parents=True, exist_ok=True)
    targets = set()
    for pattern in patterns:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError("输出文件模式必须限定在结果目录内。")
        for path in output.glob(pattern):
            reject_links(path)
            if not path.resolve().is_relative_to(output) or not path.is_file():
                raise ValueError(f"输出清理目标不是目录内的普通文件：{path}")
            targets.add(path)

    combined = []
    if invalidate_combined:
        info_path = output.parent / "run_info.json"
        if info_path.is_file():
            reject_links(info_path)
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                info = None
            # 仅识别main的综合元信息，不能删除其他用途的同名run_info.json。
            if isinstance(info, dict) and "indicator_identity" in info and "motion_units" in info:
                for name in ("behavior_results.json", "run_info.json"):
                    path = output.parent / name
                    reject_links(path)
                    if path.exists():
                        if not path.is_file() or path.resolve().parent != output.parent:
                            raise ValueError(f"综合输出清理路径异常：{path}")
                        combined.append(path)

    # 上面已验证全部目标；至此才开始删除。先撤下失效的综合结果。
    parents = {parent for path in targets for parent in path.parents
               if parent != output and parent.is_relative_to(output)}
    for path in [*combined, *sorted(targets)]:
        path.unlink()
    for parent in sorted(parents, key=lambda p: len(p.parts), reverse=True):
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    return output


def _integer(value, name, minimum=0):
    """JSON整数和NumPy整数均可，拒绝布尔值、1.0和数字字符串。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} 必须是整数，不能是布尔值、浮点数或字符串。")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}。")
    return value


def _number(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} 必须是有限数字。")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} 必须是{'有限正数' if positive else '有限数字'}。")
    return value


def _decode_compressed_counts(text, pixels):
    """安全解析COCO压缩字符串，不把未经校验的长度交给C扩展。

    COCO按每字符5个数据位保存带符号整数，从第4个run起还要加上前两个run
    的长度。带符号的是差分编码值，恢复后的每个run长度必须非负。
    限制单run编码位数并逐段检查总长度，避免损坏输入造成越界写或巨额分配。
    """
    if isinstance(text, bytes):
        try:
            text = text.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("压缩RLE counts 必须是ASCII字符串。") from exc
    counts, position, total = [], 0, 0
    while position < len(text):
        value, shift = 0, 0
        while True:
            if position >= len(text):
                raise ValueError("压缩RLE字符串在run编码中间截断。")
            c = ord(text[position]) - 48
            position += 1
            if not 0 <= c <= 63:
                raise ValueError("压缩RLE含非法字符。")
            value |= (c & 31) << shift
            shift += 5
            if shift > 60:
                raise ValueError("压缩RLE单个run编码过长。")
            if not c & 32:
                if c & 16:
                    value |= -1 << shift
                break
        if len(counts) > 2:
            value += counts[-2]
        if value < 0:
            raise ValueError("压缩RLE解码出了负run长度。")
        total += value
        if total > pixels:
            raise ValueError("RLE run总长度超过size指定的像素数。")
        counts.append(value)
    if total != pixels:
        raise ValueError(f"RLE run总长度{total}与size像素数{pixels}不一致。")
    return counts


def decode_rle(segmentation):
    """恢复二维bool掩膜，支持标准整数run列表与旧COCO压缩counts。

    segmentation={'size':[height,width], 'counts':...}；按COCO列优先顺序
    （先从上到下，再向右换列）展开，run依次为背景、前景、背景……。
    如果首像素是前景，counts第一个背景run必须是0。所有run必须为非负整数，
    包括零长度run；总和必须恰好等于height*width。size为正整数且不超过
    MAX_MASK_PIXELS。检查完成后才分配掩膜；不使用不安全的原生RLE解码入口。
    """
    if not isinstance(segmentation, dict):
        raise ValueError("segmentation 必须是含size和counts的字典。")
    size = segmentation.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("segmentation.size 必须为 [height, width]。")
    height, width = [_integer(v, "segmentation.size", 1) for v in size]
    pixels = height * width
    if pixels > MAX_MASK_PIXELS:
        raise ValueError(f"掩膜尺寸超过安全上限 {MAX_MASK_PIXELS:,} 像素。")
    counts = segmentation.get("counts")
    if isinstance(counts, (str, bytes)):
        counts = _decode_compressed_counts(counts, pixels)
    elif isinstance(counts, (list, tuple)):
        validated, total = [], 0
        for value in counts:
            run = _integer(value, "segmentation.counts中的run")
            total += run
            if total > pixels:
                raise ValueError("RLE run总长度超过size指定的像素数。")
            validated.append(run)
        if total != pixels:
            raise ValueError(f"RLE run总长度{total}与size像素数{pixels}不一致。")
        counts = validated
    else:
        raise ValueError("segmentation.counts 必须是整数列表或COCO压缩字符串。")
    flat = np.zeros(pixels, dtype=bool)
    position = 0
    for index, run in enumerate(counts):
        if index % 2:
            flat[position:position + run] = True
        position += run
    return flat.reshape((height, width), order="F")


def encode_uncompressed_rle(mask):
    """把二维二值掩膜编码为可直接JSON序列化的COCO整数run列表。

    返回counts列表[背景长度,前景长度,...]，调用者另附size=[height,width]。
    非零值按前景处理；输出采用列优先，首像素为前景时自动补0背景run。
    仅编码，不写文件。与decode_rle互为逆操作。
    """
    array = np.asarray(mask)
    if array.ndim != 2 or min(array.shape) <= 0:
        raise ValueError("encode_uncompressed_rle 需要非空二维掩膜。")
    flat = (array != 0).ravel(order="F")
    changes = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1, flat.size]
    counts = np.diff(changes).astype(np.int64).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return counts


def _validate_identity(row):
    if not isinstance(row, dict):
        raise ValueError("data中的每一条记录必须是字典。")
    if "frame_id" not in row or "track_id" not in row:
        raise ValueError("每条记录必须提供非负整数frame_id和track_id。")
    return _integer(row["frame_id"], "frame_id"), _integer(row["track_id"], "track_id")


def _video_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, numbers.Integral)):
        raise ValueError("video_info.video_id 必须是非空字符串或非负整数。")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("video_info.video_id 不能为空。")
        return value
    return _integer(value, "video_info.video_id")


class _DirectoryRecords:
    """具有meta清单的惰性记录流。仅清单常驻内存，RLE逐行读取。

    同一实例可重新迭代，文件每次独立打开。标准文件优先；没有标准文件时
    适配旧native_masks.jsonl.gz。空鱼帧不产生伪实例，由meta完整保留。
    """
    def __init__(self, directory, use_filename_timestamps):
        if not isinstance(use_filename_timestamps, bool):
            raise ValueError("use_filename_timestamps必须是True或False。")
        self.root = Path(directory)
        if not self.root.is_absolute():
            raise ValueError("分割结果目录必须使用绝对路径。")
        info = json.loads((self.root / "run_info.json").read_text(encoding="utf-8-sig"))
        if info.get("status") != "complete":
            raise ValueError(f"只读取完整分割结果，当前status={info.get('status')}。")
        standard = self.root / "standard_data.jsonl.gz"
        self.standard = standard.is_file()
        self.path = standard if self.standard else self.root / "native_masks.jsonl.gz"
        # 标准记录的时间已经由上游确定；这个布尔开关只控制旧native兼容入口。
        # 新标准按run_info记录的原时间源核验，不能因为调用方传False而换一套时钟。
        declared_source = info.get("standard_time_source") if self.standard else None
        if declared_source is not None and declared_source not in (
                "filename _t seconds, relative to first frame", "frame_index / fps"):
            raise ValueError("run_info.standard_time_source不是支持的标准时间源。")
        parse_filename_time = ((declared_source != "frame_index / fps") if self.standard
                               else use_filename_timestamps)
        fps = _number(info.get("fps"), "run_info.fps", positive=True)
        resolution = info.get("resolution")
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            raise ValueError("run_info.resolution必须是[width,height]。")
        width, height = [_integer(v, "run_info.resolution", 1) for v in resolution]
        with (self.root / "frames.csv").open(encoding="utf-8-sig", newline="") as file:
            self.manifest = list(csv.DictReader(file))
        if len(self.manifest) != int(info["frames"]) or int(info.get("completed_frames", -1)) != len(self.manifest):
            raise ValueError("run_info与frames.csv中的帧数不一致。")
        frame_ids = []
        image_paths = {}
        for row in self.manifest:
            text = row.get("frame", row.get("frame_id", ""))
            if not isinstance(text, str) or not text.isdigit():
                raise ValueError("frames.csv帧号必须是非负整数字符串。")
            frame = int(text)
            if frame_ids and frame <= frame_ids[-1]:
                raise ValueError("frames.csv帧号必须严格递增且不能重复。")
            frame_ids.append(frame)
            row["frame"] = frame
            path = row.get("absolute_path") or row.get("image_path")
            if path:
                if not Path(path).is_absolute():
                    raise ValueError("frames.csv中的图片路径必须为绝对路径。")
                image_paths[frame] = path
            row["filename"] = row.get("filename") or (Path(path).name if path else f"frame_{frame:06d}")
        times, source, used_filename_time = [], "frame_index / fps", False
        if parse_filename_time:
            for row in self.manifest:
                match = re.search(r"_t(-?\d+(?:\.\d+)?)", row["filename"])
                if match is None:
                    times = []; break
                times.append(float(match.group(1)))
            if len(times) == len(self.manifest) and np.isfinite(times).all() and np.all(np.diff(times) > 0):
                times = np.asarray(times) - times[0] if times else []
                source = "filename _t seconds, relative to first frame"
                used_filename_time = bool(self.manifest)
            else:
                times = []
                source = "frame_index / fps (filename timestamps missing or non-increasing)"
        if not len(times):
            times = [(frame - frame_ids[0]) / fps for frame in frame_ids]
        if self.standard and declared_source == "filename _t seconds, relative to first frame" and not used_filename_time and self.manifest:
            raise ValueError("标准元数据声明使用文件名采集时间，但frames.csv文件名时间缺失或不递增。")
        video_id = _video_id(info.get("video_id", Path(info.get("image_directory", self.root)).name))
        if info.get("view_info", "top") != "top":
            raise ValueError("该分割目录不是top顶部视角。")
        self.meta = {"fps": fps, "width": width, "height": height, "frame_count": len(frame_ids),
                     "frame_ids": frame_ids, "image_paths": image_paths,
                     "frame_times": dict(zip(frame_ids, map(float, times))), "time_source": source,
                     "video_id": video_id, "view_info": "top", "segmentation_dir": str(self.root),
                     "use_filename_timestamps": None if self.standard else use_filename_timestamps,
                     "requested_use_filename_timestamps": use_filename_timestamps,
                     "used_filename_timestamps": used_filename_time}
        self.info = info

    def __iter__(self):
        line_count, supplied_time_mode = 0, None
        with gzip.open(self.path, "rt", encoding="utf-8") as file:
            for index, line in enumerate(file):
                if index >= len(self.manifest):
                    raise ValueError("掩膜文件含有超出frames.csv清单的多余帧。")
                data = json.loads(line)
                manifest = self.manifest[index]
                frame, filename = manifest["frame"], manifest["filename"]
                time_s = self.meta["frame_times"][frame]
                path = self.meta["image_paths"].get(frame)
                if self.standard:
                    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
                        raise ValueError("standard_data.jsonl.gz每行必须是{'data':[该帧实例]}。")
                    objects = data["data"]
                else:
                    if int(data["frame"]) != frame or data["filename"] != filename:
                        raise ValueError("旧掩膜文件帧号/文件名与frames.csv不一致。")
                    objects = data["objects"]
                    if not isinstance(objects, list):
                        raise ValueError("旧掩膜objects必须是列表。")
                for obj in objects:
                    if not isinstance(obj, dict):
                        raise ValueError("每个掩膜对象必须是字典。")
                    if self.standard:
                        item = dict(obj)
                        this_frame, _ = _validate_identity(item)
                        if this_frame != frame:
                            raise ValueError("standard_data行中的frame_id与frames.csv对应帧不一致。")
                        video = item.get("video_info", {})
                        if not isinstance(video, dict):
                            raise ValueError("video_info必须是字典。")
                        video = dict(video)
                        has_time = video.get("timestamp_seconds") is not None
                        if supplied_time_mode is None:
                            supplied_time_mode = has_time
                        elif supplied_time_mode != has_time:
                            raise ValueError("标准文件中的timestamp_seconds不能部分有、部分无。")
                        if has_time:
                            value = _number(video["timestamp_seconds"], "timestamp_seconds")
                            if not math.isclose(value, time_s, abs_tol=1e-7, rel_tol=0):
                                raise ValueError("标准文件时间戳与frames.csv按当前时间设置得到的相对秒数不一致。")
                        video.setdefault("video_id", self.meta["video_id"])
                        video.setdefault("fps", self.meta["fps"])
                        if path is not None:
                            video.setdefault("image_path", path)
                    else:
                        item = {"track_id": obj["id"], "frame_id": frame, "category_id": 1,
                                "segmentation": obj["rle"], "bbox": obj.get("bbox_xywh"),
                                "model_info": self.info.get("model_info", {"name": "SAM3"}), "view_info": "top"}
                        if obj.get("area") is not None:
                            item["area"] = obj["area"]
                        video = {"video_id": self.meta["video_id"], "fps": self.meta["fps"]}
                        if path is not None:
                            video["image_path"] = path
                    # 清单统一决定相对时间，完整保留开头/中间无鱼帧的时间基准。
                    if not self.standard or video.get("timestamp_seconds") is None:
                        video["timestamp_seconds"] = time_s
                    item["video_info"] = video
                    item.setdefault("filename", filename)
                    yield item
                line_count += 1
        if line_count != len(self.manifest):
            raise ValueError(f"掩膜文件帧数不完整：{line_count}/{len(self.manifest)}。")


def iter_standard_records(segmentation_dir=DEFAULT_SEGMENTATION_DIR, use_filename_timestamps=True):
    """把完整分割目录转成标准记录流，返回值另带只读使用的.meta清单。

    每帧占一行的standard_data.jsonl.gz优先；旧native_masks逐行适配成同一字段。
    旧native先读取全部frames.csv判定文件名时间是否全部有效递增，否则整段回退；
    与原150帧流程相同。新标准优先保留记录时间，按run_info.standard_time_source
    核验；use_filename_timestamps仅用于旧native，不覆盖新标准采集时钟。
    新标准meta.use_filename_timestamps为None（不适用），另记requested和used。
    标准时间相对首帧；记录时间须与原时间源相符，不悄悄改写时间尺度。
    输出counts可保持旧压缩字符串；如需导出整数列表，可调用decode_rle再encode。
    """
    return _DirectoryRecords(segmentation_dir, use_filename_timestamps)




def _inspect_annotation(row, mask, warnings, frame, identity):
    """验证外部area/bbox的合法性；几何一律使用掩膜实测值。

    不合法数字或越界框是输入错误；合法但与mask不一致是注释偏差，记入meta
    warnings而不覆盖mask实测质心/面积。bbox明确使用COCO[x,y,width,height]。
    """
    height, width = mask.shape
    y, x = np.nonzero(mask)
    if len(x) == 0:
        raise ValueError(f"frame_id={frame}, track_id={identity} 的实例掩膜为空。")
    area = len(x)
    bbox = [float(x.min()), float(y.min()), float(x.max()-x.min()+1), float(y.max()-y.min()+1)]
    if row.get("area") is not None:
        supplied = _number(row["area"], "area")
        if not 0 <= supplied <= height * width:
            raise ValueError("area必须处于0与图像像素总数之间。")
        if not math.isclose(supplied, area, abs_tol=1e-6, rel_tol=0):
            warnings.append(f"frame_id={frame},track_id={identity}: area与掩膜不符，采用掩膜面积{area}。")
    if row.get("bbox") is not None:
        supplied = row["bbox"]
        if not isinstance(supplied, (list, tuple)) or len(supplied) != 4:
            raise ValueError("bbox必须是COCO [x,y,width,height] 四个数字。")
        supplied = [_number(v, "bbox") for v in supplied]
        xx, yy, ww, hh = supplied
        if min(supplied) < 0 or xx + ww > width + 1e-6 or yy + hh > height + 1e-6:
            raise ValueError("bbox必须非负且位于segmentation.size的图像范围内。")
        if not np.allclose(supplied, bbox, atol=1e-6, rtol=0):
            warnings.append(f"frame_id={frame},track_id={identity}: bbox与掩膜不符，采用掩膜外接框{bbox}。")
    return x, y, area, bbox

def map_standard_data(payload, measure, fps=None):
    """集中校验标准输入，逐条解码并交给measure，返回(records,meta)。

    payload字段详见本文件顶部；fps为可选帧率补充，不猜测默认值。
    measure是调用方提供的单掩膜处理函数，参数为
      (mask, frame, identity, time_s, filename, x, y, area)，返回一个小型字典。
    本函数只做格式、时间、ID、RLE、面积/边界框校验和来源字段附加，
    不计算头向、体长或行为指标，不导入任何模型或行为模块。
    list可乱序；其他迭代器必须按帧有序。每次只保留当前掩膜，返回记录
    按frame_id/track_id排序。measure不得把大掩膜留在返回记录中。
    空输入仍返回可用meta；错误输入抛ValueError，不静默修复时间或ID。
    """
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError("输入必须是含data的字典：{'data': [...]}。")
    data = payload["data"]
    if isinstance(data, (str, bytes, dict)) or not isinstance(data, Iterable):
        raise ValueError("data必须是记录列表或按帧顺序提供记录的可迭代对象。")
    directory_meta = getattr(data, "meta", None) if isinstance(data, _DirectoryRecords) else None
    requested_fps = _number(fps, "fps参数", positive=True) if fps is not None else None
    if directory_meta is not None:
        source_fps = directory_meta["fps"]
        if requested_fps is not None and not math.isclose(requested_fps, source_fps, rel_tol=1e-9):
            raise ValueError("fps参数与分割目录元数据fps不一致。")
        requested_fps = source_fps
    selected_fps = requested_fps
    if isinstance(data, list):
        # 只复制记录引用并排序，不额外复制/解码RLE。
        data = sorted(data, key=_validate_identity)
        if selected_fps is None:
            for row in data:
                info = row.get("video_info", {})
                if isinstance(info, dict) and info.get("fps") is not None:
                    selected_fps = _number(info["fps"], "video_info.fps", positive=True)
                    break
    records, warnings, seen_keys = [], [], set()
    frame_ids, frame_times, image_paths = [], {}, {}
    width = height = 0
    video_id = "unspecified"
    timestamp_mode, origin_timestamp, previous_timestamp, previous_frame = None, None, None, None
    first_frame = directory_meta["frame_ids"][0] if directory_meta and directory_meta["frame_ids"] else None
    if directory_meta is not None:
        width, height, video_id = directory_meta["width"], directory_meta["height"], directory_meta["video_id"]
        frame_ids = list(directory_meta["frame_ids"])
        frame_times = dict(directory_meta["frame_times"])
        image_paths = dict(directory_meta["image_paths"])
        origin_timestamp = 0.0  # 目录适配器已提供相对整个视频首帧的秒数，包括空鱼帧。
    for row in data:
        frame, identity = _validate_identity(row)
        if previous_frame is not None and frame < previous_frame:
            raise ValueError("流式data必须按frame_id非降序排列；乱序数据请先转为list。")
        if (frame, identity) in seen_keys:
            raise ValueError(f"重复记录：frame_id={frame}, track_id={identity}。")
        seen_keys.add((frame, identity))
        category = _integer(row.get("category_id", 1), "category_id")
        model_info = row.get("model_info", {})
        info = row.get("video_info", {})
        if not isinstance(model_info, dict) or not isinstance(info, dict):
            raise ValueError("model_info和video_info必须是字典。")
        view = row.get("view_info", "top")
        if view != "top":
            raise ValueError("本项目仅支持view_info='top'，不得混入侧视角或其他视角。")
        this_video = _video_id(info.get("video_id", "unspecified"))
        if records or directory_meta is not None:
            if type(this_video) is not type(video_id) or this_video != video_id:
                raise ValueError("同一批data只能包含同一个video_id。")
        else:
            video_id = this_video
        if info.get("fps") is not None:
            supplied_fps = _number(info["fps"], "video_info.fps", positive=True)
            if selected_fps is None:
                selected_fps = supplied_fps
            elif not math.isclose(selected_fps, supplied_fps, rel_tol=1e-9, abs_tol=0):
                raise ValueError("同一批data的fps必须一致，且与fps参数一致。")
        if selected_fps is None:
            raise ValueError("缺少fps。请调用时传fps=实际帧率，或在video_info中提供fps；程序不会猜测30fps。")
        has_timestamp = info.get("timestamp_seconds") is not None
        if timestamp_mode is None:
            timestamp_mode = has_timestamp
        elif timestamp_mode != has_timestamp:
            raise ValueError("timestamp_seconds必须全部提供或全部省略，不能混用。")
        if first_frame is None:
            first_frame = frame
        if has_timestamp:
            timestamp = _number(info["timestamp_seconds"], "video_info.timestamp_seconds")
            if previous_frame == frame:
                if timestamp != previous_timestamp:
                    raise ValueError("同一frame_id中所有对象的timestamp_seconds必须完全一致。")
            elif previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError("不同frame_id的timestamp_seconds必须严格递增。")
            if origin_timestamp is None:
                origin_timestamp = timestamp
            time_s = float(timestamp - origin_timestamp)
            previous_timestamp = timestamp
        else:
            time_s = float((frame - first_frame) / selected_fps)
        image_path = info.get("image_path")
        if image_path is not None:
            if not isinstance(image_path, str) or not Path(image_path).is_absolute():
                raise ValueError("video_info.image_path如提供，必须是绝对路径字符串。")
            if frame in image_paths and image_paths[frame] != image_path:
                raise ValueError("同一frame_id不能对应不同image_path。")
            image_paths[frame] = image_path
        filename = row.get("filename") or (Path(image_path).name if image_path else f"frame_{frame:06d}")
        if not isinstance(filename, str):
            raise ValueError("filename如提供，必须是字符串。")
        segmentation = row.get("segmentation", row.get("rle"))
        mask = decode_rle(segmentation)
        if width == height == 0:
            height, width = mask.shape
        if mask.shape != (height, width):
            raise ValueError("同一批data的所有segmentation.size必须一致。")
        x, y, area, bbox = _inspect_annotation(row, mask, warnings, frame, identity)
        record = measure(mask, frame, identity, time_s, filename, x, y, area)
        record.update(frame_id=frame, track_id=identity, category_id=category, bbox=bbox,
                      model_info=dict(model_info), video_info=dict(info), view_info="top")
        records.append(record)
        if directory_meta is not None:
            if frame not in frame_times or not math.isclose(frame_times[frame], time_s, abs_tol=1e-7, rel_tol=0):
                raise ValueError("标准记录的帧号/时间与frames.csv清单不一致。")
            # 标准记录有自己的采集时刻时保留其原数值；清单只负责时间源核验及空鱼帧。
            frame_times[frame] = time_s
        elif previous_frame != frame:
            frame_ids.append(frame)
            frame_times[frame] = time_s
        previous_frame = frame
        del mask, x, y  # 不保留完整图像掩膜；仅回调返回的小型测量记录进入结果。
    records.sort(key=lambda item: (item["frame_id"], item["track_id"]))
    source = ("video_info.timestamp_seconds, relative to first frame" if timestamp_mode else
              "(frame_id - first_frame_id) / fps")
    meta = {"fps": selected_fps, "width": width, "height": height, "frame_count": len(frame_ids),
            "frame_ids": frame_ids, "image_paths": image_paths, "frame_times": frame_times,
            "time_source": directory_meta["time_source"] if directory_meta is not None else source,
            "video_id": video_id, "view_info": "top", "warnings": warnings,
            "use_filename_timestamps": directory_meta["use_filename_timestamps"] if directory_meta is not None else None,
            "requested_use_filename_timestamps": directory_meta["requested_use_filename_timestamps"] if directory_meta is not None else None,
            "used_filename_timestamps": directory_meta["used_filename_timestamps"] if directory_meta is not None else None}
    if directory_meta is not None:
        meta["segmentation_dir"] = directory_meta["segmentation_dir"]
    return records, meta

def make_segmentation_frame(objects, frame_id, image_path, time_seconds, video_id, *, fps, checkpoint):
    """把上游提取的一帧对象构造成统一分割输入，返回{"data":[...]}。

    objects为跟踪端提取的小字典列表，每项含id、rle、score、bbox_xywh；
    frame_id为原帧号，image_path为原图绝对路径，time_seconds为相对采集秒数，
    video_id标识本视频，fps为正帧率，checkpoint为用于追溯的权重绝对路径。
    保留上游ID和bbox，area由掩膜实测；RLE统一转成按列展开的整数游程。
    本函数不加载SAM3、不进行跟踪，仅负责输出字段和掩膜格式的构造。
    """
    frame_id = _integer(frame_id, "frame_id")
    fps = _number(fps, "fps", positive=True)
    time_seconds = _number(time_seconds, "timestamp_seconds")
    video_id = _video_id(video_id)
    if not Path(image_path).is_absolute() or not Path(checkpoint).is_absolute():
        raise ValueError("原图和权重路径必须为绝对路径。")
    data = []
    for obj in objects:
        mask = decode_rle(obj["rle"])
        data.append({
            "track_id": obj["id"],
            "model_info": {"name": "SAM3", "checkpoint": str(checkpoint), "score": obj["score"]},
            "video_info": {"video_id": video_id, "fps": fps,
                           "timestamp_seconds": float(time_seconds), "image_path": str(image_path)},
            "view_info": "top", "category_id": 1, "frame_id": int(frame_id),
            "segmentation": {"size": obj["rle"]["size"], "counts": encode_uncompressed_rle(mask)},
            "area": float(mask.sum()), "bbox": obj["bbox_xywh"],
        })
    return {"data": data}

def make_behavior_indicator(track_id, frame_id, name, value, unit, frames,
               status="ok", reason="", frame_step=None, *, reference=None):
    """构造唯一的标准指标记录；参数含义见本文件顶部的输出协议。

    value不可测时允许传NaN，统一转换为0占位+非ok状态；有效0保持ok。
    frame_step只用于运动指标。reference只用于BL，传入含参考体长、样本数
    和有效性字段的测量记录；未提供时不附加BL字段。这里不执行指标公式。
    """
    number = float(value)
    if not math.isfinite(number):
        number = 0.0
        if status == "ok":
            status, reason = "invalid", reason or "non_finite_result"
    row = {
        "track_id": int(track_id),
        "frame_id": int(frame_id),
        "behavior_indicator": name,
        "behavior_value": number,
        "unit": unit,
        "frame_range": [int(frame) for frame in frames],
        "status": status,
        "reason": reason,
    }
    if frame_step is not None:
        row["frame_step"] = int(frame_step)
    if reference is not None:
        row.update(reference_body_length_px=(reference["reference_body_length_px"]
                   if reference["body_length_valid"] else None),
                   reference_length_sample_count=reference["reference_length_sample_count"],
                   body_length_valid=reference["body_length_valid"])
    return row

def convert_segmentation(segmentation_dir, output_file):
    """旧/新分割目录转标准JSONL(.jsonl或.jsonl.gz)，每行是单帧的{"data":[...]}。

    这是文件传输格式，逐行读取后将各行data合并/顺序迭代即可调用analyze_behavior。
    保留各条鱼独立掩膜，重叠区域也分别保留。输出路径必须绝对且尚不存在。
    """
    destination = Path(output_file)
    if not destination.is_absolute():
        raise ValueError("output_file 必须为绝对路径。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if destination.suffix == ".gz" else open
    stream = iter_standard_records(Path(segmentation_dir))
    # 使用完整帧清单，空鱼帧也输出{"data":[]}，使文件行数与frames.csv一致。
    # 用一个前瞻实例逐帧写入，内存只保存当前帧，转换同样经过统一RLE校验。
    records = iter(stream)
    current = next(records, None)
    with opener(destination, "xt", encoding="utf-8") as sink:
        for frame in stream.meta["frame_ids"]:
            batch = []
            while current is not None and current["frame_id"] == frame:
                mask = decode_rle(current["segmentation"])
                current["segmentation"] = {"size": list(mask.shape),
                                           "counts": encode_uncompressed_rle(mask)}
                current["area"] = float(mask.sum())
                batch.append(current)
                current = next(records, None)
            sink.write(json.dumps({"data": batch}, ensure_ascii=False, allow_nan=False) + "\n")
        if current is not None:
            raise ValueError("标准转换遇到清单之外的帧。")
    return destination
