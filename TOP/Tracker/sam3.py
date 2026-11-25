# 顶部视角维护负责人：YHC。此脚本仅属于 FishBehavior/Top，不依赖 Side 子项目。
"""SAM3 图片序列分割与连续跟踪（独立运行，第 1 步）。

直接运行：D:\\Anaconda\\envs\\sam3\\python.exe 本脚本的绝对路径。
输出：Tracker/Output/tracking.mp4、Information/frames.csv。
frames.csv 每行是某一帧的一条鱼，掩膜以 COCO 压缩 RLE 文本保存，重叠鱼各自独立。
没有鱼的帧保留 empty_frame 行。它不是行为指标结果；下一步单独运行 geometry.py。
默认仅生成上述两个文件，不导出大批掩膜 PNG，不调用 geometry 或任何行为脚本。
CSV/视频在整段成功且视频核验通过后发布；失败不留下可误认为完成的 frames.csv。
"""
from pathlib import Path
from dataclasses import dataclass, replace
from threading import Lock
from collections import OrderedDict
from collections.abc import MutableMapping
from uuid import uuid4
import csv
import math
import os
import random
import re
import sys
import time

# ====================== 日常修改区（绝对路径） ======================
IMAGE_DIR = Path(r"E:\Phd\FishBehavior\Top\Tracker\Input\Images")
CHECKPOINT = Path(r"E:\Phd\FishBehavior\Top\Model\sam3.pt")
BPE_PATH = Path(r"E:\Phd\FishBehavior\Top\Model\bpe_simple_vocab_16e6.txt.gz")
MODEL_CODE_DIR = Path(r"E:\Phd\FishBehavior\Top\Model\vendor")
OUTPUT_DIR = Path(r"E:\Phd\FishBehavior\Top\Tracker\Output")
PROMPT = "fish"                 # 首帧文本提示，之后在同一会话连续传播 ID。
FPS = 30.0                      # 标称帧率/视频播放速度；实际测量时间优先采用文件名 _t 秒数。
IMAGE_SIZE = 1008                # 保留已验证的模型尺度，不通过降分辨率改变分割效果。
SCORE_THRESHOLD = 0.25           # 仅过滤输出对象，不改模型内部跟踪阈值。
MAX_FRAMES = 0                  # 0=全部；试跑可改成正整数。
GPU_INDEX = 0
CPU_THREADS = 8
SEED = 0
FRAME_CACHE_SIZE = 8            # 图像预处理缓存，不缩短模型连续跟踪历史。
PROMPT_RESIDENT_FRAMES = 32      # 较早提示掩膜无损转为 CPU RLE，降低显存占用。
MASK_ALPHA = 0.35
# ===================================================================

# 脚本名为 sam3.py。优先把模型 vendor 放在模块搜索路径首位，防止模型内部
# “import sam3” 错把这个入口脚本当成模型包；保留模型自带 __init__.py。
if MODEL_CODE_DIR.is_dir():
    sys.path.insert(0, str(MODEL_CODE_DIR))
if os.name == "nt":
    import ctypes
    for dll_name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        dll_path = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / dll_name
        if dll_path.is_file():
            try:
                ctypes.WinDLL(str(dll_path))
            except OSError:
                pass

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco

# 格式规范：首行为英文列标题，UTF-8 BOM，ID=0 合法，坐标 x 向右、y 向下。
# schema_version 协议版本；run_id 本次运行标识；record_type=observation/empty_frame；
# video_id 原图目录标识；frame_id 自然排序后从 0 开始；time_s 相对第一帧的测量秒数；
# fps 标称帧率；time_source 标明 _t 或 frame_index/fps；image_path 原图绝对路径；
# image_width/height 原图像素尺寸；track_id 原生 SAM3 ID；category_id=1 表示鱼；
# score 模型对象分数，不是每个像素正确概率；area_px 原始掩膜前景像素个数；
# bbox_x/y/width/height 是 COCO 外接框；mask_height/width 是完整掩膜尺寸；
# mask_rle 是 COCO compressed counts 的 ASCII 文本（按列展开），不是文件路径。
# 恢复：coco.decode({"size":[mask_height,mask_width],"counts":mask_rle.encode("ascii")})。
# 空鱼帧的身份/掩膜字段为空，但帧时间、图像路径及图像尺寸完整保留。
FRAME_FIELDS = "schema_version,run_id,record_type,video_id,frame_id,time_s,fps,time_source,image_path,image_width,image_height,track_id,category_id,score,area_px,bbox_x,bbox_y,bbox_width,bbox_height,mask_height,mask_width,mask_rle".split(",")

def initialize_runtime(config):
    """初始化本地单GPU模型、随机种子与数值精度，使用随项目附带的模型代码。"""
    # 使用本项目附带的模型代码，不需要安装整个服务器项目。
    shadow = sys.modules.get("sam3")
    if shadow is not None and Path(getattr(shadow, "__file__", "")).resolve() == Path(__file__).resolve():
        raise RuntimeError("函数调用请使用 top_tracker 等独立模块名导入此脚本，不能以 sam3 名称导入；直接运行脚本不受影响。")
    sys.path.insert(0, str(config.MODEL_CODE_DIR))
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    torch.set_num_threads(config.CPU_THREADS)
    cv2.setNumThreads(config.CPU_THREADS)
    if not torch.cuda.is_available():
        raise RuntimeError("当前 Python 没有可用的 CUDA。请使用 D:\\Anaconda\\envs\\sam3\\python.exe。")
    torch.cuda.set_device(config.GPU_INDEX)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("此版本保留原 bfloat16 推理精度，需要支持 bfloat16 的 NVIDIA 显卡。")
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    from sam3.model import io_utils
    import sam3.model.sam3_video_inference as video_module
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    return io_utils, video_module, Sam3VideoPredictor

def collect_images(config):
    """自然排序图片名并逐文件校验格式/尺寸，返回绝对路径清单与(width,height)。"""
    if not config.IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"找不到图片文件夹：{config.IMAGE_DIR}")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    def natural_key(path):
        return tuple((1, int(t)) if t.isdigit() else (0, t.casefold())
                     for t in re.split(r"(\d+)", path.name)), path.name
    paths = sorted((p for p in config.IMAGE_DIR.iterdir() if p.is_file()
                    and p.suffix.lower() in extensions), key=natural_key)
    if config.MAX_FRAMES:
        paths = paths[:config.MAX_FRAMES]
    if not paths:
        raise ValueError(f"文件夹里没有图片：{config.IMAGE_DIR}")
    size = None
    for path in paths:
        with Image.open(path) as im:
            if size is None:
                size = im.size
            if im.size != size:
                raise ValueError(f"同一段视频的图片尺寸必须一致：{path}")
            im.verify()
    return paths, size

class BoundedFrames:
    """原始预处理不变；只缓存最近使用的几帧，不把整段视频放入显存。"""
    def __init__(self, paths, size, mean, std, cpu, loader, cache_size):
        self.cache_size = cache_size
        self.paths, self.size, self.cpu, self.loader = paths, size, cpu, loader
        self.cache = OrderedDict()
        self.mean = torch.tensor(mean, dtype=torch.float16)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float16)[:, None, None]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        index = int(index)
        if index not in self.cache:
            image, _, _ = self.loader(str(self.paths[index]), self.size)
            image = image.to(dtype=torch.float16)
            image -= self.mean
            image /= self.std
            self.cache[index] = image
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index] if self.cpu else self.cache[index].cuda()

    def to(self, *args, **kwargs):
        return self

def pack_mask(tensor):
    """将单张布尔张量转为CPU COCO压缩RLE，同时保留原shape和设备供精确还原。"""
    if tensor.dtype != torch.bool or math.prod(tensor.shape[:-2]) != 1:
        raise ValueError("缓存接口发生变化，不能无损压缩该掩膜。")
    array = tensor.detach().cpu().numpy().reshape(tensor.shape[-2:])
    rle = coco.encode(np.asfortranarray(array.astype(np.uint8)))
    return rle, tuple(tensor.shape), tensor.device

def unpack_mask(packed):
    """恢复原shape、布尔类型与设备上的掩膜，不对掩膜重采样或近似量化。"""
    rle, shape, device = packed
    return torch.from_numpy(coco.decode(rle).astype(bool).reshape(shape)).to(device)

class MaskCache(MutableMapping):
    """将已输出的布尔掩膜无损压缩；需要时恢复，保留完整连续跟踪历史。"""
    def __init__(self, initial, nested=False):
        self.data = dict(initial)
        self.pending = set(initial)
        self.nested = nested

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __delitem__(self, key):
        del self.data[key]
        self.pending.discard(key)

    def __setitem__(self, key, value):
        self.data[key] = value
        self.pending.add(key)

    def __getitem__(self, key):
        value = self.data[key]
        if isinstance(value, tuple):
            return ({k: unpack_mask(v) for k, v in value[0].items()}
                    if self.nested else unpack_mask(value))
        return value

    def offload(self, key):
        """只压缩尚未处理的缓存项；nested表示一个帧键下面还有多个鱼ID。"""
        if key not in self.pending:
            return
        value = self.data[key]
        self.data[key] = ({k: pack_mask(v) for k, v in value.items()},) if self.nested else pack_mask(value)
        self.pending.discard(key)

    def offload_before(self, threshold):
        """将指定帧号之前的待处理掩膜转到压缩CPU缓存，控制驻留显存。"""
        for key in tuple(self.pending):
            if key < threshold:
                self.offload(key)

def extract_objects(config, values, width, height):
    """保留模型原生ID及独立掩膜，输出阈值只过滤结果，不改变内部跟踪阈值。

    score来自模型对象输出分数，不能解释为每个像素的正确概率。
    bbox_xywh与RLE均使用原图尺寸，重叠鱼体各自保留独立掩膜。
    """
    ids, probs, masks = values["out_obj_ids"], values["out_probs"], values["out_binary_masks"]
    if not (len(ids) == len(probs) == len(masks)):
        raise ValueError("SAM3 输出的 ID、分数与掩膜数量不一致。")
    objects = []
    for identity, probability, mask in zip(ids, probs, masks):
        identity, probability = int(identity), float(probability)
        if identity < 0 or not math.isfinite(probability):
            raise ValueError("SAM3 输出包含负 ID 或非有限对象分数。")
        if probability < config.SCORE_THRESHOLD:
            continue
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask, dtype=np.uint8).squeeze()
        if mask.shape != (height, width):
            raise ValueError(f"掩膜尺寸 {mask.shape} 与原图 {(height, width)} 不一致。")
        if not mask.any():
            continue
        rle = coco.encode(np.asfortranarray(mask))
        objects.append({"id": identity, "score": probability,
                        "bbox_xywh": coco.toBbox(rle).tolist(),
                        "rle": {"size": rle["size"], "counts": rle["counts"].decode("ascii")}})
    return objects

def standard_times(config, paths):
    """标准接口使用采集时间：全部文件名_t有效且递增时减去首帧时间。

    若任一图片没有有效_t，整段统一回退到frame/FPS，避免混合两种时间轴。
    frames.csv 的 time_s 保存实际测量时间，视频只按 FPS 等间隔播放。
    """
    values = []
    for path in paths:
        match = re.search(r"_t(-?\d+(?:\.\d+)?)", path.name)
        if match is None:
            return [i / config.FPS for i in range(len(paths))], "frame_index / fps"
        values.append(float(match.group(1)))
    if np.isfinite(values).all() and np.all(np.diff(values) > 0):
        return [value - values[0] for value in values], "filename _t seconds, relative to first frame"
    return [i / config.FPS for i in range(len(paths))], "frame_index / fps"

def validate_settings(config):
    """在创建输出前检查绝对路径、模型文件和必要数值参数。"""
    for name, path in [("IMAGE_DIR", config.IMAGE_DIR), ("CHECKPOINT", config.CHECKPOINT),
                       ("BPE_PATH", config.BPE_PATH), ("MODEL_CODE_DIR", config.MODEL_CODE_DIR), ("OUTPUT_DIR", config.OUTPUT_DIR)]:
        if not path.is_absolute():
            raise ValueError(f"{name} 必须使用绝对路径。")
    if not math.isfinite(config.FPS) or config.FPS <= 0 or not 0 <= config.SCORE_THRESHOLD <= 1:
        raise ValueError("FPS 必须为正数，SCORE_THRESHOLD 必须在 0~1 之间。")
    if config.MAX_FRAMES < 0 or config.FRAME_CACHE_SIZE < 1 or config.CPU_THREADS < 1 or config.PROMPT_RESIDENT_FRAMES < 0:
        raise ValueError("帧数、缓存数量或线程数设置无效。")
    if config.IMAGE_SIZE != 1008:
        raise ValueError("此版本固定使用原始 1008 分辨率以保留参考效果，请不要降低 IMAGE_SIZE。")
    if not config.PROMPT.strip() or not 0 <= config.MASK_ALPHA <= 1:
        raise ValueError("提示词不能为空，MASK_ALPHA 必须在 0~1 之间。")
    if not config.CHECKPOINT.is_file():
        raise FileNotFoundError(f"请先把完整的 SAM3 权重放到：\n{config.CHECKPOINT}\n"
                                "也可以修改脚本顶部 CHECKPOINT，指向已有 sam3.pt 的绝对路径。")
    if not config.BPE_PATH.is_file() or not (config.MODEL_CODE_DIR / "sam3" / "model_builder.py").is_file():
        raise FileNotFoundError("模型词表或 vendor 文件夹缺失，请保持项目完整。")


def render_frame(config, path, objects, frame_index):
    """掩膜、外轮廓及稳定 ID 颜色叠加；不生成丢失重叠信息的合并标签掩膜。"""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    layer = image.copy()
    palette = [(255, 150, 20), (220, 210, 20), (60, 70, 240),
               (20, 215, 245), (220, 70, 200), (60, 205, 100)]
    outlines, labels = [], []
    for obj in objects:
        identity = obj["id"]
        mask = coco.decode({"size": obj["rle"]["size"],
                            "counts": obj["rle"]["counts"].encode("ascii")}).astype(bool)
        color = palette[identity % len(palette)]
        layer[mask] = color
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outlines.append((contours, color))
        moments = cv2.moments(mask.astype(np.uint8))
        labels.append((int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]), identity, color))
    image = cv2.addWeighted(layer, config.MASK_ALPHA, image, 1 - config.MASK_ALPHA, 0)
    for contours, color in outlines:
        cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
    for x, y, identity, color in labels:
        cv2.putText(image, f"ID {identity}", (max(0, x - 15), max(18, y)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    cv2.putText(image, f"SAM3 | frame {frame_index + 1} | objects {len(objects)}", (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def frame_rows(objects, frame, image_path, time_s, width, height, fps, time_source, video_id, run_id):
    """一帧对象 -> CSV 行；纯构造函数，不加载模型、不写文件。

    objects 为 extract_objects 返回的原生 ID/掩膜；空列表仍生成 empty_frame 行。
    time_s 来自本段统一时间轴；image_path 是绝对路径。图像坐标始终以原图像素计。
    重叠个体不做互斥裁剪；每条鱼的 COCO RLE 单独保存，便于恢复完整轮廓。
    """
    base = dict(schema_version=1, run_id=run_id, record_type="empty_frame", video_id=video_id,
                frame_id=frame, time_s=float(time_s), fps=float(fps), time_source=time_source,
                image_path=str(image_path), image_width=width, image_height=height)
    if not objects:
        return [base]
    seen, rows = set(), []
    for obj in objects:
        identity = int(obj["id"])
        if identity in seen:
            raise ValueError(f"同一帧重复的 track_id={identity}。")
        seen.add(identity)
        rle = {"size": obj["rle"]["size"], "counts": obj["rle"]["counts"].encode("ascii")}
        x, y, w, h = obj["bbox_xywh"]
        rows.append(dict(base, record_type="observation", track_id=identity, category_id=1,
                         score=float(obj["score"]), area_px=int(coco.area(rle)),
                         bbox_x=x, bbox_y=y, bbox_width=w, bbox_height=h,
                         mask_height=height, mask_width=width, mask_rle=obj["rle"]["counts"]))
    return rows


def _run(config):
    """沿用原单卡模型与连续会话；边推理边写临时 CSV，完整成功后原子发布。"""
    validate_settings(config)
    paths, (width, height) = collect_images(config)
    times, source = standard_times(config, paths)
    io_utils, video_module, predictor_class = initialize_runtime(config)
    free, total = torch.cuda.mem_get_info()
    print(f"显卡：{torch.cuda.get_device_name()}；可用显存 {free / 2**30:.2f}/{total / 2**30:.2f} GiB", flush=True)
    print(f"图片 {len(paths)} 张，{width}x{height}，单次会话连续跟踪。", flush=True)
    output = config.OUTPUT_DIR
    info = output / "Information"
    info.mkdir(parents=True, exist_ok=True)
    temp_csv, temp_video = info / ".frames.pending.csv", output / ".tracking.pending.mp4"
    # 仅清理本流程明确拥有的固定文件，不遍历删除用户目录。旧 result.csv 基于旧掩膜，
    # 必须撤下，防止用户在重新分割后误把旧几何表当成新结果。
    for owned in (info / "frames.csv", info / "result.csv", output / "tracking.mp4", temp_csv, temp_video):
        owned.unlink(missing_ok=True)
    run_id = uuid4().hex
    video_id = str(config.IMAGE_DIR.resolve())
    original_loader = video_module.load_resource_as_video_frames

    def lazy_load(resource_path, image_size, offload_video_to_cpu, img_mean=(.5,) * 3, img_std=(.5,) * 3, **kwargs):
        if str(resource_path) == str(config.IMAGE_DIR):
            return BoundedFrames(paths, image_size, img_mean, img_std, offload_video_to_cpu,
                                 io_utils._load_img_as_tensor, config.FRAME_CACHE_SIZE), height, width
        return original_loader(resource_path, image_size, offload_video_to_cpu,
                               img_mean=img_mean, img_std=img_std, **kwargs)

    video_module.load_resource_as_video_frames = lazy_load
    predictor, session, writer = None, None, None
    count, started, published = 0, time.monotonic(), False
    try:
        print("正在加载模型……", flush=True)
        predictor = predictor_class(checkpoint_path=str(config.CHECKPOINT), bpe_path=str(config.BPE_PATH), image_size=config.IMAGE_SIZE)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            session = predictor.handle_request({"type": "start_session", "resource_path": str(config.IMAGE_DIR)})["session_id"]
            state = predictor._ALL_INFERENCE_STATES[session]["state"]
            state["cached_frame_outputs"] = MaskCache(state["cached_frame_outputs"], nested=True)
            predictor.handle_request({"type": "add_prompt", "session_id": session, "frame_index": 0, "text": config.PROMPT})
            request = {"type": "propagate_in_video", "session_id": session,
                       "propagation_direction": "forward", "start_frame_index": 0, "max_frame_num_to_track": len(paths)}
            writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), config.FPS, (width, height))
            if not writer.isOpened():
                raise RuntimeError("无法打开 MP4 编码器。")
            with temp_csv.open("w", encoding="utf-8-sig", newline="") as sink:
                table = csv.DictWriter(sink, fieldnames=FRAME_FIELDS)
                table.writeheader()
                for result in predictor.handle_stream_request(request):
                    index = int(result["frame_index"])
                    if index != count or index >= len(paths):
                        raise ValueError("SAM3 输出帧号不连续。")
                    objects = extract_objects(config, result["outputs"], width, height)
                    table.writerows(frame_rows(objects, index, paths[index], times[index], width, height,
                                               config.FPS, source, video_id, run_id))
                    writer.write(render_frame(config, paths[index], objects, index))
                    state["cached_frame_outputs"].offload(index)
                    for tracker in state["tracker_inference_states"]:
                        for identity, history in list(tracker["mask_inputs_per_obj"].items()):
                            if not isinstance(history, MaskCache):
                                history = MaskCache(history)
                                tracker["mask_inputs_per_obj"][identity] = history
                            history.offload_before(index - config.PROMPT_RESIDENT_FRAMES)
                    count += 1
                    if count == 1 or count % 10 == 0 or count == len(paths):
                        sink.flush()
                        print(f"已完成 {count}/{len(paths)} 帧；当前 {len(objects)} 条鱼；耗时 {time.monotonic()-started:.1f} 秒", flush=True)
            if count != len(paths):
                raise RuntimeError(f"输出不完整：{count}/{len(paths)} 帧。")
        writer.release()
        writer = None
        cap = cv2.VideoCapture(str(temp_video))
        encoded_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        cap.release()
        if not ok or encoded_count != count or frame.shape[:2] != (height, width):
            raise RuntimeError(f"视频核验失败，实际 {encoded_count} 帧，预期 {count} 帧。")
        temp_video.replace(output / "tracking.mp4")
        temp_csv.replace(info / "frames.csv")
        published = True
        print(f"完成：{info / 'frames.csv'}\n下一步单独运行 geometry.py。", flush=True)
        return info / "frames.csv"
    finally:
        if writer is not None:
            writer.release()
        video_module.load_resource_as_video_frames = original_loader
        if predictor is not None and session is not None:
            predictor.handle_request({"type": "close_session", "session_id": session})
        if not published:
            temp_csv.unlink(missing_ok=True)
            temp_video.unlink(missing_ok=True)


@dataclass(frozen=True)
class TrackerSettings:
    """不可变参数集合；字段含义见顶部同名参数。无需单独配置文件。"""
    IMAGE_DIR: Path = IMAGE_DIR
    CHECKPOINT: Path = CHECKPOINT
    BPE_PATH: Path = BPE_PATH
    MODEL_CODE_DIR: Path = MODEL_CODE_DIR
    OUTPUT_DIR: Path = OUTPUT_DIR
    PROMPT: str = PROMPT
    FPS: float = FPS
    IMAGE_SIZE: int = IMAGE_SIZE
    SCORE_THRESHOLD: float = SCORE_THRESHOLD
    MAX_FRAMES: int = MAX_FRAMES
    GPU_INDEX: int = GPU_INDEX
    CPU_THREADS: int = CPU_THREADS
    SEED: int = SEED
    FRAME_CACHE_SIZE: int = FRAME_CACHE_SIZE
    PROMPT_RESIDENT_FRAMES: int = PROMPT_RESIDENT_FRAMES
    MASK_ALPHA: float = MASK_ALPHA


_RUN_LOCK = Lock()


def run(image_dir=None, output_dir=None, *, checkpoint=None,
        fps=None, max_frames=None, settings=None):
    r"""Python 函数入口：返回本次完成的 frames.csv 绝对路径。

    image_dir：图片目录；output_dir：Tracker 输出目录；checkpoint：完整权重路径。
    fps/max_frames：非 None 时覆盖设置；settings：可选 TrackerSettings。
    路径省略/传 None 时采用 settings 对应路径；未传 settings 则采用顶部默认值。
    显式参数优先于 settings，不改变全局设置。所有路径必须绝对路径。
    示例（通过 importlib.util.spec_from_file_location 以 top_tracker 名称载入脚本后）：
        tracker.run(image_dir=r"E:\Phd\FishBehavior\Top\Tracker\Input\Images",
                    max_frames=30)
    不以 sam3 名称导入入口，避免与模型包同名；直接执行本文件最简单。
    重复运行会覆盖已知输出，并撤下旧 result.csv；失败抛异常，不发布部分帧表。
    """
    config = TrackerSettings() if settings is None else settings
    if not isinstance(config, TrackerSettings):
        raise TypeError("settings 必须为 TrackerSettings。")
    changes = {key: Path(value) for key, value in
               (("IMAGE_DIR", image_dir), ("OUTPUT_DIR", output_dir), ("CHECKPOINT", checkpoint)) if value is not None}
    changes.update({key: value for key, value in (("FPS", fps), ("MAX_FRAMES", max_frames)) if value is not None})
    with _RUN_LOCK:
        return _run(replace(config, **changes))


run_tracking = run

if __name__ == "__main__":
    run()
