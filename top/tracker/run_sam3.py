"""顶部视角 SAM3 分割入口，按图片顺序进行一次连续跟踪。

输出 standard_data.jsonl.gz 供 main.py 的行为分析使用，每行是单帧的
{"data": [鱼1标准记录, 鱼2标准记录, ...]}。标准记录的详细字段定义见
data_format.py 顶部及其中的公共构造函数。整段分割完成后再执行分析。
"""
from pathlib import Path
# 支持绝对路径直接运行本文件；正常包导入时无需修改模块搜索路径。
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import dataclass, replace
from threading import Lock


# ==================== 只需要修改这里 ====================
IMAGE_DIR = Path(r"E:\Phd\temperature\sam3\000000")
CHECKPOINT = Path(r"E:\Phd\temperature\sam3\top\tracker\models\sam3.pt")
BPE_PATH = Path(r"E:\Phd\temperature\sam3\top\tracker\models\bpe_simple_vocab_16e6.txt.gz")
MODEL_CODE_DIR = Path(r"E:\Phd\temperature\sam3\top\tracker\vendor")
OUTPUT_DIR = Path(r"E:\Phd\temperature\sam3\top\output\tracking")

PROMPT = "fish"
FPS = 30.0                  # 图片序列的采样帧率，同时用于效果视频播放。
IMAGE_SIZE = 1008             # 保留原设置；降低会改变分割结果
SCORE_THRESHOLD = 0.25       # 输出过滤阈值，模型内部检测阈值保留原值
MAX_FRAMES = 0               # 0 = 全部；试运行可设 30
GPU_INDEX = 0
CPU_THREADS = 8              # 不再限制 CPU 亲和性或只能使用 25% 核心
SEED = 0                    # 固定随机种子，便于同环境内复核。
FRAME_CACHE_SIZE = 8         # 最近使用的预处理图像缓存数量。
PROMPT_RESIDENT_FRAMES = 32  # 较早的跟踪提示掩膜转为CPU压缩缓存。
SAVE_MASK_PNG = True         # 16 位 PNG：背景=0，像素值=原始 SAM3 ID+1
SAVE_OVERLAY_IMAGES = False  # 如需每张图片的彩色效果，改为 True
MASK_ALPHA = 0.35
# ========================================================

from data_format import make_segmentation_frame, prepare_output_directory
from uuid import uuid4

import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
from collections import OrderedDict
from collections.abc import MutableMapping

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco


def initialize_runtime(config):
    """初始化本地单GPU模型、随机种子与数值精度，使用随项目附带的模型代码。"""
    # 使用本项目附带的模型代码，不需要安装整个服务器项目。
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


def sha256(path):
    """分块计算权重校验值，便于记录每次运行实际使用的模型文件。"""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    """写入UTF-8运行状态，分割完成/失败时更新同一批次的run_info。"""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_image(path, image):
    """按路径扩展名编码并保存图片，支持Windows中文路径。"""
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")
    data.tofile(str(path))  # 支持 Windows 中文路径


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


def render_frame(config, path, objects, frame_index):
    """将掩膜/轮廓/ID叠加到原图；标签PNG以ID+1存储，0留给背景。"""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    layer = image.copy()
    id_mask = np.zeros(image.shape[:2], dtype=np.uint16)
    # 固定颜色随原生 ID 保持一致，画面样式参考提供的 MP4。
    palette = [(255, 150, 20), (220, 210, 20), (60, 70, 240),
               (20, 215, 245), (220, 70, 200), (60, 205, 100)]
    outlines, labels = [], []
    for obj in objects:
        identity = obj["id"]
        mask = coco.decode({"size": obj["rle"]["size"],
                            "counts": obj["rle"]["counts"].encode("ascii")}).astype(bool)
        color = palette[identity % len(palette)]
        layer[mask] = color
        if config.SAVE_MASK_PNG:
            if not 0 <= identity < 65535:
                raise ValueError("ID 超出 16 位 PNG 范围，请将 SAVE_MASK_PNG 改为 False；RLE 保留原 ID。")
            id_mask[mask] = identity + 1
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outlines.append((contours, color))
        moments = cv2.moments(mask.astype(np.uint8))
        x, y = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
        labels.append((x, y, identity, color))
    image = cv2.addWeighted(layer, config.MASK_ALPHA, image, 1 - config.MASK_ALPHA, 0)
    for contours, color in outlines:
        cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
    for x, y, identity, color in labels:
        cv2.putText(image, f"ID {identity}", (max(0, x - 15), max(18, y)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    cv2.putText(image, f"SAM3 | frame {frame_index + 1} | objects {len(objects)}", (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
    return image, id_mask


def standard_times(config, paths):
    """标准接口使用采集时间：全部文件名_t有效且递增时减去首帧时间。

    若任一图片没有有效_t，整段统一回退到frame/FPS，避免混合两种时间轴。
    frames.csv原播放时间仍由frame/FPS生成，实际测量时间写在标准记录中。
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


def _run(config):
    """完整离线分割：一次会话持续跟踪，结果逐帧写盘，最后核验帧数与视频。"""
    validate_settings(config)
    paths, (width, height) = collect_images(config)
    measurement_times, measurement_source = standard_times(config, paths)
    io_utils, video_module, predictor_class = initialize_runtime(config)
    free, total = torch.cuda.mem_get_info()
    print(f"显卡：{torch.cuda.get_device_name()}；可用显存 {free / 2**30:.2f}/{total / 2**30:.2f} GiB", flush=True)
    print(f"图片：{len(paths)} 张，{width}x{height}；提示词：{config.PROMPT}；单次连续跟踪", flush=True)
    # 固定输出位置；清理本跟踪器生成的旧文件，短视频重跑也不会残留旧帧。
    output = prepare_output_directory(config.OUTPUT_DIR, [
        "run_info.json", "frames.csv", "tracks.csv", "native_masks.jsonl.gz",
        "standard_data.jsonl.gz", "tracking.mp4", "preview.jpg",
        "masks/[0-9]*.png", "overlays/[0-9]*.jpg"])
    # 文件夹名固定为tracking；用独立标识区分本次视频批次，不用文件夹名充当video_id。
    video_id = uuid4().hex
    if config.SAVE_MASK_PNG:
        (output / "masks").mkdir(exist_ok=True)
    if config.SAVE_OVERLAY_IMAGES:
        (output / "overlays").mkdir(exist_ok=True)
    metadata = {"status": "running", "image_directory": str(config.IMAGE_DIR),
                "checkpoint": str(config.CHECKPOINT), "checkpoint_sha256": sha256(config.CHECKPOINT),
                "prompt": config.PROMPT, "fps": config.FPS, "image_size": config.IMAGE_SIZE, "score_threshold": config.SCORE_THRESHOLD,
                "seed": config.SEED, "frames": len(paths), "resolution": [width, height],
                "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "autocast_dtype": "bfloat16",
                "sessions": 1, "prompts": 1, "mask_png_value": "original SAM3 ID + 1; background=0",
                "timing": "frame_index / FPS; synthetic playback time",
                "view_info": "top", "category_id": 1, "video_id": video_id,
                "standard_data_file": "standard_data.jsonl.gz",
                "standard_time_source": measurement_source,
                "standard_rle": "COCO column-major uncompressed integer runs, background first"}
    write_json(output / "run_info.json", metadata)
    with (output / "frames.csv").open("w", encoding="utf-8-sig", newline="") as f:
        table = csv.writer(f)
        table.writerow(["frame", "filename", "absolute_path", "time_seconds"])
        table.writerows((i, p.name, str(p), i / config.FPS) for i, p in enumerate(paths))

    original_loader = video_module.load_resource_as_video_frames
    def lazy_load(resource_path, image_size, offload_video_to_cpu, img_mean=(.5,) * 3, img_std=(.5,) * 3, **kwargs):
        if str(resource_path) == str(config.IMAGE_DIR):
            return BoundedFrames(paths, image_size, img_mean, img_std, offload_video_to_cpu,
                                 io_utils._load_img_as_tensor, config.FRAME_CACHE_SIZE), height, width
        return original_loader(resource_path, image_size, offload_video_to_cpu,
                               img_mean=img_mean, img_std=img_std, **kwargs)
    video_module.load_resource_as_video_frames = lazy_load
    predictor, session, writer = None, None, None
    count, started = 0, time.monotonic()
    try:
        # 单卡基类直接运行，避免服务器多进程、NCCL、端口和 GPU 空闲检查。
        print("正在加载模型，首次加载需要一些时间……", flush=True)
        predictor = predictor_class(checkpoint_path=str(config.CHECKPOINT), bpe_path=str(config.BPE_PATH), image_size=config.IMAGE_SIZE)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            session = predictor.handle_request({"type": "start_session", "resource_path": str(config.IMAGE_DIR)})["session_id"]
            state = predictor._ALL_INFERENCE_STATES[session]["state"]
            state["cached_frame_outputs"] = MaskCache(state["cached_frame_outputs"], nested=True)
            predictor.handle_request({"type": "add_prompt", "session_id": session, "frame_index": 0, "text": config.PROMPT})
            request = {"type": "propagate_in_video", "session_id": session,
                       "propagation_direction": "forward", "start_frame_index": 0, "max_frame_num_to_track": len(paths)}
            writer = cv2.VideoWriter(str(output / "tracking.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), config.FPS, (width, height))
            if not writer.isOpened():
                raise RuntimeError("MP4 编码器无法打开，请检查 OpenCV 安装及输出路径。")
            with gzip.open(output / "native_masks.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) as sink, \
                    gzip.open(output / "standard_data.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) as standard_sink, \
                    (output / "tracks.csv").open("w", encoding="utf-8-sig", newline="") as f:
                table = csv.writer(f)
                table.writerow(["frame", "filename", "time_seconds", "id", "x", "y", "width", "height", "score"])
                for result in predictor.handle_stream_request(request):
                    index = int(result["frame_index"])
                    if index != count or index >= len(paths):
                        raise ValueError("SAM3 输出顺序不连续。")
                    objects = extract_objects(config, result["outputs"], width, height)
                    standard = make_segmentation_frame(objects, index, paths[index], measurement_times[index], video_id,
                                                       fps=config.FPS, checkpoint=config.CHECKPOINT)
                    standard_sink.write(json.dumps(standard, ensure_ascii=False, allow_nan=False) + "\n")
                    sink.write(json.dumps({"frame": index, "filename": paths[index].name,
                                          "time_seconds": index / config.FPS, "objects": objects}) + "\n")
                    for obj in objects:
                        table.writerow([index, paths[index].name, index / config.FPS, obj["id"], *obj["bbox_xywh"], obj["score"]])
                    image, id_mask = render_frame(config, paths[index], objects, index)
                    writer.write(image)
                    if config.SAVE_MASK_PNG:
                        save_image(output / "masks" / f"{index:06d}.png", id_mask)
                    if config.SAVE_OVERLAY_IMAGES:
                        save_image(output / "overlays" / f"{index:06d}.jpg", image)
                    if index == 0:
                        save_image(output / "preview.jpg", image)
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
                        standard_sink.flush()
                        f.flush()
                        print(f"已完成 {count}/{len(paths)} 帧；当前 {len(objects)} 条鱼；耗时 {time.monotonic() - started:.1f} 秒", flush=True)
            if count != len(paths):
                raise RuntimeError(f"输出不完整：{count}/{len(paths)} 帧。")
        writer.release()
        writer = None
        cap = cv2.VideoCapture(str(output / "tracking.mp4"))
        encoded_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
        cap.release()
        if not ok or encoded_count != count:
            raise RuntimeError(f"视频验证失败：编码 {encoded_count} 帧，预期 {count} 帧。")
        metadata.update(status="complete", completed_frames=count, seconds=time.monotonic() - started,
                        peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 2**30)
        write_json(output / "run_info.json", metadata)
        print(f"完成。分割视频：{output / 'tracking.mp4'}", flush=True)
    except BaseException:
        metadata.update(status="failed", completed_frames=count, error=traceback.format_exc())
        write_json(output / "run_info.json", metadata)
        raise
    finally:
        if writer is not None:
            writer.release()
        video_module.load_resource_as_video_frames = original_loader
        if predictor is not None and session is not None:
            predictor.handle_request({"type": "close_session", "session_id": session})

    return output

@dataclass(frozen=True)
class TrackerSettings:
    """本次跟踪配置。字段含义与文件顶部同名参数一致；所有路径使用绝对路径。

    frozen保证调用中不修改配置。单GPU运行串行化，避免SAM3加载器临时替换冲突。
    不含行为分析设置，也不传入任何行为计算函数。
    """
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
    SAVE_MASK_PNG: bool = SAVE_MASK_PNG
    SAVE_OVERLAY_IMAGES: bool = SAVE_OVERLAY_IMAGES
    MASK_ALPHA: float = MASK_ALPHA

_RUN_LOCK = Lock()

def run_tracking(image_dir=None, output_dir=None, checkpoint=None, *, fps=None,
                 max_frames=None, settings=None):
    r"""函数入口：连续图片 -> 固定结果目录(Path)，完成后可交给行为端读取。

    image_dir：图片序列绝对路径；output_dir：结果绝对路径，直接覆盖此目录中的跟踪生成文件，不追加时间戳；
    checkpoint：sam3.pt绝对路径；fps：帧率；max_frames：0全部，正整数用于试跑。
    settings：可选TrackerSettings，含词表、模型源码路径、提示词、GPU和缓存等设置。
    显式非None参数覆盖settings中的对应值；不修改模块全局参数。
    返回目录内standard_data.jsonl.gz每行是单帧标准{"data":[...]}，
    同时有frames.csv完整帧清单、run_info.json等；不返回整段大掩膜列表。
    失败抛出异常并把运行状态记为failed；仅complete结果可交给行为分析。
    示例：run_tracking(image_dir=r"E:\Phd\temperature\sam3\000000", max_frames=10)
    """
    config = settings if settings is not None else TrackerSettings()
    if not isinstance(config, TrackerSettings):
        raise TypeError("settings必须是TrackerSettings。")
    changes = {key: Path(value) for key, value in
               [("IMAGE_DIR", image_dir), ("OUTPUT_DIR", output_dir), ("CHECKPOINT", checkpoint)] if value is not None}
    changes.update({key: value for key, value in [("FPS", fps), ("MAX_FRAMES", max_frames)] if value is not None})
    config = replace(config, **changes)
    with _RUN_LOCK:
        return _run(config)

if __name__ == "__main__":
    run_tracking()
