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
from contextlib import nullcontext
import ctypes
import torch
from PIL import Image
# sam3.py与模型包同名，必须让vendor优先于脚本目录，防止导入本入口自身。
MODEL_CODE_DIR=Path(r"E:\Phd\FishBehavior\Side\Model\vendor")
sys.path.insert(0,str(MODEL_CODE_DIR))
ROOT=SIDE_ROOT
VIDEO=Path(r"E:\Phd\FishBehavior\Side\Tracker\Input\body\angle.mp4")
MOT=Path(r"E:\Phd\FishBehavior\Side\Tracker\Input\body\gt\gt.txt")
CHECKPOINT=Path(r"E:\Phd\FishBehavior\Side\Model\sam3.pt")
BPE=Path(r"E:\Phd\FishBehavior\Side\Model\bpe_simple_vocab_16e6.txt.gz")
OUTPUT=Path(r"E:\Phd\FishBehavior\Side\Tracker\Output\body")
def resolve_device(requested: str) -> str:
    """解析 auto/cpu/cuda 设备选项，并在 CUDA 不可用时给出明确错误。"""
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is false."
            )
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"

def autocast_context(device: str, enabled: bool):
    """按设备和开关创建 CUDA FP16 autocast 上下文，CPU 下返回空上下文。"""
    if device == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

def resolve_mot_file(path: Path) -> Path:
    """定位 MOT 跟踪文件，兼容目录、gt/gt.txt 和 gt.txt 三种形式。"""
    if path.is_file():
        return path
    candidates = (path / "gt" / "gt.txt", path / "gt.txt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"MOT annotation file not found under: {path}")

def load_mot(path: Path) -> pd.DataFrame:
    """读取并清洗 MOT 跟踪表，只保留有效框和正置信度记录。"""
    mot_path = resolve_mot_file(path)
    columns = [
        "frame",
        "track_id",
        "x",
        "y",
        "w",
        "h",
        "mark",
        "class_id",
        "confidence",
    ]
    if mot_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(mot_path) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name.lower().endswith("gt.txt") and not name.endswith("/")
            ]
            if not candidates:
                raise FileNotFoundError(f"gt.txt not found in archive: {mot_path}")
            data = archive.read(candidates[0])
        frame = pd.read_csv(io.BytesIO(data), header=None, names=columns)
    else:
        frame = pd.read_csv(mot_path, header=None, names=columns)

    for name in ("frame", "track_id"):
        frame[name] = frame[name].astype(int)
    for name in ("x", "y", "w", "h", "confidence"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna(subset=["x", "y", "w", "h"])
    # 小于 2 像素的框无法提供稳定的 SAM 提示区域。
    frame = frame[(frame["w"] > 1) & (frame["h"] > 1)]
    frame["confidence"] = frame["confidence"].fillna(1.0)
    frame = frame[frame["confidence"] > 0.0]
    return frame.sort_values(["frame", "track_id"]).reset_index(drop=True)

def clip_box(
    row: object, width: int, height: int
) -> tuple[float, float, float, float] | None:
    """把 MOT 框裁剪到视频边界，返回有效的 xywh 框或 None。"""
    x1 = max(0.0, min(float(width), float(row.x)))
    y1 = max(0.0, min(float(height), float(row.y)))
    x2 = max(x1 + 1.0, min(float(width), float(row.x + row.w)))
    y2 = max(y1 + 1.0, min(float(height), float(row.y + row.h)))
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < 2.0 or box_height < 2.0:
        return None
    return x1, y1, box_width, box_height

def normalized_box(
    box_xywh: tuple[float, float, float, float], width: int, height: int
) -> list[float]:
    """将像素 xywh 框转换为 SAM3 期望的归一化 cx/cy/w/h。"""
    x, y, box_width, box_height = box_xywh
    return [
        (x + box_width / 2.0) / width,
        (y + box_height / 2.0) / height,
        box_width / width,
        box_height / height,
    ]

def box_iou_xyxy(first: np.ndarray, second: np.ndarray) -> float:
    """计算两个 xyxy 框的交并比 IoU。"""
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0

def clean_mask(mask: np.ndarray, minimum_area: int) -> tuple[np.ndarray | None, int]:
    """清理 SAM 二值掩膜，保留与最大连通域同量级的前景区域。

    先做闭运算填补小孔，再删除面积过小的孤立连通域。返回清理后的掩膜和
    前景面积；面积不足时返回 (None, area)。
    """
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim > 2:
        binary = np.squeeze(binary)
    if binary.shape[0] == 0 or binary.shape[1] == 0:
        return None, 0
    binary = (binary > 0).astype(np.uint8)
    # 闭运算只连接相邻小缺口，不改变鱼体的整体朝向。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return None, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    area = int(areas[largest - 1])
    if area < minimum_area:
        return None, area
    # 保留最大连通域以及至少达到其面积 8% 的显著连通域。
    keep_threshold = max(minimum_area, int(round(area * 0.08)))
    keep_labels = np.flatnonzero(areas >= keep_threshold) + 1
    merged = np.isin(labels, keep_labels)
    merged_area = int(merged.sum())
    if merged_area < minimum_area:
        return None, merged_area
    return merged.astype(np.uint8), merged_area

def select_best_mask(
    state: dict,
    prompt_box_xywh: tuple[float, float, float, float],
    minimum_area: int,
) -> tuple[np.ndarray | None, float, float]:
    """从 SAM3 的多个候选掩膜中选择与 MOT 提示框最匹配的一个。

    评分由提示框 IoU、模型置信度和面积异常惩罚组成。mask=None 表示没有
    候选通过面积、重叠或清理检查。
    """
    masks = state.get("masks")
    scores = state.get("scores")
    if masks is None or scores is None or len(scores) == 0:
        return None, 0.0, 0.0

    prompt_x, prompt_y, prompt_w, prompt_h = prompt_box_xywh
    prompt_xyxy = np.array(
        [prompt_x, prompt_y, prompt_x + prompt_w, prompt_y + prompt_h],
        dtype=np.float32,
    )
    prompt_area = prompt_w * prompt_h
    best: tuple[float, np.ndarray | None, float, float] | None = None

    masks_cpu = masks.detach().float().cpu().numpy()
    scores_cpu = scores.detach().float().cpu().numpy()
    for index in range(masks_cpu.shape[0]):
        mask, area = clean_mask(masks_cpu[index], minimum_area)
        if mask is None:
            continue
        ys, xs = np.nonzero(mask)
        mask_box = np.array(
            [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32
        )
        overlap = box_iou_xyxy(mask_box, prompt_xyxy)
        # 与提示框几乎不重叠的掩膜通常来自其他鱼或背景。
        if overlap < 0.05:
            continue
        area_ratio = area / max(prompt_area, 1.0)
        area_penalty = 0.0
        # 掩膜远大于提示框时更不可信；过小的掩膜保留较小惩罚。
        if area_ratio > 2.5:
            area_penalty = 1.0
        elif area_ratio < 0.03:
            area_penalty = 0.5
        quality = overlap + 0.25 * float(scores_cpu[index]) - area_penalty
        if best is None or quality > best[0]:
            best = (quality, mask, float(scores_cpu[index]), overlap)

    if best is None:
        return None, 0.0, 0.0
    return best[1], best[2], best[3]

def build_model(
    checkpoint: Path,
    bpe: Path,
    device: str,
    use_fp16: bool,
    resolution: int,
) -> tuple[torch.nn.Module, Sam3Processor]:
    """加载 SAM3 权重和处理器，并按设备配置 FP32/FP16 精度。"""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")
    if not bpe.is_file():
        raise FileNotFoundError(f"SAM3 BPE vocabulary not found: {bpe}")

    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        bpe_path=str(bpe),
        device="cpu",
        enable_inst_interactivity=False,
        load_from_HF=False,
    )
    model.eval()
    if device == "cuda":
        if use_fp16:
            model = model.half()
        model = model.cuda()
    else:
        model = model.float()

    processor = Sam3Processor(
        model,
        resolution=resolution,
        device=device,
        confidence_threshold=0.25,
    )
    if device == "cuda" and use_fp16:
        original_transform = processor.transform

        def half_transform(image: object) -> torch.Tensor:
            # 权重为 FP16 时，图像预处理结果也必须保持相同 dtype。
            return original_transform(image).half()

        processor.transform = half_transform
    return model, processor

def clear_prompt(state: dict) -> None:
    """清除当前图像的文本/几何提示和候选掩膜，避免跨鱼复用。"""
    for key in ("geometric_prompt", "boxes", "masks", "masks_logits", "scores"):
        state.pop(key, None)

def add_box_prompts(
    processor: Sam3Processor,
    boxes_xywh: list[list[float]],
    state: dict,
    text_prompt: str = "fish",
) -> dict:
    """向 SAM3 状态添加一个或多个框提示，并执行 grounding 前向计算。"""
    if "language_features" not in state["backbone_out"]:
        prompt = text_prompt.strip() or "visual"
        state["backbone_out"].update(
            processor.model.backbone.forward_text([prompt], device=processor.device)
        )
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()

    dtype = next(
        parameter.dtype
        for parameter in processor.model.parameters()
        if parameter.is_floating_point()
    )
    # SAM3 几何提示使用归一化 cx/cy/w/h，并在末尾增加提示维度。
    boxes = torch.tensor(
        boxes_xywh, device=processor.device, dtype=dtype
    ).view(-1, 1, 4)
    labels = torch.ones(
        (len(boxes_xywh), 1), device=processor.device, dtype=torch.bool
    )
    state["geometric_prompt"].append_boxes(boxes, labels)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=processor.device == "cuda" and dtype == torch.float16,
    ):
        return processor._forward_grounding(state)

def add_box_prompt(
    processor: Sam3Processor,
    box: list[float],
    state: dict,
    text_prompt: str = "fish",
) -> dict:
    """单框提示的便捷包装。"""
    return add_box_prompts(processor, [box], state, text_prompt)


def run(video=VIDEO,mot=MOT,checkpoint=CHECKPOINT,bpe=BPE,output=OUTPUT,*,max_frames=0,import_mask_shapes=None):
    """run(...):生成完整RLE基础分割表；不计算角度、不调用行为或网页。
    默认逐帧单框提示，与原项目一致。import_mask_shapes仅显式导入已有简化轮廓，
    source_kind=imported_polygon；导入不能恢复原掩膜细节，禁止声称重新运行了模型。
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    meta=video_metadata(video,'imported_polygon' if import_mask_shapes else 'sam3')
    if max_frames<0:raise ValueError('max_frames must be >=0')
    limit=min(max_frames or meta['frame_count'],meta['frame_count'])
    meta['run_id']=__import__('uuid').uuid4().hex
    # 删除旧基础测量，新的frames必须重新经过geometry，防止跨批次复用。
    for name in ['frames.csv','result.csv']:
        if (output/name).exists():(output/name).unlink()
    rows=[]
    if import_mask_shapes:
        payload=json.loads(Path(import_mask_shapes).read_text('utf-8-sig'));pm=payload.get('meta',{})
        for k in ['width','height','fps']:
            if k in pm and abs(float(pm[k])-float(meta[k]))>1e-6:raise ValueError('Imported shapes metadata differs: '+k)
        by_frame={}
        for sample in payload.get('samples',[]):by_frame.setdefault(int(sample['frame']),[]).append(sample)
        seed_fields='center_x center_y angle_horizontal_deg orientation_deg directed_angle_deg elongation head_x head_y tail_x tail_y head_width tail_width'.split()
        for frame in range(1,limit+1):
            found=False
            for sample in by_frame.get(frame,[]):
                mask=fill_polygons(sample.get('polygons',[]),meta['width'],meta['height'])
                if not mask.any():continue
                box=sample.get('box',[0,0,0,0]);r=base_row(meta,frame,int(sample['track_id']))
                r.update(dict(zip(['bbox_x','bbox_y','bbox_w','bbox_h'],box)))
                r.update(mask_rle=encode_mask(mask),polygons=sample.get('polygons',[]),source_geometry={k:sample[k] for k in seed_fields if k in sample},score=sample.get('score',0),box_iou=sample.get('box_iou',0),annotation_path=str(Path(mot).resolve()))
                rows.append(r);found=True
            if not found:rows.append(base_row(meta,frame))
    else:
        # 仅实际分割分支加载模型；导入已有结果不需要GPU/模型包。
        global Sam3Processor,build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        device=resolve_device('auto')
        if device=='cuda':
            torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.set_float32_matmul_precision('high')
        model,processor=build_model(Path(checkpoint),Path(bpe),device,False,1008)
        processor.set_confidence_threshold(.25)
        tracking=load_mot(Path(mot));tracking['x']*=meta['width']/1280;tracking['w']*=meta['width']/1280;tracking['y']*=meta['height']/720;tracking['h']*=meta['height']/720
        by_frame={int(n):g for n,g in tracking.groupby('frame')}
        cap=cv2.VideoCapture(str(video))
        try:
            for frame in range(1,limit+1):
                ok,image=cap.read()
                if not ok:raise RuntimeError('Video ended early')
                group=by_frame.get(frame);count=0
                if group is not None:
                    state=processor.set_image(Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)))
                    for item in group.itertuples(index=False):
                        box=clip_box(item,meta['width'],meta['height'])
                        if box is None:continue
                        clear_prompt(state);state=add_box_prompt(processor,normalized_box(box,meta['width'],meta['height']),state,'fish')
                        mask,score,iou=select_best_mask(state,box,50)
                        r=base_row(meta,frame,int(item.track_id));r.update(dict(zip(['bbox_x','bbox_y','bbox_w','bbox_h'],box)))
                        r.update(score=score,box_iou=iou,mask_rle=encode_mask(mask) if mask is not None else '',polygons=mask_polygons(mask) if mask is not None else [],annotation_path=str(Path(mot).resolve()))
                        rows.append(r);count+=1
                if not count:rows.append(base_row(meta,frame))
                if frame%25==0:print(f'SAM3 frame {frame}/{limit}',flush=True)
        finally:cap.release()
    write_table(output/'frames.csv',rows,FRAME_FIELDS)
    print(f'Side frames: {len(rows)} rows, {limit} frames; source={meta["source_kind"]}; {output}',flush=True)
    return output/'frames.csv'

if __name__=='__main__':
    p=argparse.ArgumentParser(description='chenxi: SAM3 fish-body segmentation')
    p.add_argument('--video',default=str(VIDEO));p.add_argument('--mot',default=str(MOT));p.add_argument('--checkpoint',default=str(CHECKPOINT));p.add_argument('--bpe',default=str(BPE));p.add_argument('--output',default=str(OUTPUT));p.add_argument('--max-frames',type=int,default=0);p.add_argument('--import-mask-shapes')
    a=p.parse_args();run(a.video,a.mot,a.checkpoint,a.bpe,a.output,max_frames=a.max_frames,import_mask_shapes=a.import_mask_shapes)
