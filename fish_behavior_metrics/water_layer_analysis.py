"""鱼体水层分布指标。

指标定义：只使用 SAM segmentation 全部前景像素的几何中心，将图像纵向
坐标映射到水面（0）和水底（1）之间的归一化水深：
- 0 表示水面；
- 1 表示水底；
- 1/3、2/3 为表层、中层、底层的分界。

主要数据流：
标准 data 记录 -> SAM RLE 掩膜 -> 掩膜质心 -> 归一化水深 ->
瞬时分层/百分比 -> 固定窗口平均深度与三层时间占比。

瞬时中心、百分比和分层每帧有效；窗口平均深度和三层占比需要累计足够多帧
且覆盖率达标后才返回非零值。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from common.segmentation_utils import mask_centroid, rle_to_mask


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT / "datasets" / "angle.mp4"
DEFAULT_MOT = ROOT / "datasets" / "angle"
DEFAULT_OUTPUT = ROOT / "results" / "water_layer"

LAYER_NAMES = ("top", "middle", "bottom")
DEPTH_MEAN = "water_depth_mean"
LAYER_TOP = "water_layer_top_ratio"
LAYER_MIDDLE = "water_layer_middle_ratio"
LAYER_BOTTOM = "water_layer_bottom_ratio"


def _view_supports_depth(view_info: object) -> bool:
    """判断视角是否能由图像 y 轴表示水深；俯视和底视不满足该前提。"""
    return str(view_info or "").strip().lower() not in {"top", "bottom"}


def _segmentation_center(record: dict) -> tuple[float, float]:
    """从标准记录的 RLE 掩膜计算前景像素算术中心 (x, y)。"""
    segmentation = record.get("segmentation")
    if not isinstance(segmentation, dict):
        raise ValueError("SAM segmentation is required for this metric")
    return mask_centroid(rle_to_mask(segmentation))


def _layer_index(depth_ratio: float) -> int:
    """按三等分水深返回层索引：0=表层，1=中层，2=底层。"""
    if depth_ratio < 1.0 / 3.0:
        return 0
    if depth_ratio < 2.0 / 3.0:
        return 1
    return 2


def extract_water_layer_series(
    data: list[dict],
    surface_y: float,
    bottom_y: float,
) -> dict[int, list[dict[str, float | int]]]:
    """把标准记录转换为按鱼分组、按帧排序的归一化水深观测。

    必需字段为 track_id、frame_id 和 segmentation。model_info、video_info、
    category_id、bbox、area 等元数据不参与计算。view_info 为 top/bottom 时
    该记录会被跳过，因为此类视角的图像 y 轴不能代表真实水深。

    归一化公式：
    ``(center_y - surface_y) / (bottom_y - surface_y)``。
    结果裁剪到 0–1，0 表示水面，1 表示水底。

    Returns:
    {track_id: [observation, ...]}，观测按 frame_id 升序排列。
    """
    if not isinstance(data, list) or not data:
        raise ValueError("data must be a non-empty standard record array")
    if bottom_y <= surface_y:
        raise ValueError("bottom_y must be greater than surface_y")

    series: dict[int, list[dict[str, float | int]]] = {}
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise TypeError(f"data[{index}] must be an object")
        if "track_id" not in record or "frame_id" not in record:
            raise ValueError(f"data[{index}] requires track_id and frame_id")
        if not _view_supports_depth(record.get("view_info")):
            continue
        # 使用 SAM 掩膜全部前景像素的质心，而不是 bbox 中心。
        center_x, center_y = _segmentation_center(record)
        # 将绝对像素 y 映射到水面到水底之间的 0–1 深度比例。
        ratio = float(
            np.clip(
                (center_y - surface_y) / (bottom_y - surface_y),
                0.0,
                1.0,
            )
        )
        track_id = int(record["track_id"])
        bbox = record.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            # bbox 只用于展示；缺失时退化为以质心为中心的零尺寸框。
            bbox = [center_x, center_y, 0.0, 0.0]
        series.setdefault(track_id, []).append(
            {
                "frame_id": int(record["frame_id"]),
                "depth_ratio": ratio,
                "layer_index": _layer_index(ratio),
                "center_x": center_x,
                "center_y": center_y,
                "bbox": [float(value) for value in bbox[:4]],
            }
        )
    for observations in series.values():
        # 后续窗口计算隐含依赖时间顺序，因此在返回前统一排序。
        observations.sort(key=lambda item: int(item["frame_id"]))
    return series


def _distribution_rows(
    track_id: int,
    observations: list[dict[str, float | int]],
    window_frames: int,
    minimum_coverage: float,
) -> list[dict[str, object]]:
    """为单条鱼生成最新瞬时指标和最近窗口的分布指标。"""
    frame_ids = [int(item["frame_id"]) for item in observations]
    last_frame = frame_ids[-1]
    start_frame = last_frame - window_frames + 1
    window = [
        item for item in observations if int(item["frame_id"]) >= start_frame
    ]
    used_frames = [int(item["frame_id"]) for item in window]
    # 窗口类指标只有在帧跨度和覆盖率同时满足时才计算；覆盖率同时能识别
    # 跟踪中断和重复/缺失帧造成的稀疏观测。
    coverage = len(window) / max(1, window_frames)
    ready = (
        last_frame - frame_ids[0] + 1 >= window_frames
        and coverage >= minimum_coverage
    )

    if ready:
        depths = np.asarray(
            [float(item["depth_ratio"]) for item in window],
            dtype=float,
        )
        layers = np.asarray(
            [int(item["layer_index"]) for item in window],
            dtype=int,
        )
        # 平均深度是连续比例；三层占比统计窗口内落在各层的帧比例。
        mean_depth = float(depths.mean())
        ratios = [float(np.mean(layers == index)) for index in range(3)]
    else:
        mean_depth = 0.0
        ratios = [0.0, 0.0, 0.0]

    # 瞬时指标始终取最新一帧，不受窗口是否已满影响。
    latest = observations[-1]
    latest_center_x = float(latest["center_x"])
    latest_center_y = float(latest["center_y"])
    latest_depth = float(latest["depth_ratio"])
    latest_layer = int(latest["layer_index"])
    base = {
        "track_id": int(track_id),
        "frame_id": int(last_frame),
        "frame_range": used_frames,
    }
    return [
        {
            **base,
            "behavior_indicator": "segmentation_center_x",
            "behavior_value": round(latest_center_x, 3),
            "unit": "pixel",
        },
        {
            **base,
            "behavior_indicator": "segmentation_center_y",
            "behavior_value": round(latest_center_y, 3),
            "unit": "pixel",
        },
        {
            **base,
            "behavior_indicator": "water_depth_percent",
            "behavior_value": round(latest_depth * 100.0, 3),
            "unit": "percent",
        },
        {
            **base,
            "behavior_indicator": "water_layer_index",
            "behavior_value": latest_layer,
            "unit": "layer",
        },
        {
            **base,
            "behavior_indicator": DEPTH_MEAN,
            "behavior_value": round(mean_depth, 6),
            "unit": "ratio",
        },
        {
            **base,
            "behavior_indicator": LAYER_TOP,
            "behavior_value": round(ratios[0], 6),
            "unit": "ratio",
        },
        {
            **base,
            "behavior_indicator": LAYER_MIDDLE,
            "behavior_value": round(ratios[1], 6),
            "unit": "ratio",
        },
        {
            **base,
            "behavior_indicator": LAYER_BOTTOM,
            "behavior_value": round(ratios[2], 6),
            "unit": "ratio",
        },
    ]


def calculate_water_layer_distribution(
    data: list[dict],
    surface_y: float = 232.0,
    bottom_y: float = 553.0,
    window_frames: int = 150,
    minimum_coverage: float = 0.80,
) -> dict[str, list[dict[str, object]]]:
    """计算每条鱼最新的水层位置和时间窗口内的水层分布。

    输入 data 使用统一标准记录格式，每条记录必需包含 track_id、frame_id 和
    SAM segmentation RLE。surface_y 与 bottom_y 是图像中水面和水底的 y
    像素坐标，且必须满足 bottom_y > surface_y。

    每条鱼输出 8 个标准行为指标：
    - segmentation_center_x/y: 最新掩膜质心，单位 pixel。
    - water_depth_percent: 最新归一化水深百分比，范围 0–100。
    - water_layer_index: 最新层索引，0=表层，1=中层，2=底层。
    - water_depth_mean: 窗口平均归一化水深，0=水面，1=水底。
    - water_layer_top/middle/bottom_ratio: 窗口内各层帧数占比。

    后四个窗口指标在窗口未满或覆盖率不足时返回 0；前四个瞬时指标始终取
    最新有效帧。所有记录都使用 frame_range 返回本次实际使用的帧号。
    """
    if window_frames < 1:
        raise ValueError("window_frames must be at least 1")
    series = extract_water_layer_series(data, surface_y, bottom_y)
    output: list[dict[str, object]] = []
    for track_id, observations in sorted(series.items()):
        output.extend(
            _distribution_rows(
                track_id,
                observations,
                window_frames,
                minimum_coverage,
            )
        )
    return {"data": output}


def load_mot_standard_data(
    path: Path,
    view_info: str = "left side",
    category_id: int = 1,
    max_frames: int | None = None,
    width: int = 1280,
    height: int = 720,
) -> list[dict]:
    """把 MOT 跟踪框转换成标准记录，用于没有 SAM 掩膜时的近似分析。

    每个 MOT 框会绘制成同尺寸实心矩形并编码为 COCO RLE。因此该函数得到的
    是框中心而非真实鱼体掩膜中心，只适合演示或缺少分割结果的场景。
    """
    mot_file = path / "gt" / "gt.txt" if path.is_dir() else path
    frame = pd.read_csv(
        mot_file,
        header=None,
        names=[
            "frame",
            "track_id",
            "x",
            "y",
            "w",
            "h",
            "mark",
            "class_id",
            "confidence",
        ],
    )
    frame = frame[frame["confidence"] > 0]
    if max_frames is not None:
        frame = frame[frame["frame"] <= max_frames]
    records: list[dict] = []
    for row in frame.itertuples(index=False):
        # 把浮点 MOT 框裁剪到图像范围，再生成矩形二值掩膜。
        mask = np.zeros((height, width), dtype=np.uint8)
        x1 = max(0, min(width - 1, int(round(row.x))))
        y1 = max(0, min(height - 1, int(round(row.y))))
        x2 = max(x1 + 1, min(width, int(round(row.x + row.w))))
        y2 = max(y1 + 1, min(height, int(round(row.y + row.h))))
        cv2.rectangle(mask, (x1, y1), (x2 - 1, y2 - 1), 1, -1)
        encoded = mask_utils.encode(np.asfortranarray(mask))
        records.append(
            {
                "track_id": int(row.track_id),
                "model_info": {},
                "video_info": {},
                "view_info": view_info,
                "category_id": int(category_id),
                "frame_id": int(row.frame),
                "segmentation": {
                    "size": [height, width],
                    "counts": encoded["counts"].decode("ascii"),
                },
                "area": float(row.w * row.h),
                "bbox": [
                    float(row.x),
                    float(row.y),
                    float(row.w),
                    float(row.h),
                ],
            }
        )
    return records


def load_mask_shape_standard_data(
    path: Path,
    width: int,
    height: int,
    view_info: str = "left side",
    category_id: int = 1,
    max_frames: int | None = None,
) -> list[dict]:
    """把角度分析输出的简化掩膜轮廓恢复为标准 RLE 记录。

    mask_shapes.json 只保存多边形轮廓；这里按视频尺寸重新填充为二值掩膜，
    再编码为 COCO RLE。多个轮廓会合并到同一帧、同一鱼的掩膜中。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    records: list[dict] = []
    for sample in samples:
        frame_id = int(sample["frame"])
        if max_frames is not None and frame_id > max_frames:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        # fillPoly 在同一掩膜上叠加所有轮廓，保留鱼体外周形状。
        for polygon in sample.get("polygons", []):
            points = np.asarray(polygon, dtype=np.int32)
            if len(points) >= 3:
                cv2.fillPoly(mask, [points], 1)
        if not mask.any():
            continue
        encoded = mask_utils.encode(np.asfortranarray(mask))
        records.append(
            {
                "track_id": int(sample["track_id"]),
                "model_info": {},
                "video_info": {},
                "view_info": view_info,
                "category_id": int(category_id),
                "frame_id": frame_id,
                "segmentation": {
                    "size": [height, width],
                    "counts": encoded["counts"].decode("ascii"),
                },
                "area": float(mask.sum()),
                "bbox": [float(value) for value in sample["box"]],
            }
        )
    return records


def run_analysis(
    data: list[dict],
    output: Path,
    surface_y: float,
    bottom_y: float,
    window_frames: int,
) -> None:
    """计算并导出逐帧水层、鱼级汇总以及最新窗口分布结果。"""
    output.mkdir(parents=True, exist_ok=True)
    series = extract_water_layer_series(data, surface_y, bottom_y)
    frame_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for track_id, observations in sorted(series.items()):
        for item in observations:
            frame_rows.append(
                {
                    "track_id": track_id,
                    "frame_id": int(item["frame_id"]),
                    "depth_ratio": round(float(item["depth_ratio"]), 6),
                    "layer_index": int(item["layer_index"]),
                    "layer_name": LAYER_NAMES[int(item["layer_index"])],
                    "center_x": round(float(item["center_x"]), 3),
                    "center_y": round(float(item["center_y"]), 3),
                    "bbox_x": round(float(item["bbox"][0]), 3),
                    "bbox_y": round(float(item["bbox"][1]), 3),
                    "bbox_w": round(float(item["bbox"][2]), 3),
                    "bbox_h": round(float(item["bbox"][3]), 3),
                }
            )
        depth = np.asarray(
            [float(item["depth_ratio"]) for item in observations],
            dtype=float,
        )
        layers = np.asarray(
            [int(item["layer_index"]) for item in observations],
            dtype=int,
        )
        summary_rows.append(
            {
                "track_id": track_id,
                "samples": len(observations),
                "mean_depth_ratio": round(float(depth.mean()), 6),
                "top_ratio": round(float(np.mean(layers == 0)), 6),
                "middle_ratio": round(float(np.mean(layers == 1)), 6),
                "bottom_ratio": round(float(np.mean(layers == 2)), 6),
            }
        )
    pd.DataFrame(frame_rows).sort_values(
        ["frame_id", "track_id"]
    ).to_csv(
        output / "frame_water_layer.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(summary_rows).to_csv(
        output / "track_water_layer_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    distribution = calculate_water_layer_distribution(
        data,
        surface_y=surface_y,
        bottom_y=bottom_y,
        window_frames=window_frames,
    )
    (output / "water_layer_distribution.json").write_text(
        json.dumps(distribution, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """构建水层分布命令行参数。"""
    parser = argparse.ArgumentParser(
        prog="water_layer_analysis.py",
        description="Calculate fish water-layer distribution.",
    )
    parser.add_argument("--input", help="standard behavior data JSON")
    parser.add_argument("--mot", default=str(DEFAULT_MOT))
    parser.add_argument(
        "--mask-shapes",
        default=str(ROOT / "results" / "angle" / "mask_shapes.json"),
    )
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--surface-y", type=float, default=232.0)
    parser.add_argument("--bottom-y", type=float, default=553.0)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main() -> None:
    """读取视频尺寸和标准数据，计算水层分布并写出结果文件。"""
    # Python 直接调用示例（不经过命令行）：
    # from water_layer_analysis import calculate_water_layer_distribution
    # result = calculate_water_layer_distribution(data)
    args = build_parser().parse_args()
    # 视频只用于取得宽高和 FPS，掩膜本身来自输入数据。
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if args.input:
        # 显式输入优先；其次使用角度分析产生的掩膜轮廓；最后退回 MOT 框。
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        data = payload if isinstance(payload, list) else payload.get("data")
        if not isinstance(data, list):
            raise ValueError("input must be a data array or contain data")
    elif Path(args.mask_shapes).is_file():
        data = load_mask_shape_standard_data(
            Path(args.mask_shapes),
            width=width,
            height=height,
            max_frames=args.max_frames,
        )
    else:
        data = load_mot_standard_data(
            Path(args.mot),
            max_frames=args.max_frames,
            width=width,
            height=height,
        )
    bottom_y = float(args.bottom_y)
    window_frames = max(1, int(round(args.window_seconds * fps)))
    output = Path(args.output).resolve()
    run_analysis(data, output, args.surface_y, bottom_y, window_frames)
    print(f"Saved: {output / 'frame_water_layer.csv'}")
    print(f"Saved: {output / 'track_water_layer_summary.csv'}")
    print(f"Saved: {output / 'water_layer_distribution.json'}")


if __name__ == "__main__":
    main()
