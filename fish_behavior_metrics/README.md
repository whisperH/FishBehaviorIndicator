# Gill Beat Frequency

[简体中文](README.zh-CN.md)

Modular analysis and visualization tools for gill movement and fish-body angle.

## Included data

- `datasets/video.mp4`
- `datasets/annotations_mot/gt/gt.txt`
- `datasets/annotations_mot/gt/labels.txt`

The extracted MOT directory is supported directly. A ZIP containing `gt/gt.txt` is also supported.

## Modules

- `gill_frequency.py`: gill-frequency analysis and behavior API
- `visualization/gill_viewer.py` + `visualization/gill_viewer.html`: gill-frequency visualization
- `angle_analysis.py`: SAM3 fish segmentation and angle calculation
- `visualization/angle_viewer.py` + `visualization/angle_viewer.html`: angle video visualization
- `water_layer_analysis.py`: fish water-layer distribution
- `visualization/water_layer_viewer.py` + `visualization/water_layer_viewer.html`: water-layer visualization
- `common/segmentation_utils.py`: shared SAM RLE decoding and mask-centroid helpers

## Directory layout

| Directory | Purpose |
|---|---|
| `datasets/` | Input videos and MOT annotations |
| `models/` | SAM3 model weights |
| `assets/` | SAM3 BPE tokenizer vocabulary |
| `common/` | Shared model and mask utility code |
| `visualization/` | Viewer Python services and HTML pages |
| `results/gill/` | Gill-frequency cache and analysis tables |
| `results/angle/` | Body-angle, SAM-mask, and summary results |
| `results/water_layer/` | Water-depth and layer-distribution results |
| `.venv/` | Local Python environment, not source code |
| `.idea/` | Optional PyCharm configuration |

## Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Standard behavior API

All three metric functions accept a standard `data` record array and return
`{"data": [...]}` with these output fields:

- `track_id`
- `frame_id`
- `behavior_indicator`
- `behavior_value`
- `unit`
- `frame_range`

The public functions are:

- `gill_frequency.calculate_gill_frequency`
- `angle_analysis.calculate_fish_body_angle`
- `water_layer_analysis.calculate_water_layer_distribution`

Window-based values return `0` until enough accumulated frames are available.
`frame_range` contains the actual frame ids used by that calculation.

## Gill frequency analysis

```powershell
python gill_frequency.py analyze
```

Outputs:

- `results/gill/gill_data.json`
- `results/gill/analysis/frame_gray_metrics.csv`
- `results/gill/analysis/peak_counts.csv`
- `results/gill/analysis/peak_details.csv`

## Gill frequency visualization

```powershell
python visualization/gill_viewer.py
```

Open `http://127.0.0.1:8768/`.

If `results/gill/gill_data.json` is missing, the viewer automatically calculates it first.
Use `--rebuild` to force recalculation.

## Fish body angle analysis

Use the MOT boxes as SAM3 prompts to segment each fish body and measure the
principal-axis angle relative to the image horizontal:

```powershell
python angle_analysis.py
```

The default processes every frame. Use `--stride 30` for roughly one sample per
second when a faster overview is sufficient.

Defaults:

- Video: `datasets/angle.mp4`
- MOT annotations: `datasets/angle/gt/gt.txt`
- SAM3 checkpoint: `models/sam3.pt`
- BPE vocabulary: `assets/bpe_simple_vocab_16e6.txt.gz`

Outputs:

- `results/angle/frame_angles.csv`
- `results/angle/track_angle_summary.csv`
- `results/angle/mask_shapes.json`
- `results/angle/angle_summary.json`

Standard Python API:

```python
from angle_analysis import calculate_fish_body_angle

result = calculate_fish_body_angle(data=payload["data"])
```

## Fish body angle visualization

```powershell
python visualization/angle_viewer.py
```

Open `http://127.0.0.1:8770/`.

`angle_horizontal_deg` is the unsigned angle between the mask PCA major axis
and the horizontal image direction, in the range `0-90` degrees.

## Fish water-layer distribution

```powershell
python water_layer_analysis.py `
  --mask-shapes "results/angle/mask_shapes.json" `
  --video "datasets/angle.mp4" `
  --output "results/water_layer" `
  --surface-y 232 `
  --bottom-y 553 `
  --window-seconds 5
```

The depth ratio is normalized from `0` at the surface to `1` at the bottom.
Frames are divided into top, middle, and bottom thirds.
This indicator uses the geometric center of all foreground pixels in the SAM
segmentation mask. Detection boxes are not used for the metric value.

For the included `angle.mp4`, the configured calibration coordinates are
surface `y=232` and bottom `y=553`. These can be changed in the viewer or CLI
and are only starting
values; the viewer allows changing both coordinates and recalculating live.

The Python entry point is:

```python
from water_layer_analysis import calculate_water_layer_distribution

result = calculate_water_layer_distribution(
    data=payload["data"],
    surface_y=232.0,
    bottom_y=553.0,
    window_frames=150,
)
```

The result uses the shared behavior schema with these indicators:

- `segmentation_center_x`
- `segmentation_center_y`
- `water_depth_percent`
- `water_layer_index`
- `water_depth_mean`
- `water_layer_top_ratio`
- `water_layer_middle_ratio`
- `water_layer_bottom_ratio`

Outputs:

- `results/water_layer/frame_water_layer.csv`
- `results/water_layer/track_water_layer_summary.csv`
- `results/water_layer/water_layer_distribution.json`

Start the visualization with:

```powershell
python visualization/water_layer_viewer.py
```

Open `http://127.0.0.1:8771/`.

## Sliding-window behavior API

```powershell
python gill_frequency.py api `
  --input input.json `
  --video "datasets/video.mp4" `
  --output output.json `
  --window-seconds 10
```

`input.json` may be a standard record array or an object containing a `data`
array. The output indicator is `gill_frequency` with unit `times/min`.
`frame_range` contains the actual accumulated frame ids used by the metric.
Frames are accumulated until a full smoothing window is available. Before
that point, `behavior_value` is `0`.

For gill frequency, `segmentation` currently comes from manual annotation.
The same RLE interface can later accept masks produced by a detection
algorithm; `bbox` is not used to define the measurement region.

The Python entry point is:

```python
from gill_frequency import calculate_gill_frequency

result = calculate_gill_frequency(
    data=payload["data"],
    video_path="datasets/video.mp4",
    window_seconds=10.0,
)
```

See the `calculate_gill_frequency` docstring for the meaning of every input and
output field.

## Requirements

- Windows, Linux, or macOS
- Python 3.12 64-bit
- Internet access for the first dependency installation

