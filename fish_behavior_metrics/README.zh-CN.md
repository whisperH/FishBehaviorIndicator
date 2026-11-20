# 鱼类行为指标分析

[English](README.md)

本项目包含四个可独立调用的行为指标模块及其可视化页面：

1. 鳃动频率
2. 鱼体与水平面夹角
3. 鱼的水层分布
4. 鱼体中心运动速度

四个指标都使用统一的标准输入和标准输出格式。

## 目录结构

### 鳃动频率

- `gill_frequency.py`：鳃动频率计算、标准行为 API、命令行 API
- `visualization/gill_viewer.py`：鳃动频率可视化 HTTP 服务
- `visualization/gill_viewer.html`：鳃动频率可视化页面
- `results/gill/gill_data.json`：鳃动可视化数据
- `results/gill/analysis/`：鳃动分析明细和峰值结果

### 鱼体角度

- `angle_analysis.py`：SAM3 鱼体分割、角度计算、标准角度 API
- `visualization/angle_viewer.py`：角度视频 HTTP 服务
- `visualization/angle_viewer.html`：角度视频可视化页面
- `results/angle/`：角度、掩膜和汇总结果

### 水层分布

- `water_layer_analysis.py`：水层比例、分层和窗口分布计算
- `visualization/water_layer_viewer.py`：水层视频 HTTP 服务
- `visualization/water_layer_viewer.html`：水层可视化页面
- `results/water_layer/`：逐帧水层结果和窗口分布结果

### 中心速度

- `speed_analysis.py`：掩膜几何中心、速度和标准行为 API
- `visualization/speed_viewer.py`：速度视频 HTTP 服务
- `visualization/speed_viewer.html`：速度可视化页面
- `results/speed/`：逐帧速度、轨迹汇总和可视化数据

### 共享文件

- `common/segmentation_utils.py`：SAM RLE 解码和掩膜几何中心计算
- `datasets/`：视频和 MOT 标注
- `models/sam3.pt`：SAM3 模型权重
- `assets/bpe_simple_vocab_16e6.txt.gz`：SAM3 BPE 文本词表
- `requirements.txt`：全部 Python 依赖

## 目录作用

| 目录 | 作用 |
|---|---|
| `datasets/` | 输入视频和 MOT 标注 |
| `models/` | SAM3 模型权重 |
| `assets/` | SAM3 BPE 分词词表 |
| `common/` | 检测、跟踪、分割等共享工具代码 |
| `visualization/` | 四个指标的 viewer Python 服务和 HTML 页面 |
| `results/gill/` | 鳃动可视化缓存和分析表格 |
| `results/angle/` | 鱼体角度、SAM 掩膜和汇总结果 |
| `results/water_layer/` | 水深比例、分层和窗口分布结果 |
| `results/speed/` | 几何中心速度、轨迹统计和汇总结果 |
| `.venv/` | 本地 Python 环境，不属于源码 |
| `.idea/` | 可选的 PyCharm 配置 |

## 安装

```powershell
python -m pip install -r requirements.txt
```

## 标准输入格式

四个计算函数都接收标准 `data` 记录数组：

```json
{
  "data": [
    {
      "track_id": 1,
      "model_info": {},
      "video_info": {},
      "view_info": "left side",
      "category_id": 1,
      "frame_id": 1,
      "segmentation": {
        "size": [480, 640],
        "counts": [245, 5, 35, 5]
      },
      "area": 150000.0,
      "bbox": [0, 0, 640, 300]
    }
  ]
}
```

字段说明：

- `track_id`：目标轨迹 ID。
- `model_info`：模型附加信息，当前指标通常忽略。
- `video_info`：视频附加信息，当前指标通常忽略。
- `view_info`：视角，例如 `top`、`left side`、`right side`、`bottom`。
- `category_id`：目标类别 ID。
- `frame_id`：帧编号，支持从 0 或从 1 开始。
- `segmentation`：目标区域掩膜。鳃动区域当前暂由人工标注，后续可替换为
  检测算法输出的同格式掩膜。
- `segmentation.size`：`[高度, 宽度]`。
- `segmentation.counts`：COCO 压缩 RLE 字符串，或普通游程数组。
- `area`：掩膜面积，作为附加信息保存。
- `bbox`：附加区域框 `[x, y, width, height]`；鳃动频率计算不使用。

## 标准输出格式

四个计算函数都返回：

```json
{
  "data": [
    {
      "track_id": 1,
      "frame_id": 3,
      "behavior_indicator": "fish_speed",
      "behavior_value": 0,
      "unit": "pixels/s",
      "frame_range": [1, 2]
    }
  ]
}
```

字段说明：

- `track_id`：目标轨迹 ID。
- `frame_id`：本次结果对应的最新帧。
- `behavior_indicator`：指标名称。
- `behavior_value`：指标数值。
- `unit`：指标单位。
- `frame_range`：本次实际累计使用的帧 ID 列表。

多帧指标在窗口未累计完成时返回 `behavior_value = 0`，但会保留已经累计的 `frame_range`。

## 1. 鳃动频率

### Python API

```python
from gill_frequency import calculate_gill_frequency

result = calculate_gill_frequency(
    data=payload["data"],
    video_path="datasets/video.mp4",
    window_seconds=10.0,
    min_distance_seconds=0.25,
    prominence=0.05,
    minimum_coverage=0.80,
)
```

输出指标：

- `behavior_indicator = "gill_frequency"`
- `unit = "times/min"`

计算过程：

1. 根据每一帧鳃动区域 RLE 掩膜计算视频灰度均值。
2. 按轨迹累计帧。
3. 对窗口信号做 `0.60–1.50 Hz` 带通滤波。
4. 平滑并归一化。
5. 检测峰值。
6. 将峰数换算为每分钟鳃动次数。

### 命令行

```powershell
python gill_frequency.py api `
  --input input.json `
  --video "datasets/video.mp4" `
  --output output.json `
  --window-seconds 10
```

### 可视化

```powershell
python visualization/gill_viewer.py
```

访问：http://127.0.0.1:8768/

## 2. 鱼体与水平面夹角

### Python API

```python
from angle_analysis import calculate_fish_body_angle

result = calculate_fish_body_angle(data=payload["data"])
```

输出指标：

- `behavior_indicator = "fish_body_angle"`
- `unit = "degree"`
- 角度范围：`0–90°`

计算过程：

1. 读取 SAM 掩膜。
2. 计算全部掩膜像素的质心。
3. 计算像素坐标协方差矩阵。
4. 取最大特征值对应的特征向量作为鱼体主轴。
5. 沿主轴统计宽度轮廓，得到备用头尾方向。
6. 如存在 SAM3 鱼头掩膜，选择与鱼体端点最匹配的鱼头并修正头尾方向。
7. 计算鱼体主轴与水平线的无符号夹角，范围为 `0–90°`。

头尾端点仍会单独保存并用于可视化箭头，但不改变角度大小。无论鱼头朝左上、
左下、右上还是右下，鱼体与水平线的夹角都在 `0–90°`。

### SAM3 离线分析

```powershell
python angle_analysis.py
```

默认逐帧处理全部视频帧；如需抽样，可通过 `--stride` 指定间隔。

默认读取 `results/angle/head_mask_shapes.json`；也可以显式指定其他鱼头
掩膜文件：

```powershell
python angle_analysis.py --head-masks path/to/mask_shapes.json
```

鱼头匹配失败时自动保留宽度轮廓备用结果。分析完成后，鱼头掩膜会整合到
`results/angle/`，角度、汇总、掩膜和可视化页面由同一次流程统一生成。

输出：

- `results/angle/frame_angles.csv`
- `results/angle/track_angle_summary.csv`
- `results/angle/mask_shapes.json`
- `results/angle/head_mask_shapes.json`
- `results/angle/angle_summary.json`

### 可视化

```powershell
python visualization/angle_viewer.py
```

访问：http://127.0.0.1:8770/

## 3. 鱼的水层分布

### Python API

```python
from water_layer_analysis import calculate_water_layer_distribution

result = calculate_water_layer_distribution(
    data=payload["data"],
    surface_y=232.0,
    bottom_y=553.0,
    window_frames=150,
    minimum_coverage=0.80,
)
```

水深比例定义：

```text
depth_ratio =
    (SAM掩膜几何中心y - surface_y)
    / (bottom_y - surface_y)
```

- `0`：水面
- `1`：水底
- `0–1/3`：表层
- `1/3–2/3`：中层
- `2/3–1`：底层

`surface_y` 和 `bottom_y` 必须与输入掩膜使用同一像素坐标系。演示视频使用：

- 水面：`y = 232`
- 水底：`y = 553`

输出指标：

- `segmentation_center_x`：SAM 掩膜几何中心 x，单位 `pixel`
- `segmentation_center_y`：SAM 掩膜几何中心 y，单位 `pixel`
- `water_depth_percent`：水深百分比，单位 `percent`
- `water_layer_index`：`0=表层，1=中层，2=底层`
- `water_depth_mean`：窗口平均水深比例，单位 `ratio`
- `water_layer_top_ratio`：表层时间占比
- `water_layer_middle_ratio`：中层时间占比
- `water_layer_bottom_ratio`：底层时间占比

窗口类指标未满足帧数要求时返回 0；瞬时中心、百分比和分层仍按最新帧返回。

### 命令行

```powershell
python water_layer_analysis.py `
  --mask-shapes "results/angle/mask_shapes.json" `
  --video "datasets/angle.mp4" `
  --output "results/water_layer" `
  --surface-y 232 `
  --bottom-y 553 `
  --window-seconds 5
```

输出：

- `results/water_layer/frame_water_layer.csv`
- `results/water_layer/track_water_layer_summary.csv`
- `results/water_layer/water_layer_distribution.json`

### 可视化

```powershell
python visualization/water_layer_viewer.py
```

访问：http://127.0.0.1:8771/

页面支持实时修改水面和水底像素 `y` 坐标，并重新计算水深百分比和分层。

## 4. 鱼体中心运动速度

### Python API

```python
from speed_analysis import calculate_fish_speed

result = calculate_fish_speed(
    data=payload["data"],
    fps=30.0,
    window_frames=10,
)
```

速度定义：

```text
velocity = linear_fit(mask_center, frame_index) * fps
speed = hypot(velocity_x, velocity_y)
```

- 对同一轨迹最近 10 个有效帧的掩膜几何中心分别拟合 `x(frame)`、`y(frame)`。
- 拟合斜率乘以 FPS 得到 `fish_velocity_x`、`fish_velocity_y`。
- 速度大小和方向由拟合后的速度分量计算。
- 窗口未满 10 帧时，速度返回 0。

输出指标：

- `segmentation_center_x`：SAM 掩膜几何中心 x，单位 `pixel`
- `segmentation_center_y`：SAM 掩膜几何中心 y，单位 `pixel`
- `fish_velocity_x`：水平速度分量，单位 `pixels/s`
- `fish_velocity_y`：垂直速度分量，单位 `pixels/s`
- `fish_speed`：中心速度大小，单位 `pixels/s`
- `fish_velocity_direction`：速度方向，范围 `0–360°`

### 命令行

```powershell
python speed_analysis.py
```

默认从 `results/angle/mask_shapes.json` 的轮廓重新计算逐帧掩膜几何中心，
无需重新运行 SAM3。

输出：

- `results/speed/frame_speed.csv`
- `results/speed/track_speed_summary.csv`
- `results/speed/speed_distribution.json`
- `results/speed/speed_summary.json`

### 可视化

```powershell
python visualization/speed_viewer.py
```

访问：http://127.0.0.1:8772/

## 可视化与计算分离

四个 viewer Python 文件只负责：

- 启动 HTTP 服务
- 提供 HTML 页面
- 提供视频流

指标计算全部在对应的 `*_analysis.py` 或 `gill_frequency.py` 中完成。

## 环境要求

- Windows、Linux 或 macOS
- Python 3.12 64 位
- NVIDIA CUDA 环境（SAM3 GPU 推理）
- 首次安装依赖和下载模型时需要网络
