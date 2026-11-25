# Side：侧面视角分析

维护负责人：**chenxi**。目录位于 `E:\Phd\FishBehavior\Side`；顶部代码由YHC维护，本子项目不导入Top中的任何脚本。

## 项目结构与脚本职责

```text
Side/                                  chenxi负责的独立侧面子项目
├─ README.md                            本说明：流程、公式、输出、字段
├─ requirements.txt                     侧面所需依赖说明；沿用当前已验证sam3环境
├─ Model/                               模型代码与权重，不含行为计算
│  ├─ sam3.pt                           SAM3模型权重
│  ├─ bpe_simple_vocab_16e6.txt.gz        文本提示分词词表
│  └─ vendor/                           本地SAM3源码及许可证
├─ common/                              仅侧面共用的基础工具
│  ├─ __init__.py                       声明Python工具包，无计算入口
│  └─ segmentation_utils.py             CSV校验、COCO掩膜编解码、质心/PCA/头尾基础测量
├─ Tracker/                             分割和基础数据准备
│  ├─ sam3.py                           用视频及MOT框分割鱼体；沿用MOT的连续ID，保存完整RLE
│  ├─ geometry.py                       body：测质心、PCA主轴角度/伸长率、头尾端点/宽度及有效性；gill：提取鳃部框灰度均值/方差及质量
│  ├─ Input/                            本子项目原始输入
│  │  ├─ body/                          鱼体分析输入；与鳃动视频分别管理
│  │  │  ├─ angle.mp4                   鱼体视频
│  │  │  ├─ gt/gt.txt                   九列MOT框、帧号和连续ID
│  │  │  └─ head_mask_shapes.json       可选已有鱼头掩膜，辅助判头
│  │  └─ gill/                          鳃动分析输入
│  │     ├─ video.mp4                   鳃动视频
│  │     └─ gt/gt.txt                   鳃部区域MOT标注
│  └─ Output/                           基础结果；固定文件名，重跑覆盖本任务
│     ├─ body/frames.csv                每鱼每帧的分割掩膜、原图信息和来源
│     ├─ body/result.csv                鱼体基础测量，供角度/速度/水层共同使用
│     └─ gill/result.csv                鳃部灰度时序基础量，仅供鳃动分析
├─ Behavior/                            四个行为脚本独立读取基础CSV，互不调用
│  ├─ angle_analysis.py                汇总0～90度鱼体主轴角度；不加载SAM3
│  ├─ speed_analysis.py                N个有效位置线性拟合得到速度、平均速率与累计像素距离
│  ├─ water_layer_analysis.py          质心映射水深、三层归属及各层观测比例
│  ├─ gill_frequency.py                灰度时序滤波、平滑、数峰、每分钟峰率
│  └─ Output/                          分别保存angle/speed/water_layer/gill结果
└─ Visualization/                      可选网页展示，运行对应viewer.py才启动
   ├─ angle_viewer.py/.html             角度视频页面服务及页面
   ├─ speed_viewer.py/.html             速度视频页面服务及页面
   ├─ water_layer_viewer.py/.html       水层视频页面服务及页面
   ├─ gill_viewer.py/.html              鳃动曲线视频页面服务及页面
   └─ data/                            页面启动时需要的缓存，不作为计算输入
```

geometry中的“主轴角度”是基础几何测量；Behavior/angle负责选择有效记录、输出标准行为量和每鱼统计。geometry不计算游泳速度、平均速度、水层比例或鳃动峰率。

## 怎么运行

通常运行总项目的main.py菜单。也可用 `D:\Anaconda\envs\sam3\python.exe` 单独执行对应脚本。所有路径默认写在脚本顶部，均为最终目录下的绝对路径；命令行--input/--output可覆盖它们。

鱼体流程：Tracker/sam3.py → Tracker/geometry.py --task body → 自选angle_analysis.py、speed_analysis.py或water_layer_analysis.py。

鳃动流程：Tracker/geometry.py --task gill → Behavior/gill_frequency.py。鳃动不需要整鱼SAM3模型。

Tracker/sam3.py默认全视频；--max-frames N用于短片试跑，测试时应指定独立--output目录。--import-mask-shapes允许显式导入既有轮廓；source_kind=imported_polygon，绝不宣称是一次新分割。导入多边形不能恢复早先完整二值掩膜的细节。默认直接运行仍为真实SAM3推理。

各Behavior都有run(input_csv=...,output=...)，只需CSV即可计算；不存在跨Behavior调用。speed可传window_frames，water可传surface_y/bottom_y/window_seconds。所有脚本底部有__main__保护，导入时不会自动运行任务。Visualization不是计算前置条件；详细页面操作见总README。

## 算法定义与兼容范围

鱼体位置沿用原speed/water的实际算法：把简化外轮廓填充为掩膜，再求全部前景像素的算术质心。angle_center另存清理后原掩膜质心，不能将两个口径混为一谈。新SAM3保存完整RLE及简化轮廓；旧轮廓导入时RLE只代表重建掩膜。提供的原几何种子可以保留原角度；source_kind始终明确来源。

角度：全部鱼体掩膜像素PCA，主轴与水平线无符号夹角0～90度；头尾先按两端宽度判断，可用预先提供的鱼头掩膜修正。它不是顶部弯曲中轴的头部倾角，没有朝上/朝下正负。没有新增“质心速度修正鱼头”的逻辑；原结果遗留的该字段不属于当前源码算法。

速度：默认最近10个有效位置对原始frame_id线性回归，斜率乘fps为vx/vy，速率为hypot(vx,vy)，px/s。保留原算法跨缺帧使用最近N观测的规则，不按鱼头判断正负。图像右正下正，方向右0/下90/左180/上270。窗口不足时输出0，但window_ready=False；应排除这些占位值。平均速率为就绪窗口速率算术均值。总距离为相邻有效观测距离之和，与拟合平均速率×时长不必相等。

水层：默认水面y=232、水底y=553；ratio=clip((center_y-surface_y)/(bottom_y-surface_y),0,1)。小于1/3为表层，小于2/3为中层，其余底层。表内各层比例按观测数量统计，不是实际dt累计的停留秒数。窗口默认5秒，覆盖率80%；最新窗口未就绪返回0占位。顶视图的图像y不能作水深，标准接口跳过top/bottom视角。

鳃动：数据准备沿用鳃部MOT框，conf≥0.5、框可见面积≥98%才有效。提取均值和方差；行为分析保留原离线流程：补齐缺失→0.60～1.50Hz四阶零相位带通（相应曲线）→IQR归一化→11点三阶Savitzky–Golay→候选峰→显著度≥0.05且间隔≥0.25秒。raw_mean/raw_variance/filtered_mean三种曲线分别数峰，峰率=峰数×60/首末包含帧时长。三种结果不能相加；真实呼吸解释受光照、整体运动、区域跟踪与频带假设限制。

普通COCO游程数组和压缩字符串现在都按列展开正确解码，已用同一已知掩膜交叉验证。角度/速度/水层的旧标准data接口仍保留；它们与本地CSV运行入口不同。鳃动公开接口改为calculate_gill_frequency(data,fps,...)，data是已提取灰度的基础记录（track_id,frame_id,gray_mean,valid_tracking），不再接旧的video_path+RLE签名。只传掩膜不能得到亮度变化。该标准适配器给每鱼最新窗口结果，本地run保留原离线三曲线结果。

空白不是零，需检查position_valid/geometry_valid/valid_tracking/window_ready。各指标标准API的0占位为兼容原约定，不能据此断言鱼静止或没有鳃动。两个侧面视频并非同一段，ID不能跨视频直接匹配，更不能直接与顶部ID关联。

## 输出文件

角度：frame_angles.csv逐帧角度与几何、track_angle_summary.csv每鱼角度统计、angle_summary.json来源及汇总。

速度：frame_speed.csv逐帧拟合窗口、速度、方向与距离；track_speed_summary.csv每鱼速率统计与累计距离；speed_distribution.json为标准data指标记录；speed_summary.json为来源、参数及汇总，包含run_id、source_kind、video_path、视频大小和修改时间等基础批次指纹，供页面识别旧结果；center_definition明确使用简化轮廓填充后的质心。

水层：frame_water_layer.csv逐帧水深与分层；track_water_layer_summary.csv每鱼整段有效观测统计；water_layer_distribution.json每鱼最新窗口8个标准指标；water_layer_summary.json保存实际水面/水底、fps、窗口长度和汇总，供页面读取。

鳃动：analysis/frame_gray_metrics.csv基础信号的分析副本；analysis/peak_counts.csv每鱼每曲线峰数与峰率；analysis/peak_details.csv逐个峰的位置和显著度；gill_data.json保存原始/滤波/平滑曲线、候选与选择峰、区域框及元数据，供复查和可选网页使用。即使不启动网页也可检查这些分析结果。

## CSV逐列中文解释

第一行是英文表头，每个后续数据行代表表格说明中的对象。JSON数组放在单元格内，必须使用CSV解析器读取，不要按逗号直接拆整行。

### Tracker/Output/body/frames.csv

一行是一帧的一条鱼；没有观测的帧保留empty_frame。

| 英文列名 | 中文说明 |
|---|---|
| `schema_version` | 基础CSV格式版本，当前为1。 |
| `run_id` | 本次基础准备批次标识；不允许混合不同批次的记录。 |
| `record_type` | observation为一帧中的一个个体；empty_frame表示该帧没有观测。 |
| `video_path` | 原视频绝对路径；Behavior计算不打开此视频，页面展示时使用。 |
| `video_size_bytes` | 准备时原视频文件大小，供入口识别输入是否变化。 |
| `video_mtime_ns` | 准备时原视频修改时间（纳秒），供入口识别输入是否变化。 |
| `fps` | 视频帧率；速度与频率使用该值换算时间。 |
| `width` | 原视频宽度，像素。 |
| `height` | 原视频高度，像素。 |
| `frame_count` | 原视频总帧数，短帧试跑不会改变这个原始值。 |
| `frame_id` | 原视频帧号，从1开始；不是有效观测重新编号。 |
| `time_s` | 相对视频起始时间=(帧号−1)/fps，秒。 |
| `track_id` | 鱼体/区域的MOT ID。不同视频的相同ID不代表同一条鱼。 |
| `source_kind` | sam3=本项目新分割；imported_polygon=显式导入已有简化轮廓；mot_gill_box=鳃部标注框灰度。 |
| `score` | SAM3对象分数；不是指标计算的准确率。 |
| `box_iou` | 所选鱼体候选掩膜外接框与MOT提示框的交并比。 |
| `bbox_x` | MOT提示框左上角x，像素，向右增加。 |
| `bbox_y` | MOT提示框左上角y，像素，向下增加。 |
| `bbox_w` | MOT提示框宽度，像素。 |
| `bbox_h` | MOT提示框高度，像素。 |
| `mask_rle` | 清理后的鱼体完整二值掩膜COCO压缩RLE ASCII字符串；按列展开，尺寸取height/width。导入旧多边形时只能保存其重建掩膜。 |
| `polygons` | 简化外轮廓JSON数组，兼容原速度/水层的多边形重建口径，并供页面画轮廓。 |
| `source_geometry` | 导入旧轮廓时可选的原基础测量JSON种子；保留原全掩膜角度等，不能当作本次新推理。新SAM3记录为空。 |
| `annotation_path` | 用于本批准备的MOT标注位置。 |

### Tracker/Output/body/result.csv

一行是一帧的一条鱼的基础几何；保留空帧。

| 英文列名 | 中文说明 |
|---|---|
| `schema_version` | 基础CSV格式版本，当前为1。 |
| `run_id` | 本次基础准备批次标识；不允许混合不同批次的记录。 |
| `record_type` | observation为一帧中的一个个体；empty_frame表示该帧没有观测。 |
| `video_path` | 原视频绝对路径；Behavior计算不打开此视频，页面展示时使用。 |
| `video_size_bytes` | 准备时原视频文件大小，供入口识别输入是否变化。 |
| `video_mtime_ns` | 准备时原视频修改时间（纳秒），供入口识别输入是否变化。 |
| `fps` | 视频帧率；速度与频率使用该值换算时间。 |
| `width` | 原视频宽度，像素。 |
| `height` | 原视频高度，像素。 |
| `frame_count` | 原视频总帧数，短帧试跑不会改变这个原始值。 |
| `frame_id` | 原视频帧号，从1开始；不是有效观测重新编号。 |
| `time_s` | 相对视频起始时间=(帧号−1)/fps，秒。 |
| `track_id` | 鱼体/区域的MOT ID。不同视频的相同ID不代表同一条鱼。 |
| `source_kind` | sam3=本项目新分割；imported_polygon=显式导入已有简化轮廓；mot_gill_box=鳃部标注框灰度。 |
| `score` | SAM3对象分数；不是指标计算的准确率。 |
| `box_iou` | 所选鱼体候选掩膜外接框与MOT提示框的交并比。 |
| `bbox_x` | MOT提示框左上角x，像素，向右增加。 |
| `bbox_y` | MOT提示框左上角y，像素，向下增加。 |
| `bbox_w` | MOT提示框宽度，像素。 |
| `bbox_h` | MOT提示框高度，像素。 |
| `center_x` | 鱼体简化轮廓重建掩膜的质心x，像素；速度、水层沿用原项目这一口径。角度输出表中的center_x则对应基础表angle_center_x。 |
| `center_y` | 鱼体简化轮廓重建掩膜的质心y，像素；角度输出表中的center_y对应angle_center_y。 |
| `angle_center_x` | 清理后原鱼体掩膜的质心x；旧轮廓导入可由原测量种子提供。 |
| `angle_center_y` | 清理后原鱼体掩膜的质心y；旧轮廓导入可由原测量种子提供。 |
| `area` | 当前基础表掩膜前景像素数；单位像素平方。 |
| `angle_horizontal_deg` | 鱼体PCA主轴与水平线的无符号夹角0～90度；不是头部中轴倾角。 |
| `orientation_deg` | 尾指向头的图像方向角0～360度，与directed_angle_deg同义。 |
| `directed_angle_deg` | 头向方向角：右0度，下90度，左180度，上270度。 |
| `elongation` | PCA主/次特征值比的平方根，越大表示整体越细长。 |
| `head_x` | 估计头端x，像素。 |
| `head_y` | 估计头端y，像素。 |
| `tail_x` | 估计尾端x，像素。 |
| `tail_y` | 估计尾端y，像素。 |
| `head_width` | 头侧宽度轮廓均值，像素。 |
| `tail_width` | 尾侧宽度轮廓均值，像素。 |
| `direction_source` | sam3_head_mask表示匹配已有鱼头掩膜；width_profile_fallback表示用两端宽度备用判断。 |
| `head_match_score` | 候选鱼头的匹配评分；不是概率。 |
| `head_match_overlap` | 鱼头候选与鱼体重叠像素占该鱼头面积的比例。 |
| `position_valid` | 位置测量可用标记；与主轴角可用性分别判断。 |
| `geometry_valid` | 鱼体主轴、角度等基础几何是否可测。 |
| `quality_reason` | 基础数据有效或不可用的原因。 |
| `annotation_path` | 用于本批准备的MOT标注位置。 |

### Tracker/Output/gill/result.csv

一行是一帧的一个鳃部标注区域；没有观测的帧保留empty_frame。

| 英文列名 | 中文说明 |
|---|---|
| `schema_version` | 基础CSV格式版本，当前为1。 |
| `run_id` | 本次基础准备批次标识；不允许混合不同批次的记录。 |
| `record_type` | observation为一帧中的一个个体；empty_frame表示该帧没有观测。 |
| `video_path` | 原视频绝对路径；Behavior计算不打开此视频，页面展示时使用。 |
| `video_size_bytes` | 准备时原视频文件大小，供入口识别输入是否变化。 |
| `video_mtime_ns` | 准备时原视频修改时间（纳秒），供入口识别输入是否变化。 |
| `fps` | 视频帧率；速度与频率使用该值换算时间。 |
| `width` | 原视频宽度，像素。 |
| `height` | 原视频高度，像素。 |
| `frame_count` | 原视频总帧数，短帧试跑不会改变这个原始值。 |
| `frame_id` | 原视频帧号，从1开始；不是有效观测重新编号。 |
| `time_s` | 相对视频起始时间=(帧号−1)/fps，秒。 |
| `track_id` | 鱼体/区域的MOT ID。不同视频的相同ID不代表同一条鱼。 |
| `source_kind` | sam3=本项目新分割；imported_polygon=显式导入已有简化轮廓；mot_gill_box=鳃部标注框灰度。 |
| `valid_tracking` | 鳃部框置信度≥0.5、框可见面积≥98%且区域非空时为True。 |
| `confidence` | 鳃部MOT标注的置信度字段。 |
| `x1` | 鳃部实际裁剪框左边界，像素。 |
| `y1` | 鳃部实际裁剪框上边界，像素。 |
| `x2` | 鳃部实际裁剪框右边界（不含），像素。 |
| `y2` | 鳃部实际裁剪框下边界（不含），像素。 |
| `gray_mean` | 有效鳃部框内灰度均值，8位灰度强度单位；无效帧为空。 |
| `gray_variance` | 有效鳃部框内灰度方差，灰度强度平方；无效帧为空。 |
| `quality_reason` | 基础数据有效或不可用的原因。 |
| `annotation_path` | 用于本批准备的MOT标注位置。 |

### Behavior/Output/angle/frame_angles.csv

每帧的一个鱼体或鳃部区域一行。

| 英文列名 | 中文说明 |
|---|---|
| `frame` | 原视频帧号，从1开始。 |
| `time_s` | 相对视频起始时间=(帧号−1)/fps，秒。 |
| `track_id` | 鱼体/区域的MOT ID。不同视频的相同ID不代表同一条鱼。 |
| `score` | SAM3对象分数；不是指标计算的准确率。 |
| `box_iou` | 所选鱼体候选掩膜外接框与MOT提示框的交并比。 |
| `valid` | 此行鱼体角度是否可测；无效角度应排除统计。 |
| `mask_area` | 基础掩膜面积，像素平方；旧轮廓导入值是多边形重建面积，不能还原原全掩膜面积。 |
| `center_x` | 鱼体简化轮廓重建掩膜的质心x，像素；速度、水层沿用原项目这一口径。角度输出表中的center_x则对应基础表angle_center_x。 |
| `center_y` | 鱼体简化轮廓重建掩膜的质心y，像素；角度输出表中的center_y对应angle_center_y。 |
| `angle_horizontal_deg` | 鱼体PCA主轴与水平线的无符号夹角0～90度；不是头部中轴倾角。 |
| `orientation_deg` | 尾指向头的图像方向角0～360度，与directed_angle_deg同义。 |
| `directed_angle_deg` | 头向方向角：右0度，下90度，左180度，上270度。 |
| `elongation` | PCA主/次特征值比的平方根，越大表示整体越细长。 |
| `head_x` | 估计头端x，像素。 |
| `head_y` | 估计头端y，像素。 |
| `tail_x` | 估计尾端x，像素。 |
| `tail_y` | 估计尾端y，像素。 |
| `head_width` | 头侧宽度轮廓均值，像素。 |
| `tail_width` | 尾侧宽度轮廓均值，像素。 |
| `direction_source` | sam3_head_mask表示匹配已有鱼头掩膜；width_profile_fallback表示用两端宽度备用判断。 |
| `head_match_score` | 候选鱼头的匹配评分；不是概率。 |
| `head_match_overlap` | 鱼头候选与鱼体重叠像素占该鱼头面积的比例。 |

### Behavior/Output/angle/track_angle_summary.csv

每条鱼一行汇总。

| 英文列名 | 中文说明 |
|---|---|
| `track_id` | 鱼体/区域的MOT ID。不同视频的相同ID不代表同一条鱼。 |
| `valid_frames` | 该鱼有效角度观测数量。 |
| `mean_angle_deg` | 有效无符号角度算术平均，度。 |
| `median_angle_deg` | 有效无符号角度中位数，度。 |
| `std_angle_deg` | 有效无符号角度总体标准差，度。 |
| `min_angle_deg` | 最小有效无符号角度，度。 |
| `max_angle_deg` | 最大有效无符号角度，度。 |

### Behavior/Output/gill/analysis/frame_gray_metrics.csv

每帧的一个鱼体或鳃部区域一行。

| 英文列名 | 中文说明 |
|---|---|
| `frame` | 原视频帧号，从1开始。 |
| `fish_id` | 鳃部区域MOT ID；不要与另一视频鱼体ID直接对应。 |
| `time_s` | 相对视频起始时间=(帧号−1)/fps，秒。 |
| `valid_tracking` | 鳃部框置信度≥0.5、框可见面积≥98%且区域非空时为True。 |
| `confidence` | 鳃部MOT标注的置信度字段。 |
| `x1` | 鳃部实际裁剪框左边界，像素。 |
| `y1` | 鳃部实际裁剪框上边界，像素。 |
| `x2` | 鳃部实际裁剪框右边界（不含），像素。 |
| `y2` | 鳃部实际裁剪框下边界（不含），像素。 |
| `gray_mean` | 有效鳃部框内灰度均值，8位灰度强度单位；无效帧为空。 |
| `gray_variance` | 有效鳃部框内灰度方差，灰度强度平方；无效帧为空。 |

### Behavior/Output/gill/analysis/peak_counts.csv

每鱼每曲线一行汇总。

| 英文列名 | 中文说明 |
|---|---|
| `fish_id` | 鳃部区域MOT ID；不要与另一视频鱼体ID直接对应。 |
| `curve` | raw_mean=原始均值曲线；raw_variance=原始方差曲线；filtered_mean=带通均值曲线。 |
| `metric_label` | 曲线中文标签。 |
| `duration_s` | 该区域首末观测包含的帧数/fps，秒；保留原鳃动统计口径。 |
| `peak_count` | 阈值和最小间隔筛选后保留的峰数。 |
| `peak_rate_per_min` | 峰数×60/统计时长，次/分钟；三种曲线分别给出，不能相加。 |
| `peak_prominence_min` | 最终筛选最小峰显著度，默认0.05，归一化后的无量纲数。 |
| `peak_distance_min_s` | 最终相邻峰最小时间间隔，默认0.25秒。 |

### Behavior/Output/gill/analysis/peak_details.csv

每鱼每曲线的一个检测峰一行。

| 英文列名 | 中文说明 |
|---|---|
| `fish_id` | 鳃部区域MOT ID；不要与另一视频鱼体ID直接对应。 |
| `curve` | raw_mean=原始均值曲线；raw_variance=原始方差曲线；filtered_mean=带通均值曲线。 |
| `metric_label` | 曲线中文标签。 |
| `peak_number` | 本鱼本曲线中的峰序号，从1开始。 |
| `frame` | 原视频帧号，从1开始。 |
| `time_s` | 相对视频起始时间=(帧号−1)/fps，秒。 |
| `prominence` | 该峰的归一化显著度。 |

### Behavior/Output/speed/frame_speed.csv

每帧的一个鱼体或鳃部区域一行。

| 英文列名 | 中文说明 |
|---|---|
| `track_id` | 鱼体/区域的MOT ID。不同视频的相同ID不代表同一条鱼。 |
| `frame_id` | 原视频帧号，从1开始；不是有效观测重新编号。 |
| `time_s` | 相对视频起始时间=(帧号−1)/fps，秒。 |
| `center_x` | 鱼体简化轮廓重建掩膜的质心x，像素；速度、水层沿用原项目这一口径。角度输出表中的center_x则对应基础表angle_center_x。 |
| `center_y` | 鱼体简化轮廓重建掩膜的质心y，像素；角度输出表中的center_y对应angle_center_y。 |
| `velocity_x_px_s` | 最近N个有效位置对帧号回归所得水平速度，px/s，向右为正。 |
| `velocity_y_px_s` | 对应纵向速度，px/s，向下为正。 |
| `speed_px_s` | 拟合速度向量模长，px/s，非负；窗口未满写0，必须看window_ready。 |
| `direction_deg` | 拟合运动方向0～360度；与鱼头朝向无关，低速或未就绪的0度不应作真实方向解释。 |
| `displacement_px` | 当前与上一个有效观测质心间欧氏距离；首观测为0；可能跨缺帧。 |
| `dt_s` | 当前拟合窗口首末帧时间跨度，秒；不是displacement_px对应的单步时间。 |
| `window_ready` | 是否已经积累N个有效观测；速度等零值可能只是未就绪占位。 |
| `window_samples` | 当前拟合窗口的有效观测数。 |
| `window_span_frames` | 当前拟合窗口末帧号减首帧号，缺帧时可以超过N−1。 |
| `window_dx_px` | 拟合窗口首末质心横向位移，px。 |
| `window_dy_px` | 拟合窗口首末质心纵向位移，px。 |
| `window_displacement_px` | 拟合窗口首末直线距离，px；速度本身来自全部窗口点的回归。 |
| `bbox_x` | MOT提示框左上角x，像素，向右增加。 |
| `bbox_y` | MOT提示框左上角y，像素，向下增加。 |
| `bbox_w` | MOT提示框宽度，像素。 |
| `bbox_h` | MOT提示框高度，像素。 |

### Behavior/Output/speed/track_speed_summary.csv

每条鱼一行汇总。

| 英文列名 | 中文说明 |
|---|---|
| `track_id` | 鱼体/区域的MOT ID。不同视频的相同ID不代表同一条鱼。 |
| `samples` | 该鱼的有效位置/水层观测数量。 |
| `speed_samples` | 已经满足拟合窗口条件的速度样本数量。 |
| `mean_speed_px_s` | 所有就绪窗口速率的算术平均，px/s；不是总路程÷总时长。 |
| `median_speed_px_s` | 就绪窗口速率中位数，px/s。 |
| `std_speed_px_s` | 就绪窗口速率总体标准差，px/s。 |
| `max_speed_px_s` | 最大就绪窗口速率，px/s。 |
| `latest_speed_px_s` | 最新窗口速率，px/s。 |
| `total_distance_px` | 相邻有效观测质心的欧氏距离累加；与拟合平均速率×时长不必相等。 |
| `mean_center_x` | 所有有效位置x的平均值，px。 |
| `mean_center_y` | 所有有效位置y的平均值，px。 |

### Behavior/Output/water_layer/frame_water_layer.csv

每帧的一个鱼体或鳃部区域一行。

| 英文列名 | 中文说明 |
|---|---|
| `track_id` | 鱼体/区域的MOT ID。不同视频的相同ID不代表同一条鱼。 |
| `frame_id` | 原视频帧号，从1开始；不是有效观测重新编号。 |
| `depth_ratio` | (质心y−水面y)/(水底y−水面y)，裁剪到0～1；0水面，1水底。 |
| `layer_index` | 水层编号：0表层，1中层，2底层。 |
| `layer_name` | top/middle/bottom，对应表层/中层/底层。 |
| `center_x` | 鱼体简化轮廓重建掩膜的质心x，像素；速度、水层沿用原项目这一口径。角度输出表中的center_x则对应基础表angle_center_x。 |
| `center_y` | 鱼体简化轮廓重建掩膜的质心y，像素；角度输出表中的center_y对应angle_center_y。 |
| `bbox_x` | MOT提示框左上角x，像素，向右增加。 |
| `bbox_y` | MOT提示框左上角y，像素，向下增加。 |
| `bbox_w` | MOT提示框宽度，像素。 |
| `bbox_h` | MOT提示框高度，像素。 |

### Behavior/Output/water_layer/track_water_layer_summary.csv

每条鱼一行汇总。

| 英文列名 | 中文说明 |
|---|---|
| `track_id` | 鱼体/区域的MOT ID。不同视频的相同ID不代表同一条鱼。 |
| `samples` | 该鱼的有效位置/水层观测数量。 |
| `mean_depth_ratio` | 该鱼全部有效观测的平均水深比例。 |
| `top_ratio` | 表层有效观测数/全部有效观测数；不是按真实dt积分的秒数。 |
| `middle_ratio` | 中层有效观测比例。 |
| `bottom_ratio` | 底层有效观测比例。 |

## 维护边界

只修改Side；新增指标增加Behavior脚本并在总main中登记。修改共同基础量需同步geometry和使用该字段的行为脚本；改变CSV格式时更新schema及文档。Model/vendor内部包结构应保留。数据准备与分析输出固定名称，重跑覆盖对应任务，不自动运行其他指标或网页。
