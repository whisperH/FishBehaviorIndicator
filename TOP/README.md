# Top 顶部视角子项目说明

维护负责人：**YHC**。本子项目位于 `E:\Phd\FishBehavior\Top`，独立维护顶部视角的模型、输入、基础测量和行为指标；不导入 Side 子项目代码。统一菜单入口位于 `E:\Phd\FishBehavior\main.py`，也可以直接运行本子项目各脚本。

## 1. 项目关系与完整目录

```text
Top/                              顶部视角独立子项目，维护负责人 YHC
├─ README.md                     顶部运行方法、指标公式、输出和CSV逐列解释
├─ requirements.txt              顶部使用的Python依赖版本记录
├─ Model/                        顶部专用SAM3模型，不与侧面互相导入模型代码
│  ├─ sam3.pt                    SAM3模型权重
│  ├─ bpe_simple_vocab_16e6.txt.gz 文本提示所用的分词词表
│  └─ vendor/                    SAM3模型源码及许可证，保留其内部包目录
├─ Tracker/                      顶部图片的分割跟踪和基础几何测量
│  ├─ sam3.py                    分割鱼体、连续跟踪ID，保存逐鱼掩膜及帧元信息
│  ├─ geometry.py                计算质心、中轴线、头尾轴、头向角、头尾夹角、中轴体长及每鱼参考BL
│  ├─ Input/                     顶部任务输入目录
│  │  └─ Images/                 按自然顺序读取的原始顶部抽帧图片
│  └─ Output/                    顶部分割与基础测量输出目录
│     ├─ tracking.mp4            分割轮廓、鱼ID和处理帧数叠加检查视频
│     └─ Information/            基础CSV数据接口目录
│        ├─ frames.csv           每鱼每帧的COCO压缩RLE、面积、外接框和时间信息
│        └─ result.csv           每鱼每帧的几何测量、参考体长和有效性标记
└─ Behavior/                     八个独立行为分析脚本，不互相调用
   ├─ speed.py                   计算BL/s速度、平均速度和BL/s²加速度，含鱼头相对正负方向
   ├─ distance.py                计算像素和BL单位的区间、累计、正向与倒退距离
   ├─ tail.py                    根据头尾夹角波动计算摆尾频率、周期变异系数及摆尾幅度
   ├─ towards.py                 计算头部中轴相对水平的有符号朝向角，上正下负
   ├─ area.py                    按质心位置和实际帧间时间统计各平面分区停留时间及占比
   ├─ resting_time.py            按相邻帧无符号BL速率阈值统计静止、运动、未知时间和静止事件
   ├─ trajectory_entanglement.py 计算每鱼二维轨迹近交事件、去重事件数及有效暴露标准化事件率
   ├─ aggregation.py             用Delaunay邻接边计算每鱼局部平均邻鱼距离，距离小表示更紧密
   └─ Output/                    行为结果固定输出目录，重复运行覆盖对应模块结果
      ├─ speed/                  速度逐帧表、每鱼汇总、曲线图和BL视频
      ├─ distance/               像素/BL距离明细及汇总、两种单位的曲线图
      ├─ tail/                   摆尾角度、频率、幅度明细和图表视频
      ├─ towards/                朝向角度明细和图表视频
      ├─ area/                   分区停留统计表和空间分布图
      ├─ resting_time/           静止事件、有效/未知时间汇总及时间轴图
      ├─ trajectory_entanglement/ 近交事件表、每鱼汇总、曲线图和轨迹视频
      └─ aggregation/            每鱼邻接距离逐帧表、汇总、曲线图和三角剖分视频
```

数据流：原图 → sam3.py → frames.csv → geometry.py → result.csv → 自选行为脚本。总项目main.py只负责菜单、检查和启动；每个脚本仍可单独运行。八个行为脚本分别读取result.csv，不读取其他行为脚本的输出，不调用Side。Top内部没有common.py、data_format.py或自己的__init__.py；Model/vendor内的__init__.py属于模型包，应保留。

geometry.py具体负责：(1)原始掩膜面积、外接框及质心cx/cy；(2)鱼体骨架主路径和头到尾的51点中轴线；(3)头部/尾部轴单位向量；(4)头轴方向heading_deg与头尾轴夹角tail_angle_deg；(5)每帧中轴弧长body_length_px；(6)同一条鱼整批有效体长的中位数reference_body_length_px及样本数；(7)位置、几何、头向有效性及质量原因。它不计算运动速度、加速度、摆尾频率、周期摆尾幅度、停留时长或鱼间关系，这些属于Behavior层。

frames.csv是分割中间表，包含每条鱼的完整COCO压缩RLE掩膜；geometry.py解码后提取中轴线。result.csv是基础测量表，不是最终行为指标，也不包含掩膜像素；它保存行为所需的坐标、中轴、时间及质量等。两表都保留无鱼帧的时间和原图路径。掩膜尺寸按[高,宽]解释，COCO按列展开；不能用外接框代替掩膜。

所有数值路径是脚本顶部的绝对路径，移动项目后需要相应修改各脚本的路径。各行为脚本只把result.csv作为数值输入，制作视频时仍需通过image_path读取原图。若以后换视频，把该段图片单独放入输入目录，不要混合两个视频的抽帧；文件名按自然顺序排列。

## 2. 怎么运行

日常建议运行总入口 `E:\Phd\FishBehavior\main.py`，选择顶部视角及所需指标。主入口检查result.csv是否可用；缺失时提示先完成Tracker。已有可用frames.csv时可仅进行geometry测量。选择取消或基础任务失败时，不会继续计算行为。

维护边界：YHC负责本目录内的模型配置、Tracker、八个行为脚本、输入输出和顶部说明；修改既有指标内部算法通常不需要修改main.py。新增指标或变更启动/输入协议时，需要同步总入口配置及本文档。

已验证环境为 `D:\Anaconda\envs\sam3\python.exe`，Python 3.12，CUDA版PyTorch，RTX 4060。SAM3阶段需要支持bfloat16的NVIDIA GPU；几何与行为阶段用CPU，不加载SAM3。现有环境已能运行，不需要为了此项目重装。requirements.txt记录当前可复现的依赖版本。

在PowerShell中，路径含空格，必须用引号。先执行：

```powershell
& "D:\Anaconda\envs\sam3\python.exe" "E:\Phd\FishBehavior\Top\Tracker\sam3.py"
& "D:\Anaconda\envs\sam3\python.exe" "E:\Phd\FishBehavior\Top\Tracker\geometry.py"
```

然后独立运行需要的指标，例如：

```powershell
& "D:\Anaconda\envs\sam3\python.exe" "E:\Phd\FishBehavior\Top\Behavior\speed.py"
& "D:\Anaconda\envs\sam3\python.exe" "E:\Phd\FishBehavior\Top\Behavior\distance.py"
& "D:\Anaconda\envs\sam3\python.exe" "E:\Phd\FishBehavior\Top\Behavior\tail.py"
& "D:\Anaconda\envs\sam3\python.exe" "E:\Phd\FishBehavior\Top\Behavior\towards.py"
& "D:\Anaconda\envs\sam3\python.exe" "E:\Phd\FishBehavior\Top\Behavior\area.py"
& "D:\Anaconda\envs\sam3\python.exe" "E:\Phd\FishBehavior\Top\Behavior\resting_time.py"
& "D:\Anaconda\envs\sam3\python.exe" "E:\Phd\FishBehavior\Top\Behavior\trajectory_entanglement.py"
& "D:\Anaconda\envs\sam3\python.exe" "E:\Phd\FishBehavior\Top\Behavior\aggregation.py"
```

上述八行是独立命令，可以只选其中一行，无需先运行speed才能运行distance或resting_time。修改行为参数后只重跑相应行为脚本。调整中轴/判头参数要重跑geometry和受影响的指标；更换图片或模型参数要从分割开始。

输出使用固定目录和文件名，重复运行覆盖该模块的已知结果。没有时间戳目录、旧版本目录或综合JSON文件。新分割开始会撤下旧frames.csv/result.csv，避免误用旧几何；但不会自动重算所有行为输出，重新生成result.csv后应重跑需要的指标。关闭SAVE_VIDEO时，本模块旧视频会删除，避免新CSV搭配旧视频。

所有脚本都提供run(...)函数，直接运行时由脚本底部调用。示例（这是外部调用示例，不要求新增项目入口）：

```python
import importlib.util
import sys

path = r"E:\Phd\FishBehavior\Top\Behavior\speed.py"
spec = importlib.util.spec_from_file_location("fish_speed", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
results = module.run(
    input_csv=r"E:\Phd\FishBehavior\Top\Tracker\Output\Information\result.csv",
    output_dir=r"E:\Phd\FishBehavior\Top\Behavior\Output\speed",
    frame_step=1,
    save_video=True,
)
```

各函数准确参数见本脚本run的注释；area使用grid_rows/grid_cols/roi，tail使用window_frames，近交使用window_frames/time_tolerance_frames/distance_threshold/merge_gap_frames。Tracker/sam3.py若用importlib导入，模块别名应为top_tracker等，不能取sam3，以免遮蔽模型包。程序返回结果字典或基础CSV路径，不另外写综合JSON。此版本本地接口以CSV路径为输入；不能直接把原分割data字典当作result.csv传入。

## 3. 时间、坐标、体长与有效性

frame_id从0开始，ID=0也是合法鱼ID。x向图像右侧增加，y向下增加。位置取完整原始掩膜的质心；中轴分析才进行连通域、孔洞修补和骨架处理。Zhang–Suen细化生成骨架后取最长端点主路径，两端延伸、重采样为51点，再做空间平滑并恢复端点；结合两端宽度和时序证据确定头尾。头部轴使用中轴10%~48%，尾部轴使用65%~95%，两轴均指向头侧。手动HEAD_HINTS只用于该鱼初次成功定向。

优先解析所有图片名中有效且递增的_t秒数，减去第一张图片时间；否则整段统一使用frame_id/FPS。计算使用time_s实际时间差；视频按fps播放，不为不规则采样插入新帧。150张30fps图片首末位置跨度通常约149/30=4.967秒，不应强行计作5秒运动时间，也不为最后一帧虚构一个后续时间间隔。

每条鱼的1 BL = 本批全部有效、有限且大于零的body_length_px中位数。所有分析使用geometry写入的固定参考体长；这是分割中轴估计长度，不是标定尺测得的真实厘米长度。每鱼不同标尺，速度与距离共用该鱼标尺。没有有效体长时BL结果留空并标无效，像素距离及局部邻鱼距离仍可能有效。各脚本校验参考体长，不能把几条鱼或某一时间段裁出CSV后冒充新的完整批次。

空CSV单元代表不可测，不等于0。有效标记为0/1。应先筛选相应valid或status，再统计数值。首观测没有上一帧速度；首个有效速度区间也没有加速度。geometry_valid和head_valid失败时，position_valid仍可能为1，位置类指标可以独立判断。quality_reason有时记录成功测量的方法/警告，非空并不必然代表无效。数组字段使用单元格内JSON文本，CSV不是额外JSON文件。

## 4. 各指标的定义与参数

### 4.1 速度、距离和平均速度

speed.py和distance.py各自有FRAME_STEP=1。5表示按该鱼首观测锚点A→A+5→A+10，非滑动每帧都取5帧，也不是把窗口内所有小步长路程相加。检查窗口中所有帧的位置/面积/时间，缺帧不跨越强算；头向判断还要求窗口内头轴有效、相邻转角不超过90度。MAX_AREA_CHANGE_RATIO默认1.5。几何测量和轨迹关系分析不会因为速度步长改变而减少基础帧。

区间移动量Δp=末质心−首质心，距离d=||Δp||，速率s=d/Δt。符号由首末单位头轴的平均方向与Δp的点积决定；纯侧移取正。点积只判符号，大小仍是完整欧氏距离。鱼头朝右时右侧半平面为正，朝左时左侧半平面为正，不固定左正右负。

路径总距离=有效无符号距离之和；有符号累计距离=有效有符号距离之和。正向距离含纯侧移，倒退距离是负项的绝对值总和。有符号累计不等于首末直线位移。平均速率=对应有效总距离/有效总时间，平均有符号速度用自己的有效区间计算。零有效时间时平均量为空，不是静止。

speed只导出BL/s和BL/s²运动量；distance在同一表中保留px和BL。图中的定位坐标和参考体长仍用像素。两个独立脚本若希望同窗口比较，应分别把FRAME_STEP设为相同值；修改speed参数不会自动修改distance或静止判断。

### 4.2 加速度

先计算二维速度v=(Δx/Δt,Δy/Δt)，各速度区间的时间位置取区间中点。只有两个相邻且连续有效的速度窗口才能计算a=(v2−v1)/(中点时间差)。无符号加速度为||a||/BL；有符号加速度把该大小按a与两个窗口平均头向的点积赋正负，纯侧向仍取正。它不是对有符号速度作数值差分，也不是将速率增长率取绝对值；恒速转弯也可能有加速度，朝前减速可以产生负加速度。

加速度属于当前窗口终点帧，acceleration_time_s另外记录差分对应的物理时间位置；acceleration_frame_range覆盖前后两个速度窗口，不能只看frame_range。中间无效/缺帧会重置差分。未对质心或速度自动平滑，因此加速度会放大定位抖动；这是原始测量规则，不把峰值自动解释成生理反应。

### 4.3 摆尾频率和幅度

WINDOW_FRAMES默认150。每条鱼每帧只使用当前及过去最多150个连续有效样本，不跨缺帧。初始窗口未满时可以计算，但必须满足相应周期条件；窗口长度不足以覆盖低频周期时会明确标无效。频率主算法与原项目相同：等间隔插值、5点二阶Savitzky–Golay轻平滑、去线性趋势、检测同相位正峰、周期中位数倒数；频谱只作交叉检查。默认频段0.4~6Hz，还受每周期至少5样本限制；最小稳健幅度3度，峰突出度至少3度，至少2个完整周期，周期CV不超过0.35。数值阈值比较保留原算法的小容差。

新增amplitude_deg取各完整正峰到正峰周期内(最大去趋势角度−最小去趋势角度)/2，再取周期中位数；至少1个完整周期。peak_to_peak_deg是其两倍。robust_amplitude_deg是旧算法95/5百分位半差，只作诊断，两种“幅度”不要混用。频率不可靠但已有完整周期时，幅度可能仍可用，各自检查valid。

最后一帧窗口若覆盖与原来同一个完整有效段，frequency_hz及频率质量与原算法一致。窗口内的频率不能解释成某一瞬间只看一帧就测得的频率。周期CV是周期时长标准差/平均周期，不能与频率混为一谈。

### 4.4 朝向

tilt_deg=atan2(−head_dy,abs(head_dx))，左右水平都为0，上正下负，范围±90度。raw_heading_deg=atan2(−head_dy,head_dx)，区分左右，范围±180度。unwrapped_heading_deg只在连续有效段内展开±180度跳变。默认水平分类容差10度。朝向来自形状，不由位移猜头。

### 4.5 停留区域

默认把整张图分成2行2列，分别是左上、右上、左下、右下。ROI_BOUNDS=None表示整幅图；可在area.py设置水槽区域(x_min,y_min,x_max,y_max)，GRID_ROWS/GRID_COLS可改为3得到九宫格。区域定义属于配置，并非程序自动识别出的水槽。

每个连续有效相邻帧区间的实际dt归给起点质心所在区域（零阶保持），末帧不增加虚构时间。ROI外单独统计outside_roi，缺帧/无效位置/过长采集间隔记未知。占比=区域时长/有效总时长。静止或运动都可以贡献停留时间。entries含首次和缺失后重新可判定片段，不代表全部都是确认跨边界进入。

### 4.6 静止时间

默认REST_SPEED_THRESHOLD_BL_S=0.05，比较相邻帧无符号BL速率，严格小于阈值为静止。它自己计算，不读取speed输出，也不随speed.FRAME_STEP改变，避免长窗口往返位移抵消被误记静止。倒退速度先取无符号速率再比较，不能把所有负速度判成静止。

静止时间是有效静止区间dt之和；运动、缺帧和无效测量切断连续静止事件。未知时间=该鱼首末观察跨度−有效判断时间。rest_fraction分母为有效时间，不包含未知时间；没有有效体长或区间时留空。单个低速区间也计入，当前没有另设最短静止持续时间；阈值需按实验需求和定位噪声调整。

### 4.7 个体二维轨迹近交

默认8帧滑动窗口、每帧推进1帧；两条分段线性轨迹的到达时间差不超过2/fps秒。在满足时间约束的位置之间求最小距离，不只比较离散采样点。两鱼各自窗口bbox宽高中位数再分别取均值，得到固定归一化宽w和高h；d=sqrt((Δx/w)²+(Δy/h)²)，d≤0.5为候选。算法在每个线段对的时间约束可行多边形上求凸二次最小值，包含线段内部相交。

同一鱼对、同一连续有效段，按候选实际插值时刻排序，相邻间隔≤8/fps秒合并为一个事件；不跨缺帧合并。每个完整有效鱼对窗口贡献1个有效鱼对帧，即使没有事件也贡献分母。每鱼事件率=参与事件数/该鱼有效鱼对帧数×1000。鱼数变化或不同帧率比较时，应同时查看有效暴露和时间参数，不能仅比次数。

这是二维接近/近交，不含深度；近距离平行或静止靠近也可能满足条件。不能据此确认身体碰撞、攻击或主动追逐。无有效暴露的事件率为空；有有效暴露而无事件为有效0。长期贴近不会每个窗口重复计为独立事件。

### 4.8 个体局部聚集

每帧在原始像素质心上做Delaunay三角剖分，对每条鱼取与其直接相连的唯一邻接边平均长度。同一条边即使属于两个三角形，也只计一次。距离越小表示邻鱼越近；这是局部间距，不是全群平均三角形周长，也不是0~1聚集概率。BL列在剖分以后除以当前鱼自己的参考体长；不能先分别缩放每条鱼坐标再剖分。

少于3个有效不同位置、重复点、共线、剖分失败或遗漏点时标无效，不随机扰动数据强行造三角形。头向无效但位置有效的鱼仍可参与。Delaunay可能连接较远小群，邻鱼数不等于亲近伙伴数；鱼数、漏检和边缘位置会影响指标。

## 5. 输出图片、视频怎么看

| 输出位置 | 含义 |
|---|---|
| Tracker/Output/tracking.mp4 | SAM3彩色分割叠加、鱼ID、已处理帧数和对象数，按输入顺序播放；用来检查漏分、串ID、轮廓跳变。该视频屏显帧计数从1开始，对应CSV的frame_id从0开始；模型分数仅保存在CSV中。 |
| Behavior/Output/speed/speed_bl.png | 六子图：A无符号速率、B无符号加速度、C每鱼平均速率、D有符号速度、E有符号加速度、F每鱼平均有符号速度。C/F是柱状汇总，不伪造逐帧平均曲线。 |
| Behavior/Output/speed/speed_bl.mp4 | 原图叠加鱼ID、运动轨迹及BL单位速度/加速度；N/A表示不可用。 |
| Behavior/Output/distance/distance_px.png、distance_bl.png | 各四子图：A区间无符号距离、B累计无符号距离、C区间有符号距离、D累计有符号距离；两个图只有单位不同。 |
| Behavior/Output/tail/tail.png | 每鱼一个子图：灰色原始角、蓝色窗口平滑终点角、绿色半峰峰值幅度；橙色频率对应右侧Hz轴，左右轴单位不同。各鱼频率轴统一尺度。断线表示不可用。 |
| Behavior/Output/tail/tail.mp4 | 中轴线、头尾轴及ID、频率f、幅度A；红色头端/头轴、蓝色尾端/尾轴，角度单位度。 |
| Behavior/Output/towards/towards.png | 每鱼一个倾角曲线，水平附近灰带为±10度分类容差，上正下负。 |
| Behavior/Output/towards/towards.mp4 | 质心与指向头部的箭头、相对水平倾角，不是速度方向箭头。 |
| Behavior/Output/area/area.png | 每鱼一个空间分区图，格内为停留秒数和有效时间占比，点表示质心轨迹；顶部标未知时间。 |
| Behavior/Output/resting_time/resting_time.png | 每鱼一个静止/运动/未知时间轴，结合图例区分三种状态；未知不代表静止。 |
| Behavior/Output/trajectory_entanglement/entanglement.png | 每鱼一个累计参与事件曲线，标题显示标准化事件率；零事件的有效曲线为0。 |
| Behavior/Output/trajectory_entanglement/entanglement.mp4 | 短轨迹、事件鱼对及累计事件数；无事件仍正常输出轨迹视频。 |
| Behavior/Output/aggregation/aggregation.png | 每鱼平均邻鱼距离随时间变化，图中单位为像素；较小表示局部更紧密。按每条鱼自身参考体长换算的BL数值同时保存在CSV中。 |
| Behavior/Output/aggregation/aggregation.mp4 | 原图叠加Delaunay连线、鱼ID和局部平均邻鱼距离。 |

## 6. CSV逐列中文解释

所有CSV均为带BOM的UTF-8，用英文列名作为程序接口。以下表格逐列给出中文含义；不是把列标题改为中文。每个CSV第一行是列标题，从第二行开始是数据。数组字段在单元格内保存JSON，读取时应解析，不要按逗号手动拆整行。

### 分割中间表

路径：`E:\Phd\FishBehavior\Top\Tracker\Output\Information\frames.csv`

一行是一帧中的一条鱼。无鱼帧为empty_frame，鱼ID及掩膜为空，但帧时间和路径保留。由sam3.py生成，只供geometry.py使用。

| 英文列标题 | 中文解释 |
|---|---|
| `schema_version` | CSV格式版本，当前为1。 |
| `run_id` | 本次分割运行的唯一标识；用于避免混合不同批次。 |
| `record_type` | 记录类型：observation表示一帧中的一条鱼；empty_frame表示这一帧没有检测到鱼。 |
| `video_id` | 输入图片序列标识，当前使用图片目录绝对路径。 |
| `frame_id` | 图片自然排序后的原始帧号，从0开始；中间没有鱼的帧也保留。 |
| `time_s` | 相对第一张图片的实际测量时间，单位秒。 |
| `fps` | 标称帧率，单位帧/秒；用于播放和采样质量判断，不替代有效的采集时间戳。 |
| `time_source` | 时间轴来源：文件名_t秒数减首帧时间，或帧号除以fps。 |
| `image_path` | 当前帧原图的绝对路径，生成可视化视频时使用。 |
| `image_width` | 原图宽度，单位像素。 |
| `image_height` | 原图高度，单位像素。 |
| `track_id` | SAM3连续跟踪的鱼ID，同一ID对应同一条鱼；0为合法ID，空帧留空。 |
| `category_id` | 类别编号，当前鱼类固定为1。 |
| `score` | 模型输出的对象分数；不表示每个像素正确概率。 |
| `area_px` | 原始鱼体掩膜的前景像素数，单位像素平方。 |
| `bbox_x` | 掩膜外接框左上角x坐标，向右为正，单位像素。 |
| `bbox_y` | 掩膜外接框左上角y坐标，向下为正，单位像素。 |
| `bbox_width` | 掩膜外接框宽度，单位像素。 |
| `bbox_height` | 掩膜外接框高度，单位像素。 |
| `mask_height` | 完整掩膜高度，与原图高度相同；COCO尺寸数组中放在第一位。 |
| `mask_width` | 完整掩膜宽度，与原图宽度相同；COCO尺寸数组中放在第二位。 |
| `mask_rle` | 完整二值掩膜的COCO压缩RLE ASCII文本，按列展开；解码可无损恢复该鱼轮廓，重叠个体分别保存。 |

### 基础测量表

路径：`E:\Phd\FishBehavior\Top\Tracker\Output\Information\result.csv`

一行是一帧中的一条鱼；保留空帧。由geometry.py生成，是所有行为脚本唯一的数值输入。每个脚本读取整批数据；不要直接截取行后沿用旧体长中位数，应对所选完整输入重新生成基础表。

| 英文列标题 | 中文解释 |
|---|---|
| `schema_version` | CSV格式版本，当前为1。 |
| `run_id` | 本次分割运行的唯一标识；用于避免混合不同批次。 |
| `record_type` | 记录类型：observation表示一帧中的一条鱼；empty_frame表示这一帧没有检测到鱼。 |
| `video_id` | 输入图片序列标识，当前使用图片目录绝对路径。 |
| `frame_id` | 图片自然排序后的原始帧号，从0开始；中间没有鱼的帧也保留。 |
| `time_s` | 相对第一张图片的实际测量时间，单位秒。 |
| `fps` | 标称帧率，单位帧/秒；用于播放和采样质量判断，不替代有效的采集时间戳。 |
| `time_source` | 时间轴来源：文件名_t秒数减首帧时间，或帧号除以fps。 |
| `image_path` | 当前帧原图的绝对路径，生成可视化视频时使用。 |
| `image_width` | 原图宽度，单位像素。 |
| `image_height` | 原图高度，单位像素。 |
| `track_id` | SAM3连续跟踪的鱼ID，同一ID对应同一条鱼；0为合法ID，空帧留空。 |
| `category_id` | 类别编号，当前鱼类固定为1。 |
| `score` | 模型输出的对象分数；不表示每个像素正确概率。 |
| `cx` | 完整原始掩膜的质心x坐标，向右为正，单位像素；不使用修补后的轮廓计算质心。 |
| `cy` | 完整原始掩膜的质心y坐标，向下为正，单位像素。 |
| `area_px` | 原始鱼体掩膜的前景像素数，单位像素平方。 |
| `bbox_x` | 掩膜外接框左上角x坐标，向右为正，单位像素。 |
| `bbox_y` | 掩膜外接框左上角y坐标，向下为正，单位像素。 |
| `bbox_width` | 掩膜外接框宽度，单位像素。 |
| `bbox_height` | 掩膜外接框高度，单位像素。 |
| `centerline` | JSON坐标数组[[x,y],...]，51个点按鱼头到鱼尾排列；坐标为原图像素；无法提取时为空。 |
| `head_dx` | 头部中轴线拟合得到的指向鱼头的单位方向向量x分量。 |
| `head_dy` | 头部中轴线拟合得到的指向鱼头的单位方向向量y分量，图像y向下。 |
| `tail_dx` | 尾部中轴线拟合得到的朝头侧单位方向向量x分量；与头轴同向定义。 |
| `tail_dy` | 尾部中轴线拟合得到的朝头侧单位方向向量y分量。 |
| `heading_deg` | 完整鱼头朝向角，单位度，右0、上+90、下-90、左±180。 |
| `tail_angle_deg` | 头部轴与尾部轴的有符号夹角，单位度；直鱼约0，供摆尾分析使用。 |
| `body_length_px` | 本帧沿平滑并延伸后的中轴折线计算的弧长，单位像素；只有几何有效时可用。 |
| `reference_body_length_px` | 该鱼整批所有有效、有限且大于零的中轴长度的中位数，作为固定1BL；不同鱼各用自己的标尺；无有效体长时为空。 |
| `reference_length_sample_count` | 该鱼参与参考体长中位数计算的有效帧数；无有效体长为0。 |
| `position_valid` | 位置有效标记：1为质心可用，0为不可用。中轴或判头失败时，位置仍可能有效。 |
| `geometry_valid` | 原中轴及头尾测量综合质量标记：1为通过原算法检查，0为失败。 |
| `head_valid` | 头向有效标记：1为头部单位轴可信可用，0为不可用于判断运动正负；当前基础几何流程与geometry_valid一致。 |
| `head_confidence` | 基于宽度证据和时序延续的启发式头向评分，不是准确率概率；空帧为0。 |
| `quality_reason` | 测量质量来源、警告或失败原因。成功时也可能非空，例如width_temporal；应先看对应有效性字段。 |

### 速度与加速度明细

路径：`E:\Phd\FishBehavior\Top\Behavior\Output\speed\speed_frames.csv`

一行是某条鱼的一个固定采样窗口，区间结果归到终点帧。首观测、不足窗口、缺失终点等仍保留行和原因。默认步长1，速度单位BL/s，加速度BL/s²。

| 英文列标题 | 中文解释 |
|---|---|
| `frame_id` | 本行结果归属的终点帧号；首行仅为初次观测，不代表有效运动区间 |
| `track_id` | 鱼的跟踪 ID，0 也是合法 ID |
| `frame_step` | 本次固定采样步长，单位为帧间隔数，例如 1 或 5 |
| `anchor_frame` | 该鱼第一次观测帧，采样网格固定锚点，缺帧后不重新定锚 |
| `start_frame` | 当前区间起点帧；汇总表中为该鱼首观测帧 |
| `expected_end_frame` | 按照固定步长计划的终点帧，尾部不足时可晚于当前实际终点 |
| `frame_range` | 当前速度/距离窗口中实际存在的观测帧列表（JSON）；汇总表为参与窗口的观测帧并集 |
| `window_complete` | 当前区间是否达到指定步长且没有缺帧，1 是、0 否；不代替其他质量判定 |
| `missing_frame_count` | 当前窗口缺失的鱼观测帧总数 |
| `missing_frame_ranges` | 缺失帧的闭区间段列表（JSON），如 [[3,5]] 表示缺少 3、4、5 帧 |
| `start_time_s` | 当前窗口起点的实际相对采集时间，秒 |
| `time_s` | 当前窗口终点的实际相对采集时间，秒 |
| `dt_s` | 窗口实际时间差，秒；即使有值也必须检查有效性 |
| `cx` | 终点鱼体质心横坐标，像素，向右为正 |
| `cy` | 终点鱼体质心纵坐标，像素，向下为正 |
| `heading_deg` | 终点鱼头完整朝向角，度，右 0、上 +90、下 -90 |
| `reference_body_length_px` | 该鱼整批有效中轴弧长中位数，像素，所有窗口共用的 1 BL |
| `reference_length_sample_count` | 计算该鱼参考体长的有效样本数 |
| `body_length_valid` | 参考体长是否有效，1 是、0 否 |
| `body_length_reason` | 体长标尺状态：ok 或 no_valid_body_length |
| `speed_bl_s` | 窗口无符号速率，完整欧氏位移 / 时间 / 本鱼参考体长，BL/s |
| `signed_speed_bl_s` | 沿平均鱼头方向判正负的完整移动速率，BL/s，纯侧移取正 |
| `velocity_x_bl_s` | 图像水平速度分量，BL/s，向右为正 |
| `velocity_y_bl_s` | 图像垂直速度分量，BL/s，向下为正 |
| `bl_distance_valid` | 当前无符号 BL 运动量是否有效；speed 表中也控制无符号速率及二维速度分量 |
| `bl_direction_valid` | 当前有符号 BL 运动量是否有效，额外要求鱼头方向可靠 |
| `quality_reason` | 当前速度/距离区间的质量原因，成功为空；首观测、缺帧、头向失效等有原因代码 |
| `quality_detail` | 质量原因的详细说明，例如具体缺失帧或跳变帧 |
| `acceleration_x_bl_s2` | 相邻连续速度窗口的水平速度分量变化率，BL/s²，右为正 |
| `acceleration_y_bl_s2` | 相邻连续速度窗口的垂直速度分量变化率，BL/s²，下为正 |
| `acceleration_bl_s2` | 二维加速度向量的大小，BL/s²，非负 |
| `signed_acceleration_bl_s2` | 二维加速度大小按平均鱼头方向赋正负，BL/s²，纯侧向取正；不是有符号速度的标量差分 |
| `acceleration_time_s` | 两次速度窗口中点时间的平均值，加速度对应的时间位置，秒 |
| `acceleration_dt_s` | 两次速度窗口中点的时间差，即加速度差分分母，秒 |
| `acceleration_frame_range` | 计算加速度所需两个速度窗口的全部观测帧并集（JSON） |
| `acceleration_valid` | 无符号 BL 加速度是否有效，1 是、0 否 |
| `signed_acceleration_valid` | 有符号 BL 加速度是否有效，还要求两个窗口头向可靠且平均方向可定义 |
| `acceleration_reason` | 无符号加速度质量状态：ok、帧不足、缺口、无体长等 |
| `signed_acceleration_reason` | 有符号加速度质量状态，还可能包含头轴无效或平均头轴抵消 |

### 每鱼平均速度

路径：`E:\Phd\FishBehavior\Top\Behavior\Output\speed\speed_summary.csv`

一行一条鱼。平均速率和平均有符号速度分别使用各自有效的距离总和/时间总和，不能把所有区间的速度简单算术平均。

| 英文列标题 | 中文解释 |
|---|---|
| `track_id` | 鱼的跟踪 ID，0 也是合法 ID |
| `frame_step` | 本次固定采样步长，单位为帧间隔数，例如 1 或 5 |
| `start_frame` | 当前区间起点帧；汇总表中为该鱼首观测帧 |
| `end_frame` | 汇总范围内该鱼末观测帧 |
| `observations` | 该鱼实际观测帧数，不包含虚构的缺帧观测 |
| `track_span_s` | 该鱼首末观测时间跨度，秒；不向视频首尾外推 |
| `valid_distance_intervals` | 坐标和时间等质量有效的无符号区间数量；BL 可用性还需检查体长标记 |
| `valid_signed_intervals` | 同时具备可靠头向的有符号区间数量；BL 可用性还需检查体长标记 |
| `valid_distance_time_s` | 无符号有效坐标区间的时间差总和，秒 |
| `valid_signed_time_s` | 有符号有效区间的时间差总和，秒 |
| `distance_time_coverage` | 无符号有效区间总时长 / 首末观测跨度，0~1；零跨度为空 |
| `signed_time_coverage` | 有符号有效区间总时长 / 首末观测跨度，0~1；零跨度为空 |
| `reference_body_length_px` | 该鱼整批有效中轴弧长中位数，像素，所有窗口共用的 1 BL |
| `reference_length_sample_count` | 计算该鱼参考体长的有效样本数 |
| `body_length_valid` | 参考体长是否有效，1 是、0 否 |
| `body_length_reason` | 体长标尺状态：ok 或 no_valid_body_length |
| `mean_speed_bl_s` | 有效无符号总距离 / 对应有效总时间 / 本鱼体长，BL/s，不是逐区间速率简单平均 |
| `mean_signed_speed_bl_s` | 有效有符号总距离 / 对应有效总时间 / 本鱼体长，BL/s |
| `bl_distance_summary_valid` | 无符号 BL 汇总是否有效，至少有一个有效区间且体长有效 |
| `bl_direction_summary_valid` | 有符号 BL 汇总是否有效，至少有一个头向有效区间且体长有效 |

### 距离明细

路径：`E:\Phd\FishBehavior\Top\Behavior\Output\distance\distance_frames.csv`

一行一条鱼的一个采样窗口，同时含像素和BL列。累计量只增加相应有效区间；无效当前行可能仍保留之前累计值，应结合有效性读取。

| 英文列标题 | 中文解释 |
|---|---|
| `frame_id` | 本行结果归属的终点帧号；首行仅为初次观测，不代表有效运动区间 |
| `track_id` | 鱼的跟踪 ID，0 也是合法 ID |
| `frame_step` | 本次固定采样步长，单位为帧间隔数，例如 1 或 5 |
| `anchor_frame` | 该鱼第一次观测帧，采样网格固定锚点，缺帧后不重新定锚 |
| `start_frame` | 当前区间起点帧；汇总表中为该鱼首观测帧 |
| `expected_end_frame` | 按照固定步长计划的终点帧，尾部不足时可晚于当前实际终点 |
| `frame_range` | 当前速度/距离窗口中实际存在的观测帧列表（JSON）；汇总表为参与窗口的观测帧并集 |
| `window_complete` | 当前区间是否达到指定步长且没有缺帧，1 是、0 否；不代替其他质量判定 |
| `missing_frame_count` | 当前窗口缺失的鱼观测帧总数 |
| `missing_frame_ranges` | 缺失帧的闭区间段列表（JSON），如 [[3,5]] 表示缺少 3、4、5 帧 |
| `start_time_s` | 当前窗口起点的实际相对采集时间，秒 |
| `time_s` | 当前窗口终点的实际相对采集时间，秒 |
| `dt_s` | 窗口实际时间差，秒；即使有值也必须检查有效性 |
| `cx` | 终点鱼体质心横坐标，像素，向右为正 |
| `cy` | 终点鱼体质心纵坐标，像素，向下为正 |
| `heading_deg` | 终点鱼头完整朝向角，度，右 0、上 +90、下 -90 |
| `reference_body_length_px` | 该鱼整批有效中轴弧长中位数，像素，所有窗口共用的 1 BL |
| `reference_length_sample_count` | 计算该鱼参考体长的有效样本数 |
| `body_length_valid` | 参考体长是否有效，1 是、0 否 |
| `body_length_reason` | 体长标尺状态：ok 或 no_valid_body_length |
| `dx_px` | 窗口首末质心横向位移，像素，向右为正 |
| `dy_image_px` | 窗口首末质心纵向位移，像素，向下为正 |
| `step_distance_px` | 窗口首末质心完整欧氏距离，像素；不是窗口内小步距离之和 |
| `signed_distance_px` | 完整欧氏距离按平均鱼头方向赋号，像素，纯侧移取正 |
| `forward_projection_px` | 位移在平均鱼头轴上的投影，像素，仅作方向诊断，不是有符号距离的大小 |
| `lateral_distance_px` | 位移在鱼头轴垂直方向的有符号投影，像素；鱼朝右时向上为正 |
| `direction_sign` | 有符号距离所用正负号，+1 或 -1；方向无效为空 |
| `cumulative_path_distance_px` | 截至当前窗口，所有有效无符号端点距离的累计值，像素 |
| `cumulative_signed_distance_px` | 截至当前窗口，所有有效有符号距离的代数和，像素 |
| `dx_bl` | 横向位移换算为该鱼体长，BL，向右为正 |
| `dy_image_bl` | 纵向位移换算为该鱼体长，BL，向下为正 |
| `step_distance_bl` | 窗口完整欧氏距离 / 本鱼参考体长，BL |
| `signed_distance_bl` | 有符号距离 / 本鱼参考体长，BL；汇总表中表示全段有效有符号累计 |
| `forward_projection_bl` | 鱼头轴方向投影 / 本鱼体长，BL，仅作诊断 |
| `lateral_distance_bl` | 侧向投影 / 本鱼体长，BL，仅作诊断 |
| `cumulative_path_distance_bl` | 累计有效无符号端点距离换算为 BL |
| `cumulative_signed_distance_bl` | 累计有效有符号距离换算为 BL |
| `distance_valid` | 无符号像素运动量是否通过位置、面积、时间及完整窗口检查 |
| `direction_valid` | 有符号像素运动量是否额外通过头向检查 |
| `bl_distance_valid` | 当前无符号 BL 运动量是否有效；speed 表中也控制无符号速率及二维速度分量 |
| `bl_direction_valid` | 当前有符号 BL 运动量是否有效，额外要求鱼头方向可靠 |
| `quality_reason` | 当前速度/距离区间的质量原因，成功为空；首观测、缺帧、头向失效等有原因代码 |
| `quality_detail` | 质量原因的详细说明，例如具体缺失帧或跳变帧 |

### 每鱼总距离

路径：`E:\Phd\FishBehavior\Top\Behavior\Output\distance\distance_summary.csv`

一行一条鱼，包括无符号路径距离、有符号累计距离、正向距离和倒退距离绝对值。它们不是首末位置直线距离，也不互相等价。

| 英文列标题 | 中文解释 |
|---|---|
| `track_id` | 鱼的跟踪 ID，0 也是合法 ID |
| `frame_step` | 本次固定采样步长，单位为帧间隔数，例如 1 或 5 |
| `anchor_frame` | 该鱼第一次观测帧，采样网格固定锚点，缺帧后不重新定锚 |
| `start_frame` | 当前区间起点帧；汇总表中为该鱼首观测帧 |
| `end_frame` | 汇总范围内该鱼末观测帧 |
| `frame_range` | 当前速度/距离窗口中实际存在的观测帧列表（JSON）；汇总表为参与窗口的观测帧并集 |
| `observations` | 该鱼实际观测帧数，不包含虚构的缺帧观测 |
| `sample_rows` | 输出的采样行数，包含首行、无效窗口及尾部不足行 |
| `candidate_intervals` | 候选区间数（采样行数减首行），包含无效和不足区间 |
| `incomplete_intervals` | 尾部未达到完整步长的区间数量 |
| `track_span_s` | 该鱼首末观测时间跨度，秒；不向视频首尾外推 |
| `valid_distance_intervals` | 坐标和时间等质量有效的无符号区间数量；BL 可用性还需检查体长标记 |
| `valid_signed_intervals` | 同时具备可靠头向的有符号区间数量；BL 可用性还需检查体长标记 |
| `valid_distance_time_s` | 无符号有效坐标区间的时间差总和，秒 |
| `valid_signed_time_s` | 有符号有效区间的时间差总和，秒 |
| `distance_time_coverage` | 无符号有效区间总时长 / 首末观测跨度，0~1；零跨度为空 |
| `signed_time_coverage` | 有符号有效区间总时长 / 首末观测跨度，0~1；零跨度为空 |
| `path_distance_px` | 整段有效无符号采样端点距离之和，像素 |
| `signed_distance_px` | 整段所有有效有符号距离之代数和，像素；可被倒退抵消，不等同首末净位移 |
| `forward_distance_px` | 所有有效正向距离之和，像素，包含按约定计正的纯侧移 |
| `backward_distance_px` | 所有有效负向距离的绝对值之和，像素，非负 |
| `path_distance_bl` | 整段无符号总距离换算为 BL |
| `signed_distance_bl` | 有符号距离 / 本鱼参考体长，BL；汇总表中表示全段有效有符号累计 |
| `forward_distance_bl` | 整段正向距离总和换算为 BL，包含纯侧移 |
| `backward_distance_bl` | 整段负向距离绝对值总和换算为 BL，非负 |
| `reference_body_length_px` | 该鱼整批有效中轴弧长中位数，像素，所有窗口共用的 1 BL |
| `reference_length_sample_count` | 计算该鱼参考体长的有效样本数 |
| `body_length_valid` | 参考体长是否有效，1 是、0 否 |
| `body_length_reason` | 体长标尺状态：ok 或 no_valid_body_length |
| `distance_summary_valid` | 无符号像素汇总是否有至少一个有效区间；否则累计0不表示有效静止 |
| `direction_summary_valid` | 有符号像素汇总是否有至少一个有效头向区间 |
| `bl_distance_summary_valid` | 无符号 BL 汇总是否有效，至少有一个有效区间且体长有效 |
| `bl_direction_summary_valid` | 有符号 BL 汇总是否有效，至少有一个头向有效区间且体长有效 |

### 摆尾逐帧与窗口结果

路径：`E:\Phd\FishBehavior\Top\Behavior\Output\tail\tail_frames.csv`

一行一条鱼的一帧。角度属于单帧；频率、幅度属于以当前帧结束的回看窗口。它们不代表仅凭当前一张图测得的瞬时频率。

| 英文列标题 | 中文解释 |
|---|---|
| `frame_id` | 当前结果所属的原始帧号，窗口终点；从0开始。 |
| `track_id` | 鱼的跟踪ID，0是合法ID。 |
| `time_s` | 当前帧相对第一帧的实际时间，秒。 |
| `window_start_frame` | 本次实际参与计算的连续窗口起点；不是整段视频起点。 |
| `window_end_frame` | 实际窗口终点，即当前帧。 |
| `window_samples` | 窗口中实际有效几何样本数；无效当前帧为0。 |
| `window_complete` | 是否达到配置的最大回看样本数；1是、0否。未满窗口也可能已有足够周期，因此它不替代频率有效性。 |
| `frame_range` | 本次实际使用的全部帧号JSON数组。 |
| `segment_id` | 该鱼连续有效几何段编号；缺帧、时间断点或无效几何会重新分段。 |
| `tail_angle_deg` | 当前帧头轴和尾轴的有符号夹角，度；几何无效时为空。 |
| `smoothed_angle_deg` | 在当前窗口内插值并作5点平滑后的终点角度，度；不是整段双向平滑的终点值。 |
| `frequency_hz` | 同相位正峰周期中位数的倒数，Hz；只在frequency_valid=1时有值。 |
| `amplitude_deg` | 完整同相位周期的半峰峰值中位数，度；去趋势后计算；至少一个完整周期。 |
| `peak_to_peak_deg` | 幅度的两倍，即完整周期峰峰值中位数，度。 |
| `robust_amplitude_deg` | 旧频率算法的去趋势波形95/5百分位半差，度；用于质量诊断，不是本次定义的周期摆尾幅度。 |
| `cycles` | 窗口中检出的完整同相位峰间周期数量；频率至少需要2个。 |
| `period_cv` | 周期时长总体标准差除以平均周期，无量纲；越大说明周期越不规则。 |
| `spectral_peak_hz` | 频谱最大功率峰的频率，Hz；仅用于与峰间周期估计交叉检查。 |
| `geometry_valid` | 当前帧中轴及头尾测量是否有效，1是、0否。 |
| `frequency_valid` | 频率是否通过全部质量检查，1是、0否。 |
| `amplitude_valid` | 周期幅度是否可测，1是、0否；可以与频率有效性不同。 |
| `frequency_quality` | 频率质量状态，ok为可用，其他值说明帧数、周期数、幅度、频段或规则性问题。 |
| `amplitude_quality` | 幅度质量状态：ok、invalid_geometry或insufficient_complete_cycle。 |

### 朝向逐帧结果

路径：`E:\Phd\FishBehavior\Top\Behavior\Output\towards\towards_frames.csv`

一行一条鱼的一帧，包含相对水平倾角、完整头朝向和连续段展开角；三个角的定义不同。

| 英文列标题 | 中文解释 |
|---|---|
| `frame_id` | 原始帧号。 |
| `track_id` | 鱼ID。 |
| `time_s` | 当前帧相对第一帧时间，秒。 |
| `cx` | 原始掩膜质心x，像素，向右正。 |
| `cy` | 原始掩膜质心y，像素，向下正。 |
| `tilt_deg` | 相对水平线的倾角，范围[-90,90]度；左右水平均为0，向上正、向下负。 |
| `raw_heading_deg` | 完整头朝向，右0、上+90、下-90、左±180度，保留左右象限。 |
| `unwrapped_heading_deg` | 连续有效段内展开后的完整头朝向，用于跨±180度时连续显示；可超出±180度。 |
| `segment_id` | 连续有效朝向段编号；缺帧和无效方向切断，不跨段展开角度。 |
| `direction` | 朝向分类：up、down、horizontal或invalid；默认距水平±10度内算horizontal。 |
| `head_confidence` | 基础测量中的启发式头向评分，不是准确率概率。 |
| `valid` | 朝向是否有效，1是、0否。 |
| `quality_reason` | 基础几何质量来源或异常原因，成功时也可能非空；应先看valid。 |

### 停留区域汇总

路径：`E:\Phd\FishBehavior\Top\Behavior\Output\area\area_summary.csv`

一行一条鱼的一个区域；默认每鱼4格加1个ROI外区域，共5行。未到访区域可为有效0；无法形成有效时间区间的鱼则测量值为空。

| 英文列标题 | 中文解释 |
|---|---|
| `track_id` | 鱼ID。 |
| `region_id` | 区域编号，从0开始，逐行由左到右；-1表示ROI外。 |
| `region_name` | 区域名称：默认top_left、top_right、bottom_left、bottom_right和outside_roi。 |
| `grid_row` | 网格行索引，从0开始；ROI外为空。 |
| `grid_col` | 网格列索引，从0开始；ROI外为空。 |
| `x_min` | 区域左边界，原图像素，包含边界。 |
| `y_min` | 区域上边界，原图像素，包含边界。 |
| `x_max` | 区域右边界，原图像素，不包含边界。 |
| `y_max` | 区域下边界，原图像素，不包含边界。 |
| `start_frame` | 该鱼首观测帧。 |
| `end_frame` | 该鱼末观测帧。 |
| `observed_frames` | 该鱼实际观测帧数。 |
| `valid_intervals` | 该鱼可判断区域的有效相邻帧区间总数。 |
| `region_intervals` | 归属本区域的有效时间区间数。 |
| `entries` | 本区域连续可判定占用片段的开始次数，包含首次观测/缺失后重新观测；不等同于已确认的跨边界次数。 |
| `dwell_time_s` | 起点质心位于本区域的有效相邻帧时间差之和，秒；无有效统计时为空。 |
| `occupancy_fraction` | 本区域停留时间除以该鱼全部有效区域判断时间，0~1；不是百分数，乘100后为百分比。 |
| `track_span_s` | 该鱼首末观测时间差，秒；不外推至视频起止边界。 |
| `valid_time_s` | 所有可判定区域的时间之和，含ROI外，秒。 |
| `unknown_time_s` | 观察跨度减去有效区域判断时间，秒；缺帧等归入未知。 |
| `coverage` | 有效区域判断时间/观察跨度；零跨度为空。 |
| `summary_valid` | 是否至少存在一个有效时间区间，1是、0否。 |
| `quality_reason` | 正常为空；无有效区间为no_valid_intervals。 |

### 每鱼静止时间

路径：`E:\Phd\FishBehavior\Top\Behavior\Output\resting_time\resting_time_summary.csv`

一行一条鱼，包含静止、运动和未知时间。rest_episodes列是连续静止事件的JSON数组，每个事件包含start_frame、end_frame、start_time_s、end_time_s、duration_s。

| 英文列标题 | 中文解释 |
|---|---|
| `track_id` | 鱼的跟踪 ID，0 也是合法 ID |
| `start_frame` | 当前区间起点帧；汇总表中为该鱼首观测帧 |
| `end_frame` | 汇总范围内该鱼末观测帧 |
| `observations` | 该鱼实际观测帧数，不包含虚构的缺帧观测 |
| `track_span_s` | 该鱼首末观测时间跨度，秒；不向视频首尾外推 |
| `threshold_bl_s` | 静止速率阈值，BL/s；速率严格小于此值才记静止 |
| `reference_body_length_px` | 该鱼整批有效中轴弧长中位数，像素，所有窗口共用的 1 BL |
| `reference_length_sample_count` | 计算该鱼参考体长的有效样本数 |
| `body_length_valid` | 参考体长是否有效，1 是、0 否 |
| `body_length_reason` | 体长标尺状态：ok 或 no_valid_body_length |
| `valid_intervals` | 可以判断静止或运动的相邻帧区间数量 |
| `rest_intervals` | 无符号 BL 速率低于阈值的有效相邻帧区间数量 |
| `moving_intervals` | 无符号 BL 速率达到或超过阈值的有效相邻帧区间数量 |
| `valid_time_s` | 可判断静止或运动的区间总时长，秒 |
| `rest_time_s` | 所有有效静止区间实际时间差之和，秒；无有效判断时为空 |
| `moving_time_s` | 所有有效运动区间实际时间差之和，秒；无有效判断时为空 |
| `unknown_time_s` | 首末观测跨度内无法判断的时长，秒，缺帧等归入这里 |
| `valid_time_coverage` | 可判断总时长 / 首末观测跨度，0~1，零跨度为空 |
| `rest_fraction` | 静止时长 / 可判断总时长，0~1；不是除以包含未知段的全部跨度 |
| `rest_event_count` | 连续静止事件的次数，运动或缺口会切断事件；没有有效判断时为空 |
| `longest_rest_s` | 最长连续静止事件的持续秒数；有效但无静止为0，无有效判断为空 |
| `rest_episodes` | 静止事件 JSON 数组，每项包含起止帧、起止时间和 duration_s，均不跨缺帧 |
| `summary_valid` | 是否至少存在一个可判断静止的有效区间，1 是、0 否 |
| `quality_reason` | 静止统计状态：ok / no_valid_body_length / no_valid_adjacent_intervals |

### 近交事件依据

路径：`E:\Phd\FishBehavior\Top\Behavior\Output\trajectory_entanglement\entanglement_events.csv`

一行一个去重后的鱼对事件，保存两个ID及插值时刻、位置、完整证据帧。没有事件时只有表头，这是正常的可复核输出。

| 英文列标题 | 中文解释 |
|---|---|
| `event_id` | 本次运行去重后的事件编号，从1开始 |
| `track_id_i` | 鱼对中较小的鱼 ID |
| `track_id_j` | 鱼对中较大的鱼 ID |
| `continuity_segment` | 该鱼对连续可判断窗口段编号；缺帧/无效窗口会切段 |
| `start_time_s` | 合并事件中候选最近位置的最早到达时间（秒） |
| `end_time_s` | 最晚到达时间（秒） |
| `representative_frame` | 与代表事件时刻最近的源帧 |
| `representative_time_s` | 最小归一化距离候选的两条鱼到达时间均值 |
| `fish_i_time_s` | 代表候选中鱼 i 到达最近位置的插值时间 |
| `fish_j_time_s` | 鱼 j 对应的插值时间 |
| `arrival_time_difference_s` | 两条鱼到达时间的绝对差（秒） |
| `min_distance_normalized` | 满足时间约束的最小尺寸归一化二维距离，无单位 |
| `normalization_width_px` | 两鱼窗口外接框宽度中位数的均值（像素） |
| `normalization_height_px` | 两鱼高度中位数的均值（像素） |
| `fish_i_x` | 代表候选中鱼 i 插值横坐标（像素） |
| `fish_i_y` | 鱼 i 插值纵坐标（像素） |
| `fish_j_x` | 代表候选中鱼 j 插值横坐标（像素） |
| `fish_j_y` | 鱼 j 插值纵坐标（像素） |
| `candidate_windows` | 合并进本事件的滑动窗口候选数量 |
| `evidence_start_frame` | 参与判断的最早窗口起帧 |
| `evidence_end_frame` | 最晚窗口末帧 |
| `frame_range` | 所有候选窗口实际参与的源帧 ID，JSON 数组 |
| `window_frames` | 滑动窗口样本帧数（8帧有7个时间间隔） |
| `time_tolerance_s` | 允许两鱼到达最近位置的最大时间差 |
| `distance_threshold` | 归一化近交距离阈值 |
| `merge_gap_s` | 同一鱼对候选事件时间的最大合并间隔 |

### 每鱼轨迹近交

路径：`E:\Phd\FishBehavior\Top\Behavior\Output\trajectory_entanglement\entanglement_summary.csv`

一行一条鱼。一次鱼对事件同时计入两个参与者，因此所有鱼的event_count之和为鱼对事件总数的2倍；有有效暴露但无事件时事件率为有效0。

| 英文列标题 | 中文解释 |
|---|---|
| `track_id` | 鱼 ID |
| `start_frame` | 首次观测帧 |
| `end_frame` | 最后观测帧 |
| `observed_frames` | 该鱼实际观测行数 |
| `valid_position_frames` | 位置有效的实际观测数 |
| `valid_observation_time_s` | 相邻连续有效位置之间的实际时间之和（秒） |
| `eligible_window_frames` | 该鱼至少有一个可比较伙伴的完整窗口末帧数 |
| `valid_pair_frames` | 各完整窗口内可比较伙伴数量累计，作为事件率分母 |
| `valid_pair_time_s` | 按窗口末端真实帧间隔加权的有效伙伴时间（鱼对秒） |
| `event_count` | 该鱼参与的去重事件数；同一事件给两条鱼各计一次 |
| `partner_count` | 至少发生一次事件的不同伙伴数 |
| `partner_ids` | 这些伙伴的 ID 列表 |
| `events_per_1000_pair_frames` | event_count/valid_pair_frames*1000；无有效暴露时留空 |
| `window_frames` | 滑动窗口样本帧数 |
| `time_tolerance_s` | 最近位置最大到达时间差（秒） |
| `distance_threshold` | 归一化近交阈值 |
| `merge_gap_s` | 候选合并最大时间间隔（秒） |
| `status` | ok、insufficient_frames 或 invalid |
| `reason` | 不足或无效的原因，正常为空 |

### 个体局部聚集明细

路径：`E:\Phd\FishBehavior\Top\Behavior\Output\aggregation\aggregation_frames.csv`

一行一条鱼的一帧。使用Delaunay三角剖分的唯一直接邻接边平均距离；越小表示周围邻鱼越近，不是0~1聚集概率。

| 英文列标题 | 中文解释 |
|---|---|
| `frame_id` | 原始帧号 |
| `time_s` | 相对第一张图片的时间（秒） |
| `track_id` | 鱼 ID，0 合法 |
| `cx` | 当前鱼质心横坐标（像素） |
| `cy` | 当前鱼质心纵坐标（像素） |
| `valid_fish_count` | 本帧位置有效的鱼数 |
| `triangle_count` | 本帧 Delaunay 三角形数量 |
| `neighbor_count` | 与当前鱼直接相连的不同邻鱼数量 |
| `neighbor_ids` | 相邻鱼 ID 的 JSON 列表 |
| `mean_neighbor_distance_px` | 当前鱼到所有直接邻鱼的平均欧氏距离（像素）；越小越集中 |
| `mean_neighbor_distance_bl` | 平均邻鱼距离除以当前鱼自己的固定参考体长（BL） |
| `reference_body_length_px` | 当前鱼整批有效中轴弧长的中位数（像素） |
| `reference_length_sample_count` | 参考体长的有效样本数 |
| `valid` | 二维邻鱼距离是否有效，1 是、0 否 |
| `bl_valid` | BL 邻鱼距离是否有效 |
| `quality_reason` | 像素结果不可用原因；正常为空 |
| `bl_quality_reason` | BL 结果不可用原因；正常为空 |

### 每鱼局部聚集汇总

路径：`E:\Phd\FishBehavior\Top\Behavior\Output\aggregation\aggregation_summary.csv`

一行一条鱼。均值/中位数/标准差按有效帧统计，均值为帧算术平均，并非时间加权；BL列按当前鱼固定参考体长换算。

| 英文列标题 | 中文解释 |
|---|---|
| `track_id` | 鱼 ID |
| `start_frame` | 首次观测帧 |
| `end_frame` | 最后观测帧 |
| `observed_frames` | 实际观测行数，包括位置无效行 |
| `missing_frames` | 首末观测之间清单中没有该鱼的帧数 |
| `valid_frames` | 邻鱼距离有效的帧数 |
| `invalid_frames` | 实际观测中无效的帧数 |
| `valid_fraction` | 有效帧数除以实际观测帧数 |
| `valid_time_s` | 相邻连续有效观测之间累计的真实时间（秒） |
| `mean_neighbor_distance_px` | 有效帧邻鱼距离的算术平均（像素，非按时间加权） |
| `median_neighbor_distance_px` | 有效帧邻鱼距离中位数（像素） |
| `std_neighbor_distance_px` | 有效帧邻鱼距离的总体标准差（像素） |
| `min_neighbor_distance_px` | 有效帧中最小平均邻鱼距离（像素） |
| `max_neighbor_distance_px` | 有效帧中最大平均邻鱼距离（像素） |
| `mean_neighbor_distance_bl` | 每鱼固定体长换算后的有效帧平均邻鱼距离（BL） |
| `median_neighbor_distance_bl` | 有效帧邻鱼距离中位数（BL） |
| `std_neighbor_distance_bl` | 有效帧邻鱼距离总体标准差（BL） |
| `mean_neighbor_count` | 有效帧中直接邻鱼数量的算术平均 |
| `reference_body_length_px` | 当前鱼固定参考体长（像素） |
| `reference_length_sample_count` | 参考体长样本数 |
| `status` | ok 或 invalid；无有效帧时均值留空 |
| `reason` | 无有效统计的原因 |
| `bl_status` | BL 汇总是否有效 |
| `bl_reason` | BL 汇总无效原因 |

## 7. 常见状态和读数问题

- `first_observation`：首观测没有前一段运动；不是有效的零速度。
- `incomplete_window`：末尾不足配置的帧步长，不能假装完整窗口。
- `frame_gap`、`invalid_time_interval`：缺帧或时间异常，不跨越插值补造行为。
- `head_axis_invalid`：无符号移动可能可用，有符号移动不可用。
- `no_valid_body_length`：没有可用BL标尺，不能借用其他鱼的长度。
- `needs_two_contiguous_velocity_windows`：加速度需要两段连续速度，单段不够。
- `too_short`、`insufficient_complete_cycles`：摆尾样本/完整周期不足。
- `amplitude_too_small`：波动幅度不足以可靠估频。
- `irregular_cycles`：周期CV过大；不把非周期运动强行报告成稳定频率。
- `frequency_above_safe_limit`、`period_outside_frequency_range`、`spectrum_peak_disagrees`：频率超采样安全范围或两种估计不一致。
- `fewer_than_three_valid_fish`、`duplicate_centroids`、`collinear_centroids`：当前帧不能可靠构造二维邻接网。
- 无近交事件但summary状态为ok：代表阈值下确实未检测到，不是代码没有运行。

视频是定位与质量检查工具，CSV是可复核数值依据。模型掩膜抖动和串ID会影响位置差分、加速度及鱼间关系；视频中应结合轮廓/ID观察。体长、位置、朝向均来自二维投影，不代表已获得水深或真实三维接触信息。

随项目提供的顶部结果对应150张输入图片、6条鱼、900条个体观测。算法与已验证的顶部实现保持一致；迁移仅更新项目路径和维护说明，数值列保持原值。指标计算定义和默认阈值详见前文，实验解释仍应结合原图及视频检查。

默认近交阈值下本次鱼对事件数为0，每条鱼有效鱼对帧为715（143个完整8帧窗口×5个伙伴）；事件表只有列标题，汇总中的事件率为有效0。修改阈值应依据实验定义和标记视频，不能为了得到非零结果而调参。
