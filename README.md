# FishBehavior：顶部与侧面鱼类行为分析

项目位置：`E:\Phd\FishBehavior`。这是独立的合并项目，两个原项目没有被替换。顶部维护负责人 **YHC**，侧面维护负责人 **chenxi**。

## 1. 整体关系

本项目由两个独立子项目和一个统一入口组成。main.py只显示菜单、检查所选任务需要的数据，并启动对应脚本；不包含任何鱼类行为公式。Top与Side不相互导入，也不读取另一视角的输出。两个视角的鱼ID没有自动对应关系。

模型权重和模型源码分别保留在两个子项目内，避免修改某一套模型代码时影响另一方。运行时只有所选脚本被启动，不同时加载两套模型。当前共用同一个已验证Python环境；后续修改环境依赖时，双方仍需要共同验证。

```text
FishBehavior/                                  总项目，统一调度两个独立视角
├─ main.py                                     选择视角/指标，检查基础表并按需启动前置准备
├─ README.md                                   总项目结构、执行方式及维护边界
│
├─ Top/                                        顶部子项目，负责人YHC
│  ├─ README.md                                顶部公式、输出解读和全部CSV中文列说明
│  ├─ requirements.txt                         顶部已验证的依赖清单
│  ├─ Model/                                   顶部独立的模型文件
│  │  ├─ sam3.pt                               SAM3预训练权重
│  │  ├─ bpe_simple_vocab_16e6.txt.gz            文本提示分词词表
│  │  └─ vendor/                               SAM3完整随附源码及许可证，内部脚本见第8节
│  ├─ Tracker/                                 顶部图片分割、跟踪及基础几何测量
│  │  ├─ sam3.py                               分割图片并连续跟踪ID，输出完整RLE掩膜和叠加视频
│  │  ├─ geometry.py                           计算质心、中轴线、头尾轴、头向、摆尾角、中轴长度、每鱼参考BL和有效性
│  │  ├─ Input/                                顶部输入资料
│  │  │  └─ Images/                            当前150张原始抽帧图片，按自然顺序处理
│  │  └─ Output/                               顶部数据准备结果
│  │     ├─ tracking.mp4                        彩色鱼体分割与跟踪ID叠加视频
│  │     └─ Information/                       顶部两个阶段间的CSV接口
│  │        ├─ frames.csv                       每帧每鱼的时间、ID、完整RLE掩膜及分割元信息
│  │        └─ result.csv                       每帧每鱼的基础几何量；所有顶部行为脚本读取此表
│  └─ Behavior/                                顶部独立行为指标，各脚本不互相调用
│     ├─ speed.py                              计算BL/s速率、有符号速度、平均速度及二维加速度
│     ├─ distance.py                           计算区间及累计距离，保留像素和每鱼BL两种单位
│     ├─ tail.py                               根据头尾夹角时序计算摆尾频率、完整周期半峰峰值幅度和质量
│     ├─ towards.py                            根据头部轴计算上正下负倾角及方向分类
│     ├─ area.py                               按图像分区统计每鱼停留时间、占比和进入次数
│     ├─ resting_time.py                       根据邻帧无符号BL速率统计静止、运动、未知时间和静止事件
│     ├─ trajectory_entanglement.py            计算每鱼二维轨迹近交事件、去重事件数和暴露标准化事件率
│     ├─ aggregation.py                        用Delaunay邻接边计算每鱼局部邻鱼平均距离及汇总
│     └─ Output/                               顶部指标固定输出位置，重复运行覆盖对应模块结果
│        ├─ speed/                             速度/加速度明细、汇总、图表和视频
│        ├─ distance/                          距离明细、汇总及像素/BL两张图
│        ├─ tail/                              逐帧角度、窗口频率/幅度、图表和视频
│        ├─ towards/                           朝向明细、曲线和视频
│        ├─ area/                              分区停留汇总及空间图
│        ├─ resting_time/                      静止汇总及静止/运动/未知时间轴
│        ├─ trajectory_entanglement/           近交事件、个体汇总、曲线和视频
│        └─ aggregation/                       聚集间距逐帧表、个体汇总、曲线和视频
│
└─ Side/                                       侧面子项目，负责人chenxi
   ├─ README.md                                侧面数据链、公式、参数和CSV字段说明
   ├─ requirements.txt                         侧面使用的环境依赖，SAM3来自本子项目Model/vendor
   ├─ Model/                                   侧面独立的模型文件
   │  ├─ sam3.pt                               SAM3预训练权重
   │  ├─ bpe_simple_vocab_16e6.txt.gz            文本提示分词词表
   │  └─ vendor/                               SAM3随附源码及许可证，内部脚本见第8节
   ├─ common/                                  仅供侧面使用的基础工具，不含四项行为统计公式
   │  ├─ __init__.py                           标识侧面内部工具包，不计算指标
   │  └─ segmentation_utils.py                 COCO掩膜编解码、CSV契约校验、质心/PCA/头尾等基础几何工具
   ├─ Tracker/                                 侧面的分割及基础测量，两类输入分别处理
   │  ├─ sam3.py                               用已有MOT框提示SAM3分割，保留MOT的鱼ID，输出完整RLE和简化轮廓
   │  ├─ geometry.py                           body任务计算质心/PCA倾角/头尾/伸长率；gill任务提取鳃部框内灰度均值和方差
   │  ├─ Input/                                侧面原始资料
   │  │  ├─ body/                              角度、速度、水层共用的鱼体输入
   │  │  │  ├─ angle.mp4                        当前724帧、30FPS、1280×720鱼体视频
   │  │  │  ├─ gt/gt.txt                        对应鱼体框、视频帧号、连续ID和置信度
   │  │  │  └─ head_mask_shapes.json            可选鱼头掩膜，用于头尾方向修正
   │  │  └─ gill/                              独立的鳃动输入，不与body的同号ID自动关联
   │  │     ├─ video.mp4                        当前1451帧、50FPS、1920×1080鳃动视频
   │  │     └─ gt/gt.txt                        对应鳃部区域框、帧号、ID和置信度
   │  └─ Output/                               侧面基础数据输出
   │     ├─ body/                              鱼体基础表
   │     │  ├─ frames.csv                       SAM3或明确导入来源的鱼体RLE、简化轮廓和元信息
   │     │  └─ result.csv                       位置、PCA角度、头尾、有效性等逐帧基础量
   │     └─ gill/                              鳃部基础表
   │        └─ result.csv                       原视频逐帧鳃部灰度均值/方差、标注框、有效性及来源
   ├─ Behavior/                                四项侧面行为分析，读取基础表，不加载模型或原视频
   │  ├─ angle_analysis.py                     导出鱼体与水平线0～90°角度及每鱼统计
   │  ├─ speed_analysis.py                     对最近10个有效位置拟合像素速度，导出速度分量、方向、均值和距离
   │  ├─ water_layer_analysis.py               将质心y映射成水深比例，统计表/中/底层位置和观测占比
   │  ├─ gill_frequency.py                     对灰度时序滤波/平滑/找峰，统计鳃动次数、频率和峰值时刻
   │  └─ Output/                               侧面指标固定输出位置
   │     ├─ angle/                             角度逐帧表和每鱼汇总
   │     ├─ speed/                             速度逐帧表、每鱼汇总及标准结果
   │     ├─ water_layer/                       水层逐帧表、每鱼汇总和窗口分布
   │     └─ gill/                              鳃动曲线数据、峰数量和峰时刻表
   └─ Visualization/                           可选的网页展示，不参与Tracker或指标计算
      ├─ angle_viewer.py                        检查角度结果并启动本地网页/视频服务
      ├─ angle_viewer.html                      展示侧面视频、轮廓、头尾方向和角度
      ├─ speed_viewer.py                        检查速度结果并启动本地网页/视频服务
      ├─ speed_viewer.html                      展示速度、运动方向和轨迹统计
      ├─ water_layer_viewer.py                  检查水层结果并启动本地网页/视频服务
      ├─ water_layer_viewer.html                展示三层位置和分布，支持页面内调整水面/水底
      ├─ gill_viewer.py                         检查鳃动结果并启动本地网页/视频服务
      ├─ gill_viewer.html                       展示鳃部灰度曲线、候选峰和频率
      └─ data/                                 仅启用网页时生成展示缓存，不保存另一份原视频
```

上面列出了所有自有脚本和运行目录。批量图片及运行生成文件按用途列出；模型第三方内部文件在第8节完整列出。各指标精确文件名和CSV中文列说明分别在Top/README.md与Side/README.md。

## 2. 通过main.py运行

在PowerShell执行：

```powershell
& "D:\Anaconda\envs\sam3\python.exe" "E:\Phd\FishBehavior\main.py"
```

菜单先选择顶部或侧面，再选择指标。顶部8项：速度/平均速度/加速度、距离、摆尾、朝向、区域、静止时间、二维近交、聚集。侧面4项：鳃动、速度、水层、角度。每次只运行所选指标，结束后返回视角菜单；输入0返回或退出。

顶部所有指标共用同一张result.csv，因此选择顶部后先检查基础表。侧面必须先选择指标，才能决定检查body还是gill。缺少可用数据会显示原因并询问是否准备：1执行，2或直接回车返回。已有可用frames.csv时只运行geometry，不重复加载模型。前一步失败或输出校验失败时，不启动行为分析。

例如：顶部→摆尾，实际调用Top/Behavior/tail.py；侧面→速度，实际调用Side/Behavior/speed_analysis.py。子进程使用启动main的同一个解释器，SAM3模型代码各自取本视角的Model/vendor。模型没有安装为全局sam3包也可以运行。

基础检查包含CSV表头/版本/批次、输入路径、图片集合与排序、视频文件大小/修改时间及依赖更新时间。数值有效性、缺帧与指标质量仍由对应脚本校验。输入数据变更后需要重新准备。若人为替换文件内容并保留了文件大小和修改时间，入口不能保证识别，必须主动重跑。

指标参数修改后重跑相应行为脚本即可。修改分割、头尾或几何参数时，应主动重跑对应Tracker步骤再分析；入口不会通过阅读任意Python公式自动识别参数变化。重新分割/测量不会自动刷新其他已存在指标结果，需重跑需要的指标。

## 3. 两个Tracker具体算什么

### 顶部

```text
图片 → sam3.py → frames.csv → geometry.py → result.csv → 各行为脚本
```

sam3.py负责鱼体掩膜和连续ID。geometry.py负责原始掩膜质心、51点头到尾中轴线、头部和尾部轴向量、头向角、头尾轴夹角（摆尾原始角）、沿中轴的像素体长、每条鱼整批有效体长中位数（参考1BL）以及质量标记。它不计算速度、加速度、摆尾频率、区域时间或群体邻接指标。

### 侧面鱼体

```text
视频＋MOT鱼体框/ID → sam3.py → body/frames.csv → geometry.py --task body
→ body/result.csv → 角度/速度/水层
```

侧面sam3.py沿用已有MOT中的身份，不创造跨视频鱼ID关联。body基础几何包括鱼体质心、PCA主轴与水平线0～90°夹角、头尾方向和端点、伸长率、两端宽度及有效性。为保持她原有数值口径，速度/水层使用简化轮廓重新填充的质心，角度使用原掩膜主轴；表中分别标明相应位置字段。侧面不把PCA主轴伪装成顶部的弯曲中轴线。

### 侧面鳃动

```text
鳃动视频＋MOT鳃部框 → geometry.py --task gill → gill/result.csv
→ gill_frequency.py → 频率与峰值结果
```

鳃部基础准备只读取标注框内灰度，输出均值、方差、位置和有效性。滤波、峰检测和次数/分钟换算在Behavior/gill_frequency.py执行。不需要整鱼SAM3，也不使用body那段视频的ID。

两个Side/result.csv同名但路径与内容不同。body与gill示例来自不同视频，不能按track_id相同就拼接成同一条鱼。未来若做跨视角同鱼联合分析，还需单独定义时间同步与ID对应表。

## 4. 保留的计算定义

| 指标 | 顶部YHC | 侧面chenxi |
|---|---|---|
| 速度 | FRAME_STEP默认1，完整位移大小，按头向赋正负，纯侧向为正；BL/s | 最近10个有效观测的x/y对帧号做线性拟合；px/s，速度大小非负 |
| 平均速度 | 相应有效距离之和除以有效时间之和 | 已就绪拟合窗口速度的算术平均 |
| 方向/角度 | 头部中轴倾角，上正下负 | PCA主轴与水平线无符号0～90°；头尾方向另存 |
| 区域 | 图像平面分区，按有效相邻观测实际dt累计秒数 | 水面到底部三等分，按有效观测帧统计水层占比 |
| 鳃动 | 无此模块 | 保留原有标注框灰度离线流程和三类曲线峰统计 |

合并没有统一双方的物理定义或单位。顶部各鱼分别用自己的参考BL，不能把所有鱼统一除以同一长度。二维轨迹近交没有深度信息，不能等同于碰撞；Delaunay局部平均邻距越小表示更紧密。

合并后的侧面行为入口统一读取基础表。鳃动的calculate_gill_frequency(data,fps,...)接收已经提取的灰度记录，不再使用旧版video_path＋RLE的调用签名；原视频读取属于数据准备阶段。Side/common中的COCO解码已修复，压缩字符串和按列展开的普通游程数组可以正确恢复同一掩膜。不要把整鱼掩膜当成鳃部掩膜。所有算法的参数与边界条件详见各自README及脚本注释。

## 5. 独立运行、函数及命令行调用

每个业务脚本仍可单独运行，路径/参数在本脚本顶部或命令行定义。主入口提供可编程调用run_selected(view, metric, prepare='ask', check_only=False)，返回是否成功。

```powershell
# 只检查，不启动计算。
& "D:\Anaconda\envs\sam3\python.exe" "E:\Phd\FishBehavior\main.py" --view top --metric tail --check-only
# 已有基础数据时运行侧面速度；缺少时退出，不等待输入。
& "D:\Anaconda\envs\sam3\python.exe" "E:\Phd\FishBehavior\main.py" --view side --metric speed --prepare no
# 必要时自动准备鳃部数据，之后分析鳃动。
& "D:\Anaconda\envs\sam3\python.exe" "E:\Phd\FishBehavior\main.py" --view side --metric gill --prepare yes
# 主动重新提取侧面鳃部灰度。
& "D:\Anaconda\envs\sam3\python.exe" "E:\Phd\FishBehavior\Side\Tracker\geometry.py" --task gill
```

指标英文键：顶部speed/distance/tail/towards/area/resting_time/trajectory_entanglement/aggregation；侧面gill/speed/water_layer/angle。参数yes只在基础数据缺失或失效时执行准备，不强制每次重新分割。独立运行不会先弹出总菜单；需要自行保证输入已准备好。

## 6. 可选HTML可视化

先完成相应行为分析，再按需运行Side/Visualization内的对应viewer.py。网页服务不自动启动SAM3或补算指标，缺少结果时明确提示。main.py不启动网页。

| 启动脚本 | 默认地址 |
|---|---|
| angle_viewer.py | http://127.0.0.1:8770/ |
| speed_viewer.py | http://127.0.0.1:8772/ |
| water_layer_viewer.py | http://127.0.0.1:8771/ |
| gill_viewer.py | http://127.0.0.1:8768/ |

例如在PowerShell运行：

```powershell
& "D:\Anaconda\envs\sam3\python.exe" "E:\Phd\FishBehavior\Side\Visualization\gill_viewer.py"
```

运行后再用浏览器打开显示的网址。保持命令窗口运行，Ctrl+C关闭服务。不需要HTML时不启动这些脚本即可。页面缓存只用于展示，不回写科学计算CSV；水层页面的水面/水底调整、鳃动页面的峰参数调整都是页面预览，正式导出应修改行为参数后重新运行分析。页面使用当前分析文件，不使用旧项目嵌入的演示数据。

## 7. 各自维护与交付数据来源

YHC只维护Top；chenxi只维护Side。修改已有指标的公式通常不改main；新增指标时，在main的TOP_METRICS或SIDE_METRICS登记菜单名称和文件名，若数据类型不同再增加对应前置检查。common是Side自己的基础工具，Top不依赖它。两份Model/vendor的内部__init__.py是模型组成部分，不能因文件为空随意删除。

本项目使用已有D:\Anaconda\envs\sam3环境，未自动安装或升级任何依赖。main只使用Python标准库；两个子项目各有依赖清单，模型代码从自己的vendor加载。不要用原项目中另一套PyTorch版本的旧清单直接覆盖已验证环境。

项目中附带可检查的实际结果。顶部基础表及既有图表来自原顶部结果的路径迁移，数值保持不变，并通过新入口实际运行验证。侧面全段鱼体基础数据明确从原有真实轮廓导入，source_kind标识导入来源；这不是声称重新对724帧做了一次SAM3推理。新的SAM3入口另用短片进行实际GPU验证。鳃部基础数据从新目录里的完整原视频和标注重新提取。

运行输出使用固定路径，重复执行覆盖对应模块结果，不生成时间戳运行目录。当前基础结果用于直接开始分析；若要重新进行侧面全段SAM3推理，独立执行Side/Tracker/sam3.py，再执行geometry.py --task body和所需行为脚本。

## 8. 模型内部源码完整结构

下面结构在Top/Model/vendor和Side/Model/vendor各保留一份，职责相同但文件独立。它们属于随附第三方模型代码，正常新增行为指标不需要修改。本节逐项说明内部Python脚本；两份SAM3_LICENSE保存原许可证。

```text
vendor/
├─ sam3/  模型Python包
│  ├─ model/  SAM3图像与视频网络
│  │  ├─ utils/  模型辅助工具
│  │  │  ├─ __init__.py  Python模型子包标记及必要导出，不是行为指标入口
│  │  │  ├─ misc.py  模型内部通用张量和配置辅助
│  │  │  ├─ sam1_utils.py  SAM图像模型辅助层和运算
│  │  │  └─ sam2_utils.py  SAM时序模型辅助层和运算
│  │  ├─ __init__.py  Python模型子包标记及必要导出，不是行为指标入口
│  │  ├─ act_ckpt_utils.py  模型激活检查点辅助工具
│  │  ├─ box_ops.py  边界框坐标转换、面积和交并比等几何运算
│  │  ├─ data_misc.py  模型输入输出的数据结构及批处理辅助
│  │  ├─ decoder.py  SAM3查询/特征解码网络
│  │  ├─ edt.py  掩膜距离变换计算
│  │  ├─ encoder.py  图像/文本特征编码和注意力模块
│  │  ├─ geometry_encoders.py  点、框等几何提示编码
│  │  ├─ io_utils.py  模型视频帧读取和预处理工具
│  │  ├─ maskformer_segmentation.py  从网络特征生成分割掩膜
│  │  ├─ memory.py  视频跟踪的记忆表示和注意力处理
│  │  ├─ model_misc.py  模型张量处理和通用网络辅助
│  │  ├─ necks.py  主干网络与下游模块间的多尺度特征连接
│  │  ├─ position_encoding.py  模型空间/序列位置编码
│  │  ├─ sam1_task_predictor.py  SAM交互式图像预测接口
│  │  ├─ sam3_image.py  SAM3图像分割模型主体
│  │  ├─ sam3_image_processor.py  图像预处理、文本/框提示和图像分割推理接口
│  │  ├─ sam3_tracker_base.py  SAM3时序目标跟踪基础网络
│  │  ├─ sam3_tracker_utils.py  跟踪状态和掩膜辅助运算
│  │  ├─ sam3_tracking_predictor.py  单目标/交互式时序跟踪预测封装
│  │  ├─ sam3_video_base.py  视频检测与跟踪组合的基础流程
│  │  ├─ sam3_video_inference.py  多帧视频分割和对象关联推理
│  │  ├─ sam3_video_predictor.py  视频会话、提示及连续传播接口
│  │  ├─ text_encoder_ve.py  文本提示特征编码网络
│  │  ├─ tokenizer_ve.py  文本提示分词及词表处理
│  │  ├─ vitdet.py  视觉Transformer图像主干网络
│  │  └─ vl_combiner.py  视觉和语言特征的组合模块
│  ├─ perflib/  模型性能运算
│  │  ├─ __init__.py  Python模型子包标记及必要导出，不是行为指标入口
│  │  ├─ associate_det_trk.py  检测对象与跟踪对象的关联计算
│  │  ├─ compile.py  性能优化和编译相关封装
│  │  ├─ connected_components.py  掩膜连通域计算
│  │  ├─ fa3.py  优化注意力运算接口
│  │  ├─ masks_ops.py  模型内部掩膜张量处理运算
│  │  └─ nms.py  对重复检测进行非极大值抑制
│  ├─ sam/  交互式SAM基础模块
│  │  ├─ __init__.py  Python模型子包标记及必要导出，不是行为指标入口
│  │  ├─ common.py  SAM模型基础网络层
│  │  ├─ mask_decoder.py  SAM交互式掩膜解码器
│  │  ├─ prompt_encoder.py  SAM点框掩膜提示编码器
│  │  ├─ rope.py  旋转位置编码
│  │  └─ transformer.py  SAM交互式Transformer注意力网络
│  ├─ train/  模型所需训练/掩膜辅助定义
│  │  ├─ __init__.py  Python模型子包标记及必要导出，不是行为指标入口
│  │  └─ masks_ops.py  模型内部掩膜张量处理运算
│  ├─ __init__.py  Python模型子包标记及必要导出，不是行为指标入口
│  ├─ logger.py  配置模型运行日志
│  └─ model_builder.py  组装SAM3图像/视频网络并加载权重
└─ SAM3_LICENSE  SAM3使用许可证
```
