# Project0714 项目交接说明

本项目用于 Orbbec RGB-D 相机与珞石 CR18 机械臂的视觉抓取实验。当前主流程是：相机采集彩色图和深度图，YOLO OBB 识别包裹，结合深度点云拟合可吸取表面，将目标点转换到机器人基坐标系，再通过 xCore SDK 控制机械臂和吸盘完成批量取放。

文档日期：2026-08-07

## 项目结构

```text
D:\Proj0714
├─ calibration_suite\                         # 标定、深度验证、表面聚类抓取主程序
│  ├─ orbbec_intrinsic_calibration.py          # Orbbec 彩色相机内参标定
│  ├─ orbbec_eye_to_hand_calibration.py        # 眼在手外手眼标定
│  ├─ depth_click_validation.py                # 点击像素并输出 camera/base 坐标
│  ├─ surface_cluster_grasp.py                 # 当前抓取主程序
│  ├─ project0714_calib\                       # 相机、机器人、坐标变换公共模块
│  ├─ project0714_grasp\                       # 前景、支撑平面、ROI 相关模块
│  └─ workspace\                               # 当前标定结果、ROI、调试图
├─ project_yolo_train_0730\                    # 0730 版本 YOLO OBB 数据和训练结果
├─ project_yolo_train_0804\                    # 0804 版本 YOLO OBB 数据和训练结果，抓取默认使用这里的 best2.pt
├─ xCoreSDK-Python-0.7.1-win\                  # 珞石 xCore Python SDK 运行时
├─ xCoreSDK-Python-main\                       # xCore SDK 源码/备用目录
├─ py_Jodell\                                  # 钧舵夹爪 Python SDK 示例工程
├─ Jodell_PN\                                  # 钧舵 Profinet GSDML 文件
├─ RobotAssist_5.0.13.0439\                    # RobotAssist 安装/交付包
├─ weights\                                    # CLIP 等模型权重缓存
├─ 六轴电缸夹爪最终版\                         # 夹具机械图纸、装配模型、PDF/DWG/STEP
├─ 说明书\                                    # 机器人和 Profinet 使用手册
├─ run_*.cmd                                   # 常用入口脚本
└─ requirements.txt                            # 根目录依赖清单
```

## 当前可复用成果

- 相机内参：`calibration_suite\workspace\intrinsics\camera_intrinsics.json`
- 内参采集图像：`calibration_suite\workspace\intrinsics\images\`，当前 20 张
- 内参重投影误差：`mean_reprojection_error = 0.21249116957187653 px`
- 手眼标定结果：`calibration_suite\workspace\eye_to_hand\eye_to_hand_result.json`
- 手眼样本：`calibration_suite\workspace\eye_to_hand\samples\`，当前 26 组
- 手眼方法：`TSAI`
- 手眼平移结果：
  - `base_to_camera.translation_mm = [-47.243, -804.426, 1032.485]`
  - `camera_to_base.translation_mm = [16.953, -835.118, 1008.785]`
- 抓取 ROI：`calibration_suite\workspace\surface_grasp\manual_roi.json`
- 抓取默认 YOLO 权重：`project_yolo_train_0804\runs\yolo11s_obb_train-4\weights\best2.pt`
- 当前类别：`parcel_box`、`document_envelope`、`soft_parcel`

以上文件和目录是当前项目能直接复现运行状态的关键资产。更换相机位置、相机分辨率、夹具 TCP、机器人安装位置或料箱位置后，需要重新确认 ROI、深度验证和手眼结果。

## 环境准备

推荐环境：

- Windows
- Python 3.8，xCore SDK 当前目录也包含部分 Python 3.8 到 3.12 的二进制支持，但现有脚本默认按 3.8 创建环境
- Orbbec 相机驱动和 `pyorbbecsdk2`
- 珞石 xCore SDK，本项目已带 `xCoreSDK-Python-0.7.1-win`
- 机器人控制柜 IP 默认 `192.168.2.160`

首次部署建议在 `D:\Proj0714` 执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\calibration_suite\bootstrap_venv.ps1
.\.venv\Scripts\Activate.ps1
```

`bootstrap_venv.ps1` 会优先复用本机已有环境：

- `D:\RoboticARMGrasping\TCP-IP\.venv\Lib\site-packages`
- `D:\orbbec_pyenv\Lib\site-packages`

如果不能复用，它会执行：

```powershell
pip install -r .\calibration_suite\requirements.txt
```

根目录也提供了完整依赖清单：

```powershell
pip install -r .\requirements.txt
```

说明：

- `xCoreSDK-Python-0.7.1-win` 是本地 SDK 目录，不通过 pip 安装。
- `py_Jodell\python版本\JodellTool-0.0.1-py3-none-any.whl` 是夹爪示例的本地 wheel。
- 如需 YOLOE/CLIP 相关功能，可运行 `install_yoloe_clip.cmd`；需要 GitHub/PyPI 网络可达。

## 一键脚本

在 `D:\Proj0714` 目录下运行：

```powershell
.\run_calibration_menu.cmd
```

也可以直接运行单个脚本：

```powershell
.\run_intrinsic_capture.cmd
.\run_intrinsic_calibrate.cmd
.\run_eye_to_hand_capture.cmd
.\run_eye_to_hand_solve.cmd
.\run_depth_click_validation.cmd
.\run_surface_cluster_grasp.cmd
.\run_surface_cluster_grasp_workspace_debug.cmd
.\run_move_robot_pose.cmd
```

默认参数：

- 棋盘格内角点：`5 x 8`
- 棋盘格边长：`25 mm`
- 机器人 IP：`192.168.2.160`
- 相机彩色流：`1280 x 800 @ 10 FPS`
- 深度流：`640 x 400 @ 5 FPS`
- 深度和彩色对齐：软件对齐 `sw`
- 当前吸盘 2 使用的 tool：`tool4`（启动脚本会覆盖为本次标定的 TCP）
- 抓取姿态忽略当前工具 Y 方向和包裹 OBB 短边，工具 X 轴水平朝向固定为现场已成功完成下降吸取的 `RZ=-96.09°`，仅根据表面法向调整吸盘接触轴。
- 抓取默认 wobj：`wobj0`
- 抓取默认速度：普通入口 `50 mm/s`，面单入口 `100 mm/s`

## 标定流程

### 1. 相机内参

采集：

```powershell
.\run_intrinsic_capture.cmd
```

窗口内按键：

- `s` 保存当前棋盘格图像
- `q` 退出

建议采集 15 到 25 张，覆盖不同距离、角度和视野位置。

求解：

```powershell
.\run_intrinsic_calibrate.cmd
```

结果写入：

```text
calibration_suite\workspace\intrinsics\camera_intrinsics.json
```

### 2. 眼在手外手眼标定

采集：

```powershell
.\run_eye_to_hand_capture.cmd
```

操作方式：

- 将棋盘格固定在机械臂末端，Orbbec 相机固定不动。
- 移动机器人到不同姿态，确认画面内能看到完整棋盘格。
- 按 `s` 保存一组图像和当前机器人位姿。
- 至少保存 12 组，推荐 15 到 20 组以上。

求解：

```powershell
.\run_eye_to_hand_solve.cmd
```

结果写入：

```text
calibration_suite\workspace\eye_to_hand\eye_to_hand_result.json
```

## 深度点击验证

运行：

```powershell
.\run_depth_click_validation.cmd
```

用途：

- 左键点击彩色图或深度图中的一点。
- 程序输出该点深度、相机坐标和机器人 `base` 坐标。
- 日志写入 `calibration_suite\workspace\validation\depth_click_validation.log`。

按键：

- `q` 退出
- `c` 清除当前标记

该工具只验证点坐标，不输出完整末端姿态。建议优先点击平整、无遮挡、无反光区域。

## 抓取主流程

真实运行：

```powershell
.\run_surface_cluster_grasp.cmd
```

调试运行，不控制机器人，且忽略工作空间过滤：

```powershell
.\run_surface_cluster_grasp_workspace_debug.cmd
```

主程序默认加载：

```text
calibration_suite\workspace\intrinsics\camera_intrinsics.json
calibration_suite\workspace\eye_to_hand\eye_to_hand_result.json
calibration_suite\workspace\surface_grasp\manual_roi.json
project_yolo_train_0804\runs\yolo11s_obb_train-4\weights\best2.pt
```

识别和候选生成逻辑：

1. Orbbec 采集 RGB-D 图像，并进行深度到彩色对齐。
2. YOLO OBB 默认识别包裹区域；可通过 `--detector rgb` 切到旧版 RGB/深度启发式方案。
3. 在 ROI 内采样深度点云，拟合支撑面和候选顶面。
4. 计算候选点在相机坐标系和机器人基坐标系下的位置。
5. 根据工作空间、法向、平整度、点数等条件过滤运动风险。
6. 点击单个候选或按 Enter 执行批量抓取。

窗口按键：

- 鼠标左键：选择并执行一个安全候选
- `Enter` 或 `a`：执行当前识别结果中的安全候选批次，并进入自动刷新批量模式
- `Space`：软件停止当前机器人运动，并退出批量自动执行
- `d` 或 `r`：重新识别当前画面
- `m`：编辑整体 ROI
- `1`、`2`、`3`、`4`：编辑 Floor、Left Wall、Right Wall、Back Wall 支撑面 ROI
- `b`：添加排除区域 ROI，例如挡板或固定障碍物
- ROI 编辑中左键添加点，右键撤销一点，`c` 或 `Enter` 保存，`Esc` 取消
- `x`：清除当前 ROI；不在 ROI 编辑中时清除全部 ROI
- `q`：退出

抓取路径概念：

```text
B -> A* -> A/approach -> pickup -> A -> A* -> B -> C -> D
```

- `A/approach` 和 `pickup` 根据当前候选目标动态计算。
- 空载进入料箱使用已示教验证的 `B -> A* -> 动态A`，两段均为 MoveJ；不再使用程序推算的固定高度绕行、运输倾角或在线姿态试探。只有动态 A 到 pickup 的短距离下降使用 MoveL。
- `A*`、`B`、`C`、`D` 是 `surface_cluster_grasp.py` 内写死的过渡/放置点；`B*` 已删除。
- 吸盘 IO 默认是 `DO3_4`，即 `--suction-do-board 3 --suction-do-port 4`。
- 默认吸盘逻辑是 ON 写 True，必要时可加 `--invert-suction-io`。
- 异常情况下默认保留吸盘状态，便于人工恢复；如需异常后关闭吸盘，可加 `--suction-off-on-error`。

常用参数示例：

```powershell
.\run_surface_cluster_grasp.cmd --dry-run
.\run_surface_cluster_grasp.cmd --detector rgb
.\run_surface_cluster_grasp.cmd --yolo-conf 0.45
.\run_surface_cluster_grasp.cmd --disable-suction-io
.\run_surface_cluster_grasp.cmd --analysis-refresh-s 0
```

## YOLO 训练和权重

训练目录：

- `project_yolo_train_0730`
- `project_yolo_train_0804`

当前主程序默认使用：

```text
project_yolo_train_0804\runs\yolo11s_obb_train-4\weights\best2.pt
```

`project_yolo_train_0804\data.yaml` 中的类别：

```yaml
names:
  0: parcel_box
  1: document_envelope
  2: soft_parcel
```

如果要替换模型，可在运行抓取主程序时传入：

```powershell
.\run_surface_cluster_grasp.cmd --yolo-model D:\path\to\best.pt
```

## 机器人和夹爪

机器人通信由 `calibration_suite\project0714_calib\xcore_robot.py` 封装，依赖本项目内的 xCore SDK 目录。程序会设置：

- 自动模式
- 非实时指令模式
- 上电
- 默认速度和转弯区

单独移动机器人可运行：

```powershell
.\run_move_robot_pose.cmd
```

输入目标位姿时，XYZ 单位为 mm，RX/RY/RZ 单位为 deg。输入 4 个值时默认只改 X/Y/Z/RZ，保持当前 RX/RY。

钧舵夹爪示例在：

```text
py_Jodell\example_project
```

该示例用于串口搜索、夹爪使能、动作控制、状态查询和 ERG32 旋转端测试。它不是当前 `surface_cluster_grasp.py` 默认抓取链路的一部分。

## 安全和交接注意事项

1. 真实运行 `run_surface_cluster_grasp.cmd` 前，先使用 `run_surface_cluster_grasp_workspace_debug.cmd` 或 `--dry-run` 看候选点、路径点和工作空间过滤结果。
2. 真实抓取前确认 RobotAssist 中存在吸盘 2 的 `tool4` 和 `wobj0`；当前启动脚本使用 TCP 覆盖值 `XYZ=[-125.370, 87.591, 279.781] mm`、`RPY=[178.740, -32.430, -87.920] deg`。
   吸盘 2 的接触方向为工具坐标系 `-Z`，因此抓取入口固定使用 `--tool-contact-axis minus-z`。
3. `surface_cluster_grasp.py` 内的 `A*`、`B`、`C`、`D` 与现场料箱、挡板、放置区强相关，移动设备或改夹具后必须重新检查。空载进料路径为 `B -> 动态A`，不再使用 B*。
4. ROI 文件 `manual_roi.json` 与相机安装姿态强相关，移动相机或料箱后需要重新绘制。
5. 工作空间默认限制为 `X[-450, 600] mm`、`Y[-1250, -500] mm`、`Z[-50, 500] mm`。`--ignore-workspace-filter` 只允许和 `--dry-run` 一起使用。
6. `Space` 是程序内软件停止，不等同于控制柜急停。现场调试仍需保留硬件急停和人工看护。
7. `.gitignore` 当前忽略 `.venv`、`calibration_suite/workspace`、`__pycache__`、`*.pyc`。本机已有的 workspace 标定资产如果要交付给他人，需要单独确认是否随包交付。
8. 根目录下存在大量模型、CAD、SDK、日志和说明书，迁移项目时不要只复制 Python 文件。

## 常见问题

### run_yoloe_package_seg.cmd 无法运行

当前 `run_yoloe_package_seg.cmd` 指向：

```text
yolo_trainning\seg2package2.py
```

但根目录下未发现该路径。该脚本看起来属于 YOLOE/CLIP 分割实验入口，不是当前 `surface_cluster_grasp.py` 默认抓取链路。交接时可先标记为历史/待确认脚本。

### pyorbbecsdk 不可用

先运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\calibration_suite\bootstrap_venv.ps1
```

如果仍失败，检查 Orbbec 驱动、相机 USB 连接和 SDK 安装状态。

### xCoreSDK_python 加载失败

检查当前 Python 版本和 SDK 二进制是否匹配。程序会在以下目录查找：

```text
xCoreSDK-Python-0.7.1-win\Release\windows
xCoreSDK-Python-main\xCoreSDK-Python-main\Release\windows
```

目录内应包含 `xCoreSDK.dll` 和匹配当前 Python 版本的 `xCoreSDK_python.cpXX-win_amd64.pyd`。

### 机器人连接失败

确认：

- 机器人 IP 是否仍为 `192.168.2.160`
- PC 与机器人控制柜网络是否连通
- 控制柜是否允许远程自动模式控制
- RobotAssist/控制器中 tool 和 wobj 名称是否一致

### YOLO 或 CLIP 依赖失败

普通 OBB 抓取默认只需要 `ultralytics` 和权重文件。YOLOE/CLIP 相关脚本需要额外网络访问 GitHub/PyPI，可运行：

```powershell
.\install_yoloe_clip.cmd
```
