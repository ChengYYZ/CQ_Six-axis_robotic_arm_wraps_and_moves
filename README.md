# 六轴机械臂包裹视觉识别与抓取系统

本项目面向包裹分拣与取放场景，集成 RGB-D 视觉、目标检测、点云处理、机器人坐标变换和机械臂控制，构建从视觉采集到抓取执行的实验系统。

系统使用 Orbbec RGB-D 相机获取彩色图像和深度数据，通过 YOLO OBB 检测包裹区域，结合深度点云分析目标表面、筛选抓取位置，并将目标坐标转换到机器人基坐标系，供机械臂和吸盘执行抓取与放置。

仓库还包含相机内参标定、眼在手外手眼标定、深度坐标验证、空平台基准采集、条码识别和桌面操作界面等工具。

## 主要功能

- **包裹检测**：使用 YOLO OBB 检测包裹、文件袋和软包裹等目标。
- **RGB-D 点云分析**：采集并处理彩色图像与深度数据，分析平台支撑面和包裹表面。
- **抓取点筛选**：结合 ROI、表面法向、平整度、点数和工作空间限制生成抓取候选。
- **机器人坐标转换**：根据相机内参与手眼标定结果，将视觉坐标转换为机器人基坐标。
- **机械臂与吸盘控制**：通过珞石 xCore SDK 控制 CR18 六轴机械臂及吸盘执行取放。
- **相机与手眼标定**：提供 Orbbec 相机内参标定、眼在手外手眼标定和深度点击验证工具。
- **空平台基准采集**：采集平台深度基准，辅助过滤未明显高于平台的抓取候选。
- **条码与面单处理**：包含条码识别、面单检测和相关批处理工具。
- **调试与验证**：提供命令行入口，支持标定、抓取调试、离线验证和机器人位姿移动。

## 技术栈

- Python
- OpenCV、NumPy、SciPy
- Orbbec RGB-D 相机 SDK
- Ultralytics YOLO OBB
- 珞石 xCore Python SDK
- RGB-D 深度处理与点云几何分析

## 项目结构

| 路径 | 内容 |
| --- | --- |
| `calibration_suite/` | 相机标定、手眼标定、深度验证、抓取与桌面程序 |
| `project_yolo_train_0730/`、`project_yolo_train_0804/` | YOLO 训练数据和训练结果 |
| `xCoreSDK-Python-0.7.1-win/` | 珞石 xCore Python SDK 运行时 |
| `py_Jodell/`、`Jodell_PN/` | 钧舵夹爪示例和 Profinet 相关文件 |
| `weights/` | 模型及相关权重文件 |
| `run_*.cmd` | 常用功能的 Windows 启动脚本 |
| `requirements.txt` | Python 依赖清单 |

## 环境准备

本项目主要运行于 Windows 实验环境。完整运行需要连接对应的相机和机械臂，并准备相机驱动、机器人 SDK、模型权重及设备标定配置。

安装 Python 依赖：

```powershell
pip install -r requirements.txt
```

Orbbec 相机 SDK 和 xCore SDK 还需按各自设备及 Python 环境配置。部分模型权重、标定数据和硬件参数与具体实验现场相关。

## 常用入口

项目提供 `.cmd` 脚本用于启动常见功能：

```powershell
.\run_calibration_menu.cmd
.\run_intrinsic_capture.cmd
.\run_intrinsic_calibrate.cmd
.\run_eye_to_hand_capture.cmd
.\run_eye_to_hand_solve.cmd
.\run_depth_click_validation.cmd
.\run_surface_cluster_grasp.cmd
```

使用前请确认启动脚本中的设备地址、相机参数、模型路径和标定文件路径符合当前环境。

## 运行提示

更换相机位置、相机分辨率、夹具 TCP、机器人安装位置或工作台布局后，应重新检查相机内参、手眼变换、抓取 ROI、工作空间和运动过渡点。

连接真实机械臂执行前，建议先使用调试模式或 `--dry-run` 检查检测结果、抓取候选和运动规划。现场操作应配备硬件急停，并由人员监护。

## 项目状态

本仓库是持续迭代中的机器人视觉抓取实验项目，包含应用程序、标定和验证工具、模型训练资料及相关 SDK。实际可运行功能取决于硬件设备、软件依赖、模型文件和本机标定配置。

## 关键词

`机械臂` `机器人视觉` `包裹分拣` `视觉抓取` `RGB-D` `Orbbec` `YOLO OBB` `珞石 CR18` `手眼标定` `点云处理` `条码识别`
