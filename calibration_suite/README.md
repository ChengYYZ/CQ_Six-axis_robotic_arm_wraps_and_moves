# Project0714 标定工具

这套脚本把 `Project0714` 里现有的机器人 SDK 和 Orbbec 工业深度相机接起来，提供两个可复用的程序：

1. `orbbec_intrinsic_calibration.py`
   作用：基于张正友标定法完成 Orbbec 彩色相机内参标定。
2. `orbbec_eye_to_hand_calibration.py`
   作用：完成眼在手外手眼标定，输出相机和机器人基座之间的外参。

## 目录结构

```text
calibration_suite/
├─ bootstrap_venv.ps1
├─ orbbec_intrinsic_calibration.py
├─ orbbec_eye_to_hand_calibration.py
├─ project0714_calib/
│  ├─ common.py
│  ├─ orbbec_camera.py
│  └─ xcore_robot.py
├─ requirements.txt
└─ workspace/
```

`workspace` 会在运行时自动生成，用来保存图片、样本和结果。

## 1. 创建虚拟环境

在 `D:\Proj0714` 下执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\calibration_suite\bootstrap_venv.ps1
.\.venv\Scripts\Activate.ps1
```

当前机器上如果已经存在你之前用过的 `D:\orbbec_pyenv`，脚本会优先复用那套本地 `numpy / cv2 / pyorbbecsdk`，不需要联网。

如果你想显式指定 Python 3.8：

```powershell
.\calibration_suite\bootstrap_venv.ps1 -PythonVersion 3.8
```

## 一键启动

如果你不想每次输入一长串命令，可以直接在 `D:\Proj0714` 里运行这些脚本：

```powershell
.\run_intrinsic_capture.cmd
.\run_intrinsic_calibrate.cmd
.\run_eye_to_hand_capture.cmd
.\run_eye_to_hand_solve.cmd
```

也可以直接打开总菜单：

```powershell
.\run_calibration_menu.cmd
```

这些脚本当前内置的默认参数是：

- 棋盘格：`5 x 8`
- 方格边长：`25 mm`
- 机器人 IP：`192.168.2.160`

如果你的实际参数不同，我可以再帮你改成你的设备专用版本。

## 深度点击验证

可以运行下面这个脚本，打开彩色图和深度图，鼠标左键点击后输出：

- 点击点深度值
- 该点在相机坐标系下的位置
- 该点通过手眼标定变换后的 `base` 坐标

```powershell
.\run_depth_click_validation.cmd
```

说明：

- 该工具输出的是点坐标，不是完整位姿
- 建议点击法兰中心附近的大平面，避免边缘、孔、反光区域
- `q` 退出，`c` 清除当前标记

## 2. 相机内参标定

### 2.1 采集棋盘格图像

```powershell
python .\calibration_suite\orbbec_intrinsic_calibration.py capture `
  --cols 5 `
  --rows 8 `
  --square-size-mm 25 `
  --save-dir .\calibration_suite\workspace\intrinsics\images
```

操作方式：

- `s`：保存当前帧
- `q`：退出

建议采集 15 到 25 张，覆盖不同距离、不同倾角、不同视野位置。

### 2.2 计算内参

```powershell
python .\calibration_suite\orbbec_intrinsic_calibration.py calibrate `
  --cols 5 `
  --rows 8 `
  --square-size-mm 25 `
  --image-dir .\calibration_suite\workspace\intrinsics\images `
  --output .\calibration_suite\workspace\intrinsics\camera_intrinsics.json
```

输出文件里包含：

- `camera_matrix`
- `dist_coeffs`
- `mean_reprojection_error`
- `image_size`

## 3. 眼在手外手眼标定

### 3.1 采集样本

把棋盘格固定到机械臂末端，保持 Orbbec 相机固定不动。

```powershell
python .\calibration_suite\orbbec_eye_to_hand_calibration.py capture `
  --robot-ip 192.168.2.160 `
  --cols 5 `
  --rows 8 `
  --square-size-mm 25 `
  --save-dir .\calibration_suite\workspace\eye_to_hand\samples
```

操作方式：

- 手动或通过 HMI 把机器人移动到一个新姿态
- 确认画面里能看到完整棋盘格
- 按 `s`，程序会同时保存图像和当前末端位姿
- 至少采集 12 组，推荐 15 到 20 组

建议让末端在三个方向上都有明显平移，并带上足够的姿态变化。

### 3.2 求解手眼外参

```powershell
python .\calibration_suite\orbbec_eye_to_hand_calibration.py solve `
  --cols 5 `
  --rows 8 `
  --square-size-mm 25 `
  --sample-dir .\calibration_suite\workspace\eye_to_hand\samples `
  --intrinsics .\calibration_suite\workspace\intrinsics\camera_intrinsics.json `
  --output .\calibration_suite\workspace\eye_to_hand\eye_to_hand_result.json
```

默认方法是 `Tsai`，也可以切换：

```powershell
python .\calibration_suite\orbbec_eye_to_hand_calibration.py solve --method PARK
python .\calibration_suite\orbbec_eye_to_hand_calibration.py solve --method DANIILIDIS
```

输出结果同时包含：

- `base_to_camera`
- `camera_to_base`
- 每个样本的 PnP 重投影误差
- 基于所有样本估计的末端到标定板一致性统计

## 4. 注意事项

1. 这两套脚本默认使用 Orbbec 彩色图像做角点检测，深度图不是必须项。
2. 机器人位姿读取依赖当前工程里的 `xCoreSDK-Python-0.7.1-win`。
3. 手眼标定脚本把机器人返回的 `base -> gripper` 位姿先求逆，再送入 `cv2.calibrateHandEye`，这是眼在手外配置对应的标准处理方式。
4. 这次环境优先复用了你旧项目留下的 `D:\orbbec_pyenv`。如果那套环境后续被删除，再让 `bootstrap_venv.ps1` 走联网安装即可。
5. 如果 Orbbec SDK 安装后仍然无法打开相机，优先重新运行 `bootstrap_venv.ps1`，它会尝试执行官方的环境初始化脚本。
