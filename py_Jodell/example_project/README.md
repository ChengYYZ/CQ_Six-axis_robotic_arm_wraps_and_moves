# 钧舵 Python SDK 示例工程

这个示例工程演示了如何基于 `JodellTool-0.0.1` 完成以下事情：

- 搜索串口
- 建立串口连接
- 使能夹爪
- 控制夹持动作
- 查询状态
- 在 ERG32 机型上控制旋转端

## 目录结构

```text
example_project/
├─ README.md
├─ requirements.txt
├─ config.example.json
├─ jodell_controller.py
├─ demo_cli.py
└─ basic_sequence.py
```

## 先决条件

1. 已准备好厂商交付的 wheel 包：
   `D:\Proj0714\py_Jodell\python版本\JodellTool-0.0.1-py3-none-any.whl`
2. Python 3.8 及以上
3. 电脑已接好串口设备并确认驱动正常

## 安装依赖

在当前目录执行：

```powershell
pip install -r .\requirements.txt
```

如果你不想通过 `requirements.txt` 安装，也可以单独执行：

```powershell
pip install pyserial modbus-tk
pip install ..\python版本\JodellTool-0.0.1-py3-none-any.whl
```

## 配置文件

先复制一份示例配置：

```powershell
Copy-Item .\config.example.json .\config.json
```

然后修改为你的设备参数，例如：

```json
{
  "model": "epg",
  "port": "COM3",
  "baud_rate": 115200,
  "slave_id": 9
}
```

支持的 `model` 取值：

- `epg`
- `hepg`
- `evs`
- `erg32`
- `erg26`

## 快速开始

### 1. 搜索串口

```powershell
python .\demo_cli.py ports
```

### 2. 查询状态

```powershell
python .\demo_cli.py status --config .\config.json
```

### 3. 夹爪使能

```powershell
python .\demo_cli.py enable on --config .\config.json
```

### 4. 控制夹持动作

```powershell
python .\demo_cli.py move 255 200 120 --config .\config.json
```

参数顺序含义：

- `255`: 目标位置
- `200`: 最大速度
- `120`: 力矩

### 5. 执行预设动作

```powershell
python .\demo_cli.py preset 1 --config .\config.json
```

### 6. 读取原始寄存器

```powershell
python .\demo_cli.py raw-status 2000 3 2 --config .\config.json
```

### 7. ERG32 旋转端使能

```powershell
python .\demo_cli.py rotate-enable on --config .\config.json
```

### 8. ERG32 旋转控制

相对角度模式：

```powershell
python .\demo_cli.py rotate 360 200 120 --config .\config.json
```

绝对位置模式：

```powershell
python .\demo_cli.py rotate 360 200 120 --absolute --cycle-num 1 --config .\config.json
```

## 基础脚本

如果你更喜欢直接改 Python 代码，而不是走命令行，可以运行：

```powershell
python .\basic_sequence.py --config .\config.json
```

这个脚本会：

1. 建立连接
2. 使能夹爪
3. 如果机型支持，则执行一次简单动作
4. 打印状态
5. 自动断开连接

## 工程说明

### `jodell_controller.py`

对厂商 SDK 做了一个轻量封装，主要解决了几个实际开发问题：

- 统一配置加载
- 统一错误检查
- 统一连接与断开
- 为部分机型补上父类初始化兼容

### `demo_cli.py`

一个命令行入口，方便测试串口连接、控制动作和查询状态。

### `basic_sequence.py`

一个适合作为二次开发起点的最小示例脚本。

## 注意事项

- `enable`、`move`、`status` 这类命令依赖设备真实在线。
- `rotate`、`rotate-enable`、`rotate-preset` 只适用于 `erg32`。
- `erg26` 的波特率修改接口在 SDK 源码里并未真正实现，不建议使用。
- 原始 SDK 的 `scanSalveId()` 有偏移问题，示例工程未直接封装该功能，避免误用。
