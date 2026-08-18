# 钧舵机器人 Python SDK API 中文说明

## 1. 文档范围

本文基于以下两份原始资料整理：

- `D:\Proj0714\py_Jodell\python版本\JodellTool-0.0.1-py3-none-any.whl`
- `D:\Proj0714\py_Jodell\python版本\SDK开发说明(python版本).docx`

同时结合 SDK 实际源码做了静态核对，因此本文不仅包含厂商说明，也补充了以下内容：

- 每个类和方法的实际签名
- 不同机型的接口差异
- 典型调用流程
- 返回值约定
- 从源码中发现的已知问题和使用建议

## 2. 资料结构

SDK 交付包本身不是完整工程，只包含安装包和说明文档：

```text
python版本/
├─ JodellTool-0.0.1-py3-none-any.whl
└─ SDK开发说明(python版本).docx
```

实际 Python 代码主要位于 wheel 包中的 `jodellSdk/jodellSdkDemo.py`。

## 3. 运行环境

### 3.1 Python 版本

建议使用 Python 3.8 及以上版本。

### 3.2 依赖

从 wheel 元数据可确认 SDK 依赖：

- `pyserial`
- `modbus-tk`

### 3.3 安装方式

在 `python版本` 目录下执行：

```powershell
pip install .\JodellTool-0.0.1-py3-none-any.whl
```

如果依赖没有自动安装，可手动补装：

```powershell
pip install pyserial modbus-tk
```

## 4. 通信模型

该 SDK 本质上是对 **串口 Modbus RTU** 指令的封装。

从源码看，串口连接参数固定为：

- 数据位：`8`
- 校验位：`N`
- 停止位：`1`
- 超时时间：`0.1s`

对应源码逻辑在 `serialOperation(com, baudRate, True)` 中创建：

```python
modbus_rtu.RtuMaster(
    serial.Serial(
        port=com,
        baudrate=int(baudRate),
        bytesize=8,
        parity='N',
        stopbits=1
    )
)
```

## 5. 导入与机型映射

SDK 将不同机型封装为不同类。推荐导入方式如下：

```python
from jodellSdk.jodellSdkDemo import (
    ClawEpgTool,
    ClawHepgTool,
    ClawEvsTool,
    ClawErgTool,
    ClawErgTool2,
)
```

机型与类的对应关系：

| 机型 | 对应类 | 说明 |
| --- | --- | --- |
| EPG | `ClawEpgTool` | 基础夹爪 |
| HEPG | `ClawHepgTool` | 带碰撞使能接口 |
| EVS | `ClawEvsTool` | 气压类接口 |
| ERG32 | `ClawErgTool` | 同时包含夹持端和旋转端 |
| ERG26 | `ClawErgTool2` | 以旋转接口为主 |

## 6. 典型开发流程

推荐的调用顺序如下：

1. 创建对应机型对象
2. `searchCom()` 搜索串口
3. `serialOperation(com, baudRate, True)` 建立通信
4. 如有需要，先 `scanSalveId()` 扫描从站地址
5. `clawEnable()` 或 `rotateEnable()` 使能
6. 执行动作指令
7. 查询状态或反馈值
8. 结束后 `serialOperation(com, baudRate, False)` 断开连接

一个最小示例：

```python
from jodellSdk.jodellSdkDemo import ClawEpgTool

claw = ClawEpgTool()

ports = claw.searchCom()
print("串口列表:", ports)

flag = claw.serialOperation("COM3", 115200, True)
print("连接结果:", flag)

flag = claw.clawEnable(9, True)
print("使能结果:", flag)

flag = claw.runWithParam(9, 255, 255, 255)
print("运动结果:", flag)

status = claw.getClawCurrentStatus(9)
print("当前状态:", status)

claw.serialOperation("COM3", 115200, False)
```

## 7. 返回值约定

这一点非常重要，SDK 没有统一抛异常，而是大量使用“返回码或返回字符串”的方式。

### 7.1 写操作

大多数写操作成功时返回：

- `1`

失败时返回：

- 异常字符串，例如 `"timed out"`、`"通讯未连接"` 等

### 7.2 读操作

大多数读操作成功时返回：

- `list`

失败时返回：

- 异常字符串

因此实际项目中建议统一做如下判断：

```python
result = claw.runWithParam(9, 255, 255, 255)
if result != 1:
    raise RuntimeError(f"执行失败: {result}")
```

## 8. 公共基础类：`JodellSDKDemo`

大部分机型的通用接口都继承自 `JodellSDKDemo`。

### 8.1 `searchCom()`

**作用**

搜索本机可用串口。

**签名**

```python
searchCom(self)
```

**返回值**

- 成功：串口名列表，例如 `["COM3", "COM5"]`

### 8.2 `serialOperation(com, baudRate, status)`

**作用**

打开或关闭串口通信。

**签名**

```python
serialOperation(self, com, baudRate, status)
```

**参数**

- `com`: 串口号，例如 `COM3`
- `baudRate`: 波特率，例如 `115200`
- `status`: `True` 表示连接，`False` 表示断开

**返回值**

- 成功：`1`
- 失败：异常字符串

### 8.3 `writeDataToRegister(salveId, address, sendCmdBuf)`

**作用**

向指定从站的指定寄存器连续写入数据。

**签名**

```python
writeDataToRegister(self, salveId, address, sendCmdBuf)
```

**参数**

- `salveId`: 从站 ID
- `address`: 起始寄存器地址
- `sendCmdBuf`: 待写入的寄存器数据列表

**示例**

```python
claw.writeDataToRegister(9, 1000, [1, 2])
```

### 8.4 `getStatus(salveId, address, readMode, count)`

**作用**

通用寄存器读取接口。

**签名**

```python
getStatus(self, salveId, address, readMode, count)
```

**参数**

- `salveId`: 从站 ID
- `address`: 起始地址
- `readMode`: Modbus 读指令码，源码和文档均按 `3` 或 `4` 使用
- `count`: 连续读取寄存器数量

**返回值**

- 成功：寄存器值列表
- 失败：异常字符串

**示例**

```python
claw.getStatus(9, 2000, 3, 2)
```

### 8.5 `clawEnable(salveId, status)`

**作用**

夹持端使能或去使能。

**签名**

```python
clawEnable(self, salveId, status)
```

**参数**

- `salveId`: 从站 ID
- `status`: `True` 使能，`False` 去使能

### 8.6 `runWithoutParam(salveId, cmdId)`

**作用**

执行无参预设动作。

**签名**

```python
runWithoutParam(self, salveId, cmdId)
```

**参数**

- `salveId`: 从站 ID
- `cmdId`: 预设命令编号

### 8.7 `runWithParam(salveId, pos, speed, torque)`

**作用**

执行有参动作，适用于 EPG / HEPG / ERG32 夹持端，以及文档描述中的 ERG26 对应接口。

**签名**

```python
runWithParam(self, salveId, pos, speed, torque)
```

**参数**

- `salveId`: 从站 ID
- `pos`: 目标位置
- `speed`: 最大速度
- `torque`: 力矩

**源码打包方式**

- `pos` 被写入高字节位置字段
- `speed` 和 `torque` 被拼接进同一寄存器

### 8.8 `getClawCurrentStatus(salveId)`

**作用**

读取夹爪当前状态。

**基础状态映射**

- `0`: 未检测到物体
- `1`: 手指在张开方向检测到物体
- `2`: 手指在闭合方向检测到物体
- `3`: 手指已到达指定位置且未检测到物体

**返回值**

- 成功：形如 `["未检测到物体"]`
- 失败：异常字符串

### 8.9 `getClawCurrentLocation(salveId)`

**作用**

读取当前位置。

**返回值**

- 成功：列表，例如 `[128]`

### 8.10 `getClawCurrentSpeed(salveId)`

**作用**

读取当前速度。

### 8.11 `getClawCurrentTorque(salveId)`

**作用**

读取当前力矩。

### 8.12 `getClawCurrentTemperature(salveId)`

**作用**

读取当前温度。

### 8.13 `getClawCurrentVoltage(salveId)`

**作用**

读取当前电压。

### 8.14 `querySoftwareVersion(salveId)`

**作用**

读取软件版本。

**基础类返回格式**

```python
["V主版本.次版本"]
```

例如：

```python
["V1.3"]
```

### 8.15 `changeSalveId(oldId, newId)`

**作用**

修改从站 ID。

**签名**

```python
changeSalveId(self, oldId, newId)
```

### 8.16 `scanSalveId(startId, stopId)`

**作用**

扫描从站范围。

**签名**

```python
scanSalveId(self, startId, stopId)
```

**注意**

源码里存在一个偏移问题：循环内部额外执行了 `myId += 1`，导致实际扫描范围变成了 `startId + 1` 到 `stopId + 1`。如果现场结果和预期不一致，优先检查这一点。

### 8.17 `changeBaudRate(salveId, baudRate)`

**作用**

修改波特率。

**波特率编号**

- `0`: `115200`
- `1`: `57600`
- `2`: `38400`
- `3`: `19200`
- `4`: `9600`
- `5`: `4800`

### 8.18 `clawEncoderZero(salveId)`

**作用**

夹持端编码器对零。

### 8.19 `switchAutoPatrolInspection(salveId, status)`

**作用**

开启或关闭自动巡检。

**参数**

- `status=True`: 开启
- `status=False`: 关闭

## 9. EPG 机型：`ClawEpgTool`

继承自 `JodellSDKDemo`，额外提供模式切换。

### 9.1 `switchMode(salveId, modeIndex)`

**作用**

切换工作模式。

**参数**

- `modeIndex=0`: 串口通讯模式
- `modeIndex=1`: IO 模式

## 10. HEPG 机型：`ClawHepgTool`

在 EPG 基础上多了碰撞使能接口。

### 10.1 `switchMode(salveId, modeIndex)`

与 `ClawEpgTool.switchMode()` 相同。

### 10.2 `collisionEnable(salveId, status)`

**作用**

打开或关闭碰撞使能。

**参数**

- `status=True`: 使能
- `status=False`: 去使能

## 11. EVS 机型：`ClawEvsTool`

EVS 的动作接口和普通夹爪不同，主要围绕气压参数。

### 11.1 `runWithoutParam(salveId, status)`

**作用**

启动或停止运行。

**参数**

- `status=True`: 启动
- `status=False`: 停止

### 11.2 `runWithParam(salveId, maxData, minData, timeout, status)`

**作用**

按气压参数启动或停止。

**参数**

- `maxData`: 气压上限
- `minData`: 气压下限
- `timeout`: 加压超时时间
- `status`: `True` 启动，`False` 停止

**源码细节**

SDK 内部会将 `maxData`、`minData` 转换为：

```text
maxPressure = 100 - maxData
minPressure = 100 - minData
```

这意味着你传入的值不是直接原样下发，而是会先做一次换算。

## 12. ERG32 机型：`ClawErgTool`

该类同时覆盖夹持端和旋转端接口。

### 12.1 夹持端状态

ERG32 重写了夹持端状态映射：

- `0`: 初始状态
- `1`: 手指正向指定位置移动
- `2`: 手指在打开方向运动时，由于接触到物体已经停止
- `3`: 手指在闭合方向运动时，由于接触到物体已经停止
- `4`: 手指打开方向到达指定位置，但没有检测到对象
- `5`: 手指闭合方向到达指定位置，但没有检测到对象

### 12.2 旋转端状态

- `0`: 初始状态
- `1`: 夹爪正向指定位置转动
- `2`: 夹爪在顺时针方向运动时，由于受到阻力已经停止
- `3`: 夹爪在逆时针方向运动时，由于受到阻力已经停止
- `4`: 夹爪顺时针方向旋转到达指定位置
- `5`: 夹爪逆时针方向旋转到达指定位置

### 12.3 `rotateEnable(salveId, status)`

**作用**

旋转端使能。

### 12.4 `runWithParam(salveId, pos, speed, torque)`

**作用**

ERG32 夹持端有参运动。

**注意**

该方法覆盖了基础类实现，寄存器地址和数据打包方式与普通夹爪不同。

### 12.5 `rotateWithoutParam(salveId, cmdId)`

**作用**

执行旋转端无参预设动作。

### 12.6 `rotateWithParam(salveId, angle, speed, torque, absStatus, cycleNum=0)`

**作用**

执行旋转端有参运动。

**参数**

- `angle`: 旋转角度
- `speed`: 最大速度
- `torque`: 力矩
- `absStatus`: 是否使用绝对位置
- `cycleNum`: 圈数，仅在绝对位置模式下使用

**文档示例**

```python
claw.rotateWithParam(9, 360, 255, 255, False, 0)
claw.rotateWithParam(9, 360, 255, 255, True, 1)
```

**补充说明**

- `absStatus=False` 时更接近相对位置控制
- `absStatus=True` 时可配合 `cycleNum` 指定绝对位置模式下的圈数
- 负角度在源码中会做补码处理

### 12.7 设备管理接口差异

ERG32 覆盖了以下方法，寄存器地址与基础类不同：

- `changeSalveId`
- `scanSalveId`
- `changeBaudRate`
- `getClawCurrentStatus`
- `getClawCurrentLocation`
- `getClawCurrentSpeed`
- `getClawCurrentTorque`
- `getClawCurrentTemperature`
- `getClawCurrentVoltage`
- `querySoftwareVersion`

其中 `querySoftwareVersion()` 返回三段版本号：

```python
["V主版本.次版本.修订版本"]
```

## 13. ERG26 机型：`ClawErgTool2`

从源码看，`ClawErgTool2` 主要提供旋转类控制接口。

### 13.1 `rotateWithParam(salveId, angle, speed, torque)`

**作用**

按角度、速度、力矩控制旋转。

**文档示例**

```python
claw.rotateWithParam(9, 3600, 255, 255)
```

### 13.2 `changeBaudRate(salveId, baudRate)`

**现状说明**

源码中的这个方法并未真正实现，函数体只有一条 `print(salveId, baudRate)`。原始文档也注明了“ERG26 机型暂不支持修改波特率”。因此在实际项目中不要依赖这个方法。

## 14. 常用调用示例

### 14.1 串口搜索与连接

```python
from jodellSdk.jodellSdkDemo import ClawEpgTool

claw = ClawEpgTool()
print(claw.searchCom())
print(claw.serialOperation("COM3", 115200, True))
```

### 14.2 使能并执行有参动作

```python
flag = claw.clawEnable(9, True)
if flag != 1:
    raise RuntimeError(flag)

flag = claw.runWithParam(9, 255, 128, 100)
if flag != 1:
    raise RuntimeError(flag)
```

### 14.3 查询当前状态

```python
print(claw.getClawCurrentStatus(9))
print(claw.getClawCurrentLocation(9))
print(claw.getClawCurrentSpeed(9))
print(claw.getClawCurrentTorque(9))
```

### 14.4 ERG32 旋转控制

```python
from jodellSdk.jodellSdkDemo import ClawErgTool

claw = ClawErgTool()
claw.serialOperation("COM3", 115200, True)
claw.rotateEnable(9, True)
claw.rotateWithParam(9, 360, 200, 100, False, 0)
```

## 15. 已知问题与建议

以下问题来自源码静态分析，建议在正式项目中提前规避。

### 15.1 若未连接串口，多数接口直接返回字符串

例如：

- `"通讯未连接"`

建议每次操作都统一校验返回值，不要假设失败一定会抛异常。

### 15.2 `scanSalveId()` 存在地址偏移问题

源码循环内部额外做了 `myId += 1`，会导致扫描范围向后偏移一位。若你希望扫描 `1~10`，实际很可能变成 `2~11`。

### 15.3 若类构造函数未初始化父类状态，部分查询方法可能不稳定

`ClawEpgTool`、`ClawHepgTool`、`ClawEvsTool`、`ClawErgTool2` 的 `__init__()` 没有调用父类初始化。虽然多数控制命令仍可用，但某些依赖基础状态字典的方法在不同场景下可能出现不一致行为。示例工程里已经对这一点做了兼容处理。

### 15.4 ERG26 波特率修改接口未实现

`ClawErgTool2.changeBaudRate()` 不能用于真实设备管理。

### 15.5 寄存器值多为“原始值”

SDK 返回的大多数温度、电压、位置、速度、力矩，本质上是从寄存器中直接拆解出的原始数值，没有进一步做单位换算。若现场需要工程单位，需要结合设备协议另行换算。

## 16. 开发建议

如果你要在生产项目中继续封装这套 SDK，建议至少补上以下几层：

- 统一异常处理，将字符串错误包装为异常
- 统一连接生命周期管理，避免重复开关串口
- 用配置文件管理 `model`、`port`、`baud_rate`、`slave_id`
- 对不同机型做能力矩阵，而不是直接假设所有接口都可用
- 在项目层修正 `scanSalveId()` 的偏移问题

## 17. 配套示例工程

本文档配套的示例工程已放在：

- `D:\Proj0714\py_Jodell\example_project`

示例工程提供：

- 可加载配置文件的控制器封装
- 串口搜索命令
- 夹爪使能、移动、查询状态命令
- ERG32 旋转控制命令
- 可直接运行的基础示例脚本
