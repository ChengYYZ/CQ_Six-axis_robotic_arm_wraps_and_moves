"""
珞石机器人位姿移动脚本。

功能：
    将机器人从当前任意位姿移动到目标笛卡尔位姿。

适用环境：
    - Windows
    - xCoreSDK Python 0.7.1
    - 非实时模式运动控制

交互方式：
    - 启动后显示当前位姿
    - 输入 M 后输入目标位姿，机器人执行一次运动
    - 运动结束后程序不会退出，而是刷新当前位姿并继续等待下一次输入
    - 在关键输入提示处输入 q 可退出程序

单位说明：
    - x/y/z：毫米
    - rx/ry/rz：终端输入为角度，程序内部自动转换为弧度

安全说明：
    1. 默认只替换 X/Y/Z/RZ，保持当前 RX/RY 不变。
       如果工艺要求 RX/RY 也固定，可以在终端输入 6 个参数。
    2. 默认使用 MoveJ，因为从任意起始位姿出发时通常比 MoveL 更稳妥。
"""

from __future__ import annotations

import argparse
import importlib
import math
import platform
import sys
import time
from pathlib import Path


# 机器人 IP 默认值为"192.168.2.160"。
DEFAULT_ROBOT_IP = "192.168.2.160"

DEFAULT_MOTION = "movej"
DEFAULT_SPEED = 1500.0
DEFAULT_ZONE = 0.0
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_PRE_APPROACH_HEIGHT_MM = 150.0
DEFAULT_TRANSIT_SAFE_Z_MM = 650.0


def add_sdk_paths() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "xCoreSDK-Python-0.7.1-win",
        script_dir / "xCoreSDK-Python-main" / "xCoreSDK-Python-main",
    ]

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

    for root in candidates:
        release_dir = root / "Release"
        windows_dir = release_dir / "windows"
        stub_dir = windows_dir / "xCoreSDK_python"

        runtime_dll = windows_dir / "xCoreSDK.dll"
        runtime_pyd = windows_dir / f"xCoreSDK_python.{py_tag}-win_amd64.pyd"

        # 优先选择同时带有 .dll 和与当前 Python 版本匹配的 .pyd 的发布包目录。
        if runtime_dll.exists() and runtime_pyd.exists():
            # 将 SDK 相关目录加入搜索路径，这样脚本可以独立运行。
            for path in (str(windows_dir), str(stub_dir), str(release_dir), str(root)):
                sys.path.insert(0, str(path))
            return root

    raise FileNotFoundError(
        "未找到可运行的 xCore SDK 目录。\n"
        f"当前 Python 版本需要的二进制标记：{py_tag}\n"
        "期望以下目录中存在 xCoreSDK.dll 和匹配版本的 xCoreSDK_python.*.pyd：\n"
        f"  - {candidates[0]}\n"
        f"  - {candidates[1]}"
    )


SDK_ROOT = add_sdk_paths()

if sys.version_info < (3, 8) or sys.version_info > (3, 12):
    raise RuntimeError(
        "当前 Python 版本不受支持。\n"
        f"检测到版本：{sys.version.split()[0]}\n"
        "xCoreSDK Python 0.7.1 仅支持 Python 3.8 ~ 3.12。\n"
        "请改用 Python 3.8、3.9、3.10、3.11 或 3.12 运行本脚本。"
    )

if platform.system() != "Windows":
    raise ImportError("当前操作系统不受支持")

# 直接导入二进制扩展模块，避免被同名 .pyi 目录覆盖。
xCoreSDK_python = importlib.import_module("xCoreSDK_python")

MOVE_EVENT_ID_KEY = "cmdID"
MOVE_EVENT_REACH_TARGET_KEY = "reachTarget"
MOVE_EVENT_ERROR_KEY = "error"
MOVE_EVENT_REMARK_KEY = "remark"


def validate_sdk_module() -> None:
    # 如果 Python 版本不匹配，或者只拷贝了源码没拷贝 .pyd/.dll，
    # import 很可能会落到 xCoreSDK_python 的存根目录上，进而找不到真实类。
    required_attrs = ("xMateRobot", "MoveJCommand", "MoveLCommand", "CoordinateType")
    missing = [name for name in required_attrs if not hasattr(xCoreSDK_python, name)]
    if missing:
        raise RuntimeError(
            "xCoreSDK Python 模块加载不完整，缺少以下接口："
            + ", ".join(missing)
            + "\n常见原因：\n"
            + "1. 当前 Python 版本与 SDK 二进制不匹配。\n"
            + "2. 只复制了 xCoreSDK-Python-main 源码，没有复制 Release/windows 下的 .pyd 和 xCoreSDK.dll。\n"
            + "3. xCoreSDK-Python-0.7.1-win.zip 没有正确解压到脚本可访问的位置。\n"
            + f"当前检测到的 SDK 根目录：{SDK_ROOT}"
        )


validate_sdk_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取当前位姿，并在终端输入目标位姿后驱动机器人运动。")
    parser.add_argument("--ip", default=DEFAULT_ROBOT_IP, help="机器人控制器 IP 地址。")
    parser.add_argument(
        "--motion",
        choices=("movej", "movel"),
        default=DEFAULT_MOTION,
        help="运动类型。对于任意起始位姿，movej 通常更稳妥。",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help="MoveJ/MoveL 的运动速度，单位 mm/s。",
    )
    parser.add_argument(
        "--zone",
        type=float,
        default=DEFAULT_ZONE,
        help="转弯区，单位 mm。设为 0 表示到点停。",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="运动超时时间，单位秒。",
    )
    parser.add_argument(
        "--pre-approach-height",
        type=float,
        default=DEFAULT_PRE_APPROACH_HEIGHT_MM,
        metavar="MM",
        help="Height of the target pre-approach point above the final target (mm). Use 0 to disable.",
    )
    parser.add_argument(
        "--transit-safe-z",
        type=float,
        default=DEFAULT_TRANSIT_SAFE_Z_MM,
        metavar="MM",
        help="Absolute high-Z used for Cartesian transit waypoints (mm). Use 0 to disable.",
    )
    return parser.parse_args()


def clear_error(ec: dict) -> None:
    ec.clear()


def error_text(ec: dict) -> str:
    try:
        return str(xCoreSDK_python.message(ec))
    except Exception:
        return str(ec)


def ensure_ok(ec: dict, action: str) -> None:
    code = ec.get("code", ec.get("ec", 0))
    if code not in (0, "0", None):
        raise RuntimeError(f"{action} 执行失败：{error_text(ec)}")


def sdk_call(action: str, ec: dict, func, *args):
    clear_error(ec)
    result = func(*args, ec)
    ensure_ok(ec, action)
    return result


def format_pose(trans: list[float], rpy_rad: list[float]) -> str:
    rpy_deg = [math.degrees(v) for v in rpy_rad]
    trans_mm = [value * 1000.0 for value in trans]
    return (
        f"X={trans_mm[0]:.1f} mm, Y={trans_mm[1]:.1f} mm, Z={trans_mm[2]:.1f} mm, "
        f"RX={rpy_deg[0]:.2f} deg, RY={rpy_deg[1]:.2f} deg, RZ={rpy_deg[2]:.2f} deg"
    )


def pose_distance_mm(current, target) -> float:
    dx = (current.trans[0] - target.trans[0]) * 1000.0
    dy = (current.trans[1] - target.trans[1]) * 1000.0
    dz = (current.trans[2] - target.trans[2]) * 1000.0
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def clone_pose(source):
    pose = xCoreSDK_python.CartesianPosition()
    pose.trans = list(source.trans)
    pose.rpy = list(source.rpy)
    pose.confData = list(source.confData)
    pose.external = list(source.external)
    pose.hasElbow = source.hasElbow
    pose.elbow = source.elbow
    return pose


def make_waypoint(position_m, orientation_source, config_source):
    waypoint = clone_pose(config_source)
    waypoint.trans = list(position_m)
    waypoint.rpy = list(orientation_source.rpy)
    return waypoint


def build_safe_route(current, target, args: argparse.Namespace):
    """Build conservative MoveJ stages without introducing new SDK/IK APIs."""
    stages = []
    current_z_mm = current.trans[2] * 1000.0
    target_z_mm = target.trans[2] * 1000.0
    pre_z_mm = target_z_mm + max(0.0, args.pre_approach_height)

    if args.transit_safe_z > 0.0:
        safe_z_mm = max(args.transit_safe_z, current_z_mm, pre_z_mm)
        stages.append(
            (
                "transit-rise",
                make_waypoint(
                    [current.trans[0], current.trans[1], safe_z_mm / 1000.0],
                    current,
                    current,
                ),
            )
        )
        stages.append(
            (
                "transit-over-target",
                make_waypoint(
                    [target.trans[0], target.trans[1], safe_z_mm / 1000.0],
                    current,
                    current,
                ),
            )
        )

    if args.pre_approach_height > 0.0:
        # The orientation changes only after reaching the target XY at a safe height.
        stages.append(
            (
                "pre-approach",
                make_waypoint(
                    [target.trans[0], target.trans[1], pre_z_mm / 1000.0],
                    target,
                    target,
                ),
            )
        )

    stages.append(("final-target", clone_pose(target)))

    # Avoid issuing no-op stages, which some controllers report as not started.
    filtered = []
    previous = current
    for name, pose in stages:
        if pose_distance_mm(previous, pose) <= 0.05 and all(
            abs(a - b) <= 1e-6 for a, b in zip(previous.rpy, pose.rpy)
        ):
            continue
        filtered.append((name, pose))
        previous = pose
    return filtered


def is_singularity_error(value) -> bool:
    text = str(value)
    return "-50102" in text or "穿越奇异点" in text


def read_current_pose(robot, ec: dict):
    # 读取当前末端位姿，后续会先显示给操作者确认。
    return sdk_call(
        "cartPosture",
        ec,
        robot.cartPosture,
        xCoreSDK_python.CoordinateType.endInRef,
    )


def parse_target_pose_input(text: str, current):
    normalized = (
        text.replace("，", ",")
        .replace("：", "=")
        .replace(":", "=")
        .strip()
    )

    if "=" in normalized:
        labeled_values = {}
        raw_items = [item.strip() for item in normalized.split(",") if item.strip()]
        for item in raw_items:
            if "=" not in item:
                raise ValueError(
                    "带参数名输入时，请使用类似 X=0.1,Y=0.2,Z=0.3,RZ=180 的格式。"
                )
            key, value = item.split("=", 1)
            key = key.strip().upper()
            if key not in {"X", "Y", "Z", "RX", "RY", "RZ"}:
                raise ValueError(f"不支持的位姿字段：{key}")
            labeled_values[key] = float(value.strip())

        required_keys = {"X", "Y", "Z", "RZ"}
        if not required_keys.issubset(labeled_values):
            missing = ", ".join(sorted(required_keys - set(labeled_values)))
            raise ValueError(f"缺少必要字段：{missing}")

        x = labeled_values["X"]
        y = labeled_values["Y"]
        z = labeled_values["Z"]
        rz_deg = labeled_values["RZ"]
        rx_rad = (
            math.radians(labeled_values["RX"])
            if "RX" in labeled_values
            else current.rpy[0]
        )
        ry_rad = (
            math.radians(labeled_values["RY"])
            if "RY" in labeled_values
            else current.rpy[1]
        )
    else:
        values = normalized.replace(",", " ").split()
        if len(values) not in (4, 6):
            raise ValueError(
                "目标位姿格式错误。请输入 4 个值：X Y Z RZ，"
                "或者 6 个值：X Y Z RX RY RZ。"
            )

        numbers = [float(value) for value in values]
        if len(numbers) == 4:
            x, y, z, rz_deg = numbers
            rx_rad = current.rpy[0]
            ry_rad = current.rpy[1]
        else:
            x, y, z, rx_deg, ry_deg, rz_deg = numbers
            rx_rad = math.radians(rx_deg)
            ry_rad = math.radians(ry_deg)

    # 终端输入按毫米处理，这里统一换算成 SDK 需要的米。
    target = xCoreSDK_python.CartesianPosition()

    target.trans = [x / 1000.0, y / 1000.0, z / 1000.0]
    target.rpy = [rx_rad, ry_rad, math.radians(rz_deg)]

    # 尽量沿用当前位姿的构型信息，降低逆解切换构型带来的不确定性。
    target.confData = list(current.confData)
    target.external = list(current.external)
    target.hasElbow = current.hasElbow
    target.elbow = current.elbow
    return target


def wait_until_idle(robot, ec: dict, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = sdk_call("operationState", ec, robot.operationState)
        if state in (
            xCoreSDK_python.OperationState.idle,
            xCoreSDK_python.OperationState.unknown,
        ):
            return
        time.sleep(0.1)
    raise TimeoutError(f"机器人在 {timeout_s:.1f} 秒内未完成运动。")


def prompt_for_move(current, args: argparse.Namespace) -> None:
    print(f"SDK 目录   : {SDK_ROOT}")
    print(f"机器人 IP  : {args.ip}")
    print(f"运动方式   : {args.motion}")
    print(f"运动速度   : {args.speed} mm/s")
    print(f"转弯区     : {args.zone} mm")
    print(f"当前位姿   : {format_pose(current.trans, current.rpy)}")
    while True:
        answer = input("输入 M 开始运动，输入 q 退出：").strip()
        if not answer:
            continue
        if answer.lower() == "q":
            raise SystemExit("已退出程序。")
        if answer.upper() == "M":
            return
        print("机械臂未使能，无法运动。请先输入 M，再输入目标位姿。")


def prompt_for_target_pose(current):
    print("请输入目标位姿：")
    print("格式1: X Y Z RZ")
    print("格式2: X Y Z RX RY RZ")
    print("单位: XYZ=mm, RX/RY/RZ=deg")
    while True:
        text = input("目标位姿(q退出)：").strip()
        if not text:
            continue
        if text.lower() == "q":
            raise SystemExit("已退出程序。")
        try:
            return parse_target_pose_input(text, current)
        except Exception as exc:
            print(f"输入格式错误：{exc}")


def precheck_target_pose(robot, ec: dict, target) -> None:
    # 先做一次逆解检查，尽早发现目标位姿不可达的问题。
    # 但部分机型的模型库并不开放，这种情况下跳过预检查，继续走运动指令本体。
    try:
        toolset = xCoreSDK_python.Toolset()
        sdk_call("calcIk", ec, robot.model().calcIk, target, toolset)
    except Exception as exc:
        message = str(exc)
        if "模型库暂不支持的机型" in message or "model" in message.lower():
            print("提示：当前机型不支持模型库逆解预检查，已跳过预检查，直接尝试执行运动指令。")
            return
        raise


def query_move_event(robot, ec: dict) -> dict:
    try:
        return sdk_call("queryEventInfo", ec, robot.queryEventInfo, xCoreSDK_python.Event.moveExecution)
    except Exception:
        return {}


def prepare_robot(robot, ec: dict, args: argparse.Namespace) -> None:
    # 使用非实时模式，后续只需要电脑能访问机器人 IP 即可，不依赖实时双网卡链路。
    # TCP以太网连接机械臂控制柜。  
    sdk_call("connectToRobot", ec, robot.connectToRobot)
    # 设置机器人运行模式：自动模式（远端程序控制）。
    sdk_call(
        "setOperateMode",
        ec,
        robot.setOperateMode,
        xCoreSDK_python.OperateMode.automatic,
    )
    sdk_call(
        "setMotionControlMode",
        ec,
        robot.setMotionControlMode,
        xCoreSDK_python.MotionControlMode.NrtCommandMode,
    )
    sdk_call("setPowerState", ec, robot.setPowerState, True)
    sdk_call("setDefaultSpeed", ec, robot.setDefaultSpeed, args.speed)
    # Safe-route stages must stop at each waypoint.
    sdk_call("setDefaultZone", ec, robot.setDefaultZone, 0.0)


def _move_to_target_single_stage(robot, ec: dict, args: argparse.Namespace, target) -> str:
    # MoveJ 更适合“从任意起始姿态到目标位姿”；
    # MoveL 会走末端直线，但对起始姿态、可达性和路径空间要求更高。
    precheck_target_pose(robot, ec, target)
    sdk_call("moveReset", ec, robot.moveReset)

    # Singularity-avoidance routing uses MoveJ and stops at every stage.
    command = xCoreSDK_python.MoveJCommand(target, args.speed, 0.0)

    cmd_id = xCoreSDK_python.PyString()
    sdk_call("moveAppend", ec, robot.moveAppend, [command], cmd_id)
    sdk_call("moveStart", ec, robot.moveStart)

    started = False
    deadline = time.time() + min(args.timeout, 3.0)
    while time.time() < deadline:
        state = sdk_call("operationState", ec, robot.operationState)
        if state in (
            xCoreSDK_python.OperationState.moving,
            xCoreSDK_python.OperationState.jogging,
        ):
            started = True
            break
        time.sleep(0.05)

    wait_until_idle(robot, ec, args.timeout)

    event_info = query_move_event(robot, ec)
    final_pose = read_current_pose(robot, ec)
    pos_error_mm = pose_distance_mm(final_pose, target)

    if event_info:
        event_cmd_id = str(event_info.get(MOVE_EVENT_ID_KEY, ""))
        event_reach_target = event_info.get(MOVE_EVENT_REACH_TARGET_KEY)
        event_error = event_info.get(MOVE_EVENT_ERROR_KEY)
        event_remark = str(event_info.get(MOVE_EVENT_REMARK_KEY, ""))
    else:
        event_cmd_id = ""
        event_reach_target = None
        event_error = None
        event_remark = ""

    if not started and pos_error_mm > 2.0:
        raise RuntimeError(
            "机器人未检测到开始运动，且最终位姿没有接近目标位姿。\n"
            f"最终位置误差：{pos_error_mm:.2f} mm\n"
            f"运动事件 cmdID：{event_cmd_id}\n"
            f"运动事件 reachTarget：{event_reach_target}\n"
            f"运动事件 error：{event_error}\n"
            f"运动事件 remark：{event_remark}"
        )

    if event_reach_target is False and pos_error_mm > 2.0:
        raise RuntimeError(
            "机器人未到达目标位姿。\n"
            f"最终位置误差：{pos_error_mm:.2f} mm\n"
            f"运动事件 error：{event_error}\n"
            f"运动事件 remark：{event_remark}"
        )

    return cmd_id.content()


def move_to_target(robot, ec: dict, args: argparse.Namespace, target) -> str:
    current = read_current_pose(robot, ec)
    stages = build_safe_route(current, target, args)
    print("Safe MoveJ route (zone=0 at every stage):")
    for index, (stage_name, stage_target) in enumerate(stages, start=1):
        print(f"  {index}. {stage_name}: {format_pose(stage_target.trans, stage_target.rpy)}")

    last_cmd_id = ""
    for index, (stage_name, stage_target) in enumerate(stages, start=1):
        print(f"Executing stage {index}/{len(stages)}: {stage_name}")
        try:
            last_cmd_id = _move_to_target_single_stage(robot, ec, args, stage_target)
        except Exception as exc:
            if is_singularity_error(exc) or is_singularity_error(ec):
                raise RuntimeError(
                    f"Stage '{stage_name}' failed: xCore ec -50102 / 穿越奇异点. "
                    "No later stage was executed.\n"
                    f"Stage target: {format_pose(stage_target.trans, stage_target.rpy)}\n"
                    "Try a higher --transit-safe-z, a larger --pre-approach-height, "
                    "or first jog the robot to a safer joint configuration."
                ) from exc
            raise RuntimeError(
                f"Stage '{stage_name}' failed; no later stage was executed: {exc}"
            ) from exc
        print(f"Completed stage {index}/{len(stages)}: {stage_name}, command id={last_cmd_id}")
    return last_cmd_id


def shutdown_robot(robot, ec: dict) -> None:
    try:
        clear_error(ec)
        robot.disconnectFromRobot(ec)
    except Exception:
        pass


def main() -> int:
    args = parse_args()

    robot = xCoreSDK_python.xMateRobot(args.ip)
    ec: dict = {}

    try:
        prepare_robot(robot, ec, args)
        while True:
            current = read_current_pose(robot, ec)
            prompt_for_move(current, args)
            target = prompt_for_target_pose(current)
            print(f"目标位姿   : {format_pose(target.trans, target.rpy)}")
            cmd_id = move_to_target(robot, ec, args, target)
            final_pose = sdk_call(
                "cartPosture",
                ec,
                robot.cartPosture,
                xCoreSDK_python.CoordinateType.endInRef,
            )
            print(f"运动完成   : command id = {cmd_id}")
            print(f"最终位姿   : {format_pose(final_pose.trans, final_pose.rpy)}")
            print("程序继续运行，可输入下一组目标位姿。")
        return 0
    except Exception as exc:
        print(f"错误：{exc}")
        return 1
    finally:
        shutdown_robot(robot, ec)


if __name__ == "__main__":
    raise SystemExit(main())
