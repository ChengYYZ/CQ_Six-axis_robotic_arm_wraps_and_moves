"""C 点吸盘局部 Z 轴旋转测试。默认仅规划；--execute 才连接机器人。

默认主吸盘/tool4。C 点直接 MoveJ 接近，必须从已确认无遮挡的起点运行。
仅测试运动，不抓取、不改变吸盘 IO、不拍照。角度采用吸盘 +Z 右手定则。
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from move_to_waypoint_c import ROBOT_IP, TCP_RPY_DEG, TCP_XYZ_MM
from project0714_calib.common import matrix_to_rpy_xyz, rpy_xyz_to_matrix
from project0714_calib.xcore_robot import MotionOptions, XCoreRobotClient


ANGLES_DEG = (0.0, 90.0, 0.0, -90.0, 0.0)
# User-supplied screenshot; test point only, not the production waypoint C.
# Interpreted in the existing tool4/wobj0 TCP convention; verify on controller.
TEST_WAYPOINT_C = (627.094, 311.420, 470.689, -2.136, 3.046, -3.744)
TAUGHT_SIDE_POSE = (627.067, 311.427, 470.629, -0.736, 3.143, 75.357)


def plan_targets(args: argparse.Namespace) -> list[np.ndarray]:
    origin = np.asarray(args.c_pose, dtype=np.float64)
    if args.taught_side:
        return [origin.copy(), np.asarray(TAUGHT_SIDE_POSE), origin.copy()]
    return build_targets(origin)


def step_labels(args: argparse.Namespace) -> list[str]:
    if args.taught_side:
        return ["到 C 点", "到截图示教侧面点", "返回 C 点原始位姿"]
    return ["到 C 点"] + [f"C 点相对角度 {angle:+.0f}°" for angle in ANGLES_DEG[1:]]


def build_targets(c_pose: np.ndarray) -> list[np.ndarray]:
    """固定吸盘 TCP，所有姿态相对同一个 C 点计算，不累积旋转误差。"""
    start_rotation = rpy_xyz_to_matrix(np.radians(c_pose[3:]))
    targets = []
    for angle in ANGLES_DEG:
        target = c_pose.copy()
        if angle:
            local_rotation = rpy_xyz_to_matrix(np.radians([0.0, 0.0, angle]))
            target[3:] = np.degrees(matrix_to_rpy_xyz(start_rotation @ local_rotation))
        targets.append(target)
    return targets


def cup_tcp(offset_mm: np.ndarray, mounting_rpy_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把主 TCP 到所选吸盘的偏移/安装姿态合成到临时工具 TCP。

    控制器由此直接以所选吸盘中心做 MoveL，避免只补偿端点而中途偏离旋转轴。
    """
    main_rotation = rpy_xyz_to_matrix(np.radians(TCP_RPY_DEG))
    mounting_rotation = rpy_xyz_to_matrix(np.radians(mounting_rpy_deg))
    xyz = np.asarray(TCP_XYZ_MM) + main_rotation @ offset_mm
    rpy = np.degrees(matrix_to_rpy_xyz(main_rotation @ mounting_rotation))
    return xyz, rpy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--taught-side", action="store_true",
                        help="只测试原位→截图示教侧面→原位，不生成另一侧目标")
    parser.add_argument("--c-pose", type=float, nargs=6, default=TEST_WAYPOINT_C,
                        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                        help="吸盘中心的测试位姿，单位 mm/deg，默认使用新截图点，不改变主流程 C 点")
    parser.add_argument("--cup-offset-mm", type=float, nargs=3, default=(0, 0, 0),
                        help="主 TCP 到实际吸盘中心的工具坐标偏移，默认主吸盘 0 0 0")
    parser.add_argument("--cup-rpy-deg", type=float, nargs=3, default=(0, 0, 0),
                        help="吸盘坐标系相对主 TCP 的安装姿态，默认 0 0 0")
    parser.add_argument("--speed-mm-s", type=float, default=300.0,
                        help="SDK 运动速度参数，不是角速度；默认 %(default)s mm/s")
    parser.add_argument("--hold-s", type=float, default=1.0,
                        help="C 点及正负90度确认到位后的停留时间，默认 1 秒")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--rotation-motion", choices=("movel", "movej"), default="movel",
                        help="旋转插补方式；默认 movel 固定吸盘中心。movej 中途可能偏移，强制逐步确认")
    parser.add_argument("--rotation-conf", choices=("current", "auto"), default="current",
                        help="current 保持当前构型；auto 不设置 confData，仅允许显式 MoveJ，可能改变关节构型")
    parser.add_argument("--execute", action="store_true", help="实际驱动机械臂")
    parser.add_argument("--step", action="store_true", help="每步运动前额外输入 MOVE，供首次单步验证")
    args = parser.parse_args(argv)
    if args.rotation_conf == "auto" and args.rotation_motion != "movej":
        parser.error("--rotation-conf auto 必须同时使用 --rotation-motion movej")
    values = [*args.c_pose, *args.cup_offset_mm, *args.cup_rpy_deg,
              args.speed_mm_s, args.hold_s, args.timeout_s]
    if not np.all(np.isfinite(values)):
        parser.error("所有数值必须有限，不能为 NaN 或无穷大")
    if args.speed_mm_s <= 0 or args.timeout_s <= 0 or args.hold_s < 0:
        parser.error("速度和超时必须大于 0，停留时间不能小于 0")
    return args


def confirm(message: str) -> None:
    if input(message + "\n输入 MOVE 继续，其他输入取消：").strip() != "MOVE":
        raise KeyboardInterrupt("操作者取消")


def verify_reached(actual, target: np.ndarray) -> None:
    position_error = float(np.linalg.norm(actual.translation_mm() - target[:3]))
    actual_rotation = rpy_xyz_to_matrix(np.radians(actual.rpy_deg_xyz()))
    target_rotation = rpy_xyz_to_matrix(np.radians(target[3:]))
    cosine = np.clip((np.trace(actual_rotation.T @ target_rotation) - 1) / 2, -1, 1)
    angle_error = float(np.degrees(np.arccos(cosine)))
    if position_error > 2.0 or angle_error > 1.0:
        raise RuntimeError(f"到位误差超限：位置 {position_error:.3f} mm，姿态 {angle_error:.3f} deg")


def execute(args: argparse.Namespace, targets: list[np.ndarray]) -> None:
    if args.rotation_motion == "movej":
        print("MoveJ 测试：仅保证目标位姿；中途吸盘中心可能偏移。"
              "不保证能够避开奇异点。每步须确认完整扫掠空间。", flush=True)
        if args.rotation_conf == "auto":
            print("已显式选择不设置 confData：控制器可能改变关节构型，产生较大的手臂/腕部运动。"
                  "回零仅指回到原始笛卡尔位姿，不保证恢复原始关节角。", flush=True)
        else:
            print("保持当前 confData，不自动放开构型约束。", flush=True)
    route = " → ".join(step_labels(args))
    confirm(f"即将执行：{route}；接近使用 MoveJ，后续使用 {args.rotation_motion}。\n"
            "请确认直达 C 的路径、完整旋转扫掠空间、工具参数及吸附状态。\n"
            "本程序不自动抓取或控制吸盘；Ctrl+C/异常时尝试软件停止，不自动回零。")
    xyz, rpy = cup_tcp(np.asarray(args.cup_offset_mm), np.asarray(args.cup_rpy_deg))
    with XCoreRobotClient(args.robot_ip) as robot:
        try:
            robot.prepare_motion(args.speed_mm_s, 0.0)
            robot.set_toolset_with_tcp_override("tool4", "wobj0", xyz, rpy)
            for index, (label, target) in enumerate(zip(step_labels(args), targets)):
                if args.step or args.rotation_motion == "movej":
                    confirm(label)
                options = MotionOptions(
                    motion="movej" if index == 0 else args.rotation_motion,
                    speed_mm_s=args.speed_mm_s, zone_mm=0.0,
                    timeout_s=args.timeout_s,
                    use_current_conf_data=(index != 0 and args.rotation_conf == "current"),
                )
                # SDK has a separate controller switch: an empty confData alone
                # does not guarantee that forced configuration solving is off.
                robot.set_conf_data_forced(options.use_current_conf_data)
                print(f"控制器已接受 setDefaultConfOpt({options.use_current_conf_data})", flush=True)
                reached = robot.move_to_pose_mm_deg(*target.tolist(), options=options)
                verify_reached(reached, target)
                print(f"已完成 {label}，实际 XYZ={reached.translation_mm().round(3).tolist()} "
                      f"RPY={reached.rpy_deg_xyz().round(3).tolist()}", flush=True)
                if index == 0 or index % 2 == 1:
                    print(f"到位后停留 {args.hold_s:g} 秒", flush=True)
                    time.sleep(args.hold_s)
        except BaseException:
            # Preserve the original failure even if the software stop also fails.
            try:
                robot.stop_motion()
            except Exception as stop_error:
                print(f"软件停止失败：{stop_error}；请使用现场停止装置。", flush=True)
            raise
    print("测试完成，已返回 C 点原始位姿。")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    targets = plan_targets(args)
    xyz, rpy = cup_tcp(np.asarray(args.cup_offset_mm), np.asarray(args.cup_rpy_deg))
    print("模式：" + ("实际运动" if args.execute else "仅规划，不连接机器人"))
    print(f"速度参数={args.speed_mm_s:g} mm/s；旋转构型策略={args.rotation_conf}")
    print(f"临时吸盘 TCP（tool4/wobj0）：XYZ={xyz.round(3).tolist()} RPY={rpy.round(3).tolist()}")
    print("请确认截图坐标使用相同的工具 TCP 和参考坐标系；本测试不会修改主流程 C 点。")
    print("角度按吸盘 +Z 右手定则；从 +Z 端看向原点，+90° 为逆时针。")
    for index, (label, target) in enumerate(zip(step_labels(args), targets)):
        hold = args.hold_s if index == 0 or index % 2 == 1 else 0.0
        print(f"{index + 1}. {'movej' if index == 0 else args.rotation_motion} {label} "
              f"XYZ={target[:3].round(3).tolist()} RPY={target[3:].round(3).tolist()} "
              f"到位后停留={hold:g}s")
    if not args.execute:
        print("检查规划后，使用 --execute 实际运行；首次可增加 --step。")
        return 0
    try:
        execute(args, targets)
    except (KeyboardInterrupt, EOFError):
        print("测试已取消；未自动回 C 点。")
        return 2
    except Exception as exc:
        print(f"测试失败，后续动作已取消，未自动回 C 点：{exc}")
        if "50021" in str(exc):
            if args.rotation_conf == "current":
                print("当前指定构型下目标无解。可在现场验证空间后，显式选择 "
                      "--rotation-motion movej --rotation-conf auto，不自动重试。")
            else:
                print("已请求关闭强制构型求解，目标仍被拒绝；需检查工具参数、目标可达性并重新示教观察点/中转路线。")
        if "50002" in str(exc):
            print("目标点超出运动范围或为奇异点；不能仅凭此错误区分原因，需检查目标及工具坐标。")
        elif "50102" in str(exc):
            print("控制器拒绝穿越奇异点的路径；未自动切换插补方式或关节构型。\n"
                  "下一步需验证其他插补路径或重新示教 C 点/中转位姿。\n"
                  "--rotation-motion movej 可用于现场逐步验证，但中途不保证吸盘中心固定。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
