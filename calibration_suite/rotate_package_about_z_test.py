"""独立测试：让指定空间点处的包裹绕基座 Z 轴分步旋转。

默认仅打印规划，不连接或驱动机器人。只有显式传入 --execute 后才会运动。
旋转过程中固定的是吸盘/包裹中心；当使用偏置副吸盘时，TCP 会自动作圆周补偿。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from project0714_calib.common import matrix_to_rpy_xyz, rpy_xyz_to_matrix
from project0714_calib.xcore_robot import MotionOptions, XCoreRobotClient


DEFAULT_ROBOT_IP = "192.168.2.160"
DEFAULT_POINT_XYZ_MM = np.array([502.637, 702.366, 345.205], dtype=np.float64)
DEFAULT_POINT_RPY_DEG = np.array([-4.157, 0.654, 87.041], dtype=np.float64)
DEFAULT_POINT_90_XYZ_MM = np.array([570.560, 768.371, 384.312], dtype=np.float64)
DEFAULT_POINT_90_RPY_DEG = np.array([-1.061, 4.339, -2.236], dtype=np.float64)
DEFAULT_SAFE_XYZ_MM = np.array([225.733, 550.238, 535.240], dtype=np.float64)
DEFAULT_SAFE_RPY_DEG = np.array([-2.080, 1.287, 83.579], dtype=np.float64)
DEFAULT_TCP_XYZ_MM = (-125.370, 87.591, 279.781)
DEFAULT_TCP_RPY_DEG = (178.740, -32.430, -87.920)


@dataclass(frozen=True)
class RotationTarget:
    angle_deg: float
    tcp_xyz_mm: np.ndarray
    tcp_rpy_deg: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在固定吸盘/包裹中心处，让包裹绕基座 Z 轴分步旋转。"
    )
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--tool-name", default="tool4")
    parser.add_argument("--wobj-name", default="wobj0")
    parser.add_argument(
        "--point-xyz-mm",
        nargs=3,
        type=float,
        default=DEFAULT_POINT_XYZ_MM.tolist(),
        metavar=("X", "Y", "Z"),
        help="截图中的检测点 TCP 坐标。",
    )
    parser.add_argument(
        "--point-rpy-deg",
        nargs=3,
        type=float,
        default=DEFAULT_POINT_RPY_DEG.tolist(),
        metavar=("RX", "RY", "RZ"),
        help="截图中的检测点姿态，按 xCore XYZ-RPY 约定解释。",
    )
    parser.add_argument(
        "--angles-deg",
        nargs="+",
        type=float,
        default=[0.0, 90.0, 0.0],
        help="相对角度序列。当前默认只验证两个示教可达点：0、90、0。",
    )
    parser.add_argument(
        "--point-90-xyz-mm",
        nargs=3,
        type=float,
        default=DEFAULT_POINT_90_XYZ_MM.tolist(),
        metavar=("X", "Y", "Z"),
        help="手动示教的第二侧面（90度）TCP 坐标。",
    )
    parser.add_argument(
        "--point-90-rpy-deg",
        nargs=3,
        type=float,
        default=DEFAULT_POINT_90_RPY_DEG.tolist(),
        metavar=("RX", "RY", "RZ"),
        help="手动示教的第二侧面（90度）姿态。",
    )
    parser.add_argument(
        "--safe-xyz-mm",
        nargs=3,
        type=float,
        default=DEFAULT_SAFE_XYZ_MM.tolist(),
        metavar=("X", "Y", "Z"),
        help="收回并旋转腕部的安全点坐标。",
    )
    parser.add_argument(
        "--safe-rpy-deg",
        nargs=3,
        type=float,
        default=DEFAULT_SAFE_RPY_DEG.tolist(),
        metavar=("RX", "RY", "RZ"),
        help="安全点的手动示教起始姿态。",
    )
    parser.add_argument(
        "--pivot-offset-tool-mm",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="从 TCP 到实际吸盘/包裹旋转中心的工具坐标偏移；主吸盘为 0 0 0。",
    )
    parser.add_argument("--speed-mm-s", type=float, default=80.0)
    parser.add_argument("--settle-s", type=float, default=0.6)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际连接并驱动机器人；不传此参数时仅打印规划。",
    )
    parser.add_argument(
        "--skip-start-move",
        action="store_true",
        help="兼容旧命令；现在默认就以实时读取姿态作为 0 度基准。",
    )
    parser.add_argument(
        "--move-to-start",
        action="store_true",
        help="显式允许先自动移动到截图中的第一侧面点；默认不执行该移动。",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过每一步 MOVE 确认，仅建议在仿真或首次单步验证完成后使用。",
    )
    return parser.parse_args()


def base_z_rotation(angle_deg: float) -> np.ndarray:
    angle_rad = np.radians(float(angle_deg))
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def build_targets(
    start_tcp_xyz_mm: np.ndarray,
    start_tcp_rpy_deg: np.ndarray,
    pivot_offset_tool_mm: np.ndarray,
    angles_deg: list[float],
) -> tuple[np.ndarray, list[RotationTarget]]:
    """固定空间中的 pivot，并以 Rz_base(angle) @ R_start 生成姿态。"""
    start_rotation = rpy_xyz_to_matrix(np.radians(start_tcp_rpy_deg))
    pivot_xyz_mm = start_tcp_xyz_mm + start_rotation @ pivot_offset_tool_mm
    targets: list[RotationTarget] = []

    for angle_deg in angles_deg:
        target_rotation = base_z_rotation(angle_deg) @ start_rotation
        target_tcp_xyz_mm = pivot_xyz_mm - target_rotation @ pivot_offset_tool_mm
        target_rpy_deg = np.degrees(matrix_to_rpy_xyz(target_rotation))
        targets.append(
            RotationTarget(float(angle_deg), target_tcp_xyz_mm, target_rpy_deg)
        )
    return pivot_xyz_mm, targets


def print_plan(pivot_xyz_mm: np.ndarray, targets: list[RotationTarget]) -> None:
    print(f"固定吸盘/包裹中心 XYZ(mm): {pivot_xyz_mm.round(3).tolist()}")
    for index, target in enumerate(targets, start=1):
        print(
            f"  {index}. angle={target.angle_deg:+.1f} deg "
            f"TCP XYZ(mm)={target.tcp_xyz_mm.round(3).tolist()} "
            f"RPY(deg)={target.tcp_rpy_deg.round(3).tolist()}"
        )


def apply_taught_90_target(
    targets: list[RotationTarget],
    point_90_xyz_mm: np.ndarray,
    point_90_rpy_deg: np.ndarray,
) -> list[RotationTarget]:
    """Use the verified reachable taught pose instead of an idealized 90-degree pose."""
    result: list[RotationTarget] = []
    for target in targets:
        normalized = target.angle_deg % 360.0
        if np.isclose(normalized, 90.0, atol=1e-6):
            result.append(
                RotationTarget(
                    angle_deg=target.angle_deg,
                    tcp_xyz_mm=point_90_xyz_mm.copy(),
                    tcp_rpy_deg=point_90_rpy_deg.copy(),
                )
            )
        else:
            result.append(target)
    return result


def require_confirmation(message: str, assume_yes: bool) -> None:
    if assume_yes:
        return
    if input(f"{message}\n确认安全后输入 MOVE，其他输入取消：").strip() != "MOVE":
        raise KeyboardInterrupt("操作者取消。")


def move_confirmed(
    robot: XCoreRobotClient,
    options: MotionOptions,
    xyz_mm: np.ndarray,
    rpy_deg: np.ndarray,
    label: str,
    assume_yes: bool,
):
    require_confirmation(
        f"即将移动到 {label}: XYZ(mm)={xyz_mm.round(3).tolist()} "
        f"RPY(deg)={rpy_deg.round(3).tolist()}",
        assume_yes,
    )
    reached = robot.move_to_pose_mm_deg(
        *xyz_mm.tolist(), *rpy_deg.tolist(), options=options
    )
    print(
        f"已到达 {label}: XYZ(mm)={reached.translation_mm().round(3).tolist()} "
        f"RPY(deg)={reached.rpy_deg_xyz().round(3).tolist()}"
    )
    return reached


def execute(args: argparse.Namespace) -> None:
    requested_xyz = np.asarray(args.point_xyz_mm, dtype=np.float64)
    requested_rpy = np.asarray(args.point_rpy_deg, dtype=np.float64)
    offset_tool = np.asarray(args.pivot_offset_tool_mm, dtype=np.float64)
    options = MotionOptions(
        motion="movej",
        speed_mm_s=float(args.speed_mm_s),
        zone_mm=0.0,
        timeout_s=float(args.timeout_s),
        # A 90-degree Cartesian yaw often requires a different wrist/joint
        # configuration. Reusing the starting confData makes xCore reject an
        # otherwise reachable target with ec=-50021.
        use_current_conf_data=False,
    )

    with XCoreRobotClient(args.robot_ip) as robot:
        robot.prepare_motion(options.speed_mm_s, options.zone_mm)
        robot.set_toolset_with_tcp_override(
            args.tool_name,
            args.wobj_name,
            DEFAULT_TCP_XYZ_MM,
            DEFAULT_TCP_RPY_DEG,
        )
        current = robot.read_current_pose()
        print(
            f"当前 TCP: XYZ(mm)={current.translation_mm().round(3).tolist()} "
            f"RPY(deg)={current.rpy_deg_xyz().round(3).tolist()}"
        )

        if args.move_to_start and not args.skip_start_move:
            require_confirmation(
                "即将以低速 MoveJ 移动到截图点。请先确认路径、工具、工件和工作区安全。",
                args.yes,
            )
            reached = robot.move_to_pose_mm_deg(
                *requested_xyz.tolist(), *requested_rpy.tolist(), options=options
            )
            start_xyz = reached.translation_mm()
            start_rpy = reached.rpy_deg_xyz()
            print(
                f"已到达测试点: XYZ(mm)={start_xyz.round(3).tolist()} "
                f"RPY(deg)={start_rpy.round(3).tolist()}"
            )
        else:
            start_xyz = current.translation_mm()
            start_rpy = current.rpy_deg_xyz()
            print("使用机器人当前实际可达位姿作为第一侧面/0度基准。")

        pivot_xyz, targets = build_targets(
            start_xyz, start_rpy, offset_tool, list(args.angles_deg)
        )
        targets = apply_taught_90_target(
            targets,
            np.asarray(args.point_90_xyz_mm, dtype=np.float64),
            np.asarray(args.point_90_rpy_deg, dtype=np.float64),
        )
        print_plan(pivot_xyz, targets)
        print("90°步骤使用手动示教的可达点；目标不锁定起始 confData。")

        safe_xyz = np.asarray(args.safe_xyz_mm, dtype=np.float64)
        safe_rpy_0 = np.asarray(args.safe_rpy_deg, dtype=np.float64)
        safe_rotation_0 = rpy_xyz_to_matrix(np.radians(safe_rpy_0))
        # The taught second face changes base yaw from about +87 to -2 deg,
        # so side number +90 corresponds to a physical base-Z turn of -90 deg.
        safe_rotation_90 = base_z_rotation(-90.0) @ safe_rotation_0
        safe_rpy_90 = np.degrees(matrix_to_rpy_xyz(safe_rotation_90))
        print(
            "安全中转路线已启用："
            f"S0 RPY={safe_rpy_0.round(3).tolist()} -> "
            f"S90 RPY={safe_rpy_90.round(3).tolist()}"
        )

        import time

        # The current pose is already face 0. Do not send a no-op Cartesian
        # command, because the controller may retain an earlier move error.
        print("第一侧面使用当前实际位姿，等待稳定。")
        time.sleep(max(0.0, float(args.settle_s)))

        move_confirmed(robot, options, safe_xyz, safe_rpy_0, "安全点 S0", args.yes)
        move_confirmed(robot, options, safe_xyz, safe_rpy_90, "安全点 S90", args.yes)
        face_90 = next(item for item in targets if np.isclose(item.angle_deg % 360.0, 90.0))
        # Preserve the orientation that was actually verified at S90.  Moving
        # to the camera changes translation only; restoring the old A/photo
        # orientation here would undo the requested side rotation.
        face_90_camera_xyz = face_90.tcp_xyz_mm
        face_90_camera_rpy = safe_rpy_90
        move_confirmed(
            robot, options, face_90_camera_xyz, face_90_camera_rpy,
            "第二侧面拍照点", args.yes,
        )
        time.sleep(max(0.0, float(args.settle_s)))

        move_confirmed(robot, options, safe_xyz, safe_rpy_90, "返回安全点 S90", args.yes)
        move_confirmed(robot, options, safe_xyz, safe_rpy_0, "返回安全点 S0", args.yes)
        move_confirmed(robot, options, start_xyz, start_rpy, "返回第一侧面拍照点", args.yes)
        time.sleep(max(0.0, float(args.settle_s)))


def main() -> int:
    args = parse_args()
    if args.speed_mm_s <= 0.0 or args.timeout_s <= 0.0 or args.settle_s < 0.0:
        raise SystemExit("速度和超时必须为正数，稳定等待时间不能为负数。")

    point_xyz = np.asarray(args.point_xyz_mm, dtype=np.float64)
    point_rpy = np.asarray(args.point_rpy_deg, dtype=np.float64)
    offset = np.asarray(args.pivot_offset_tool_mm, dtype=np.float64)
    pivot, targets = build_targets(point_xyz, point_rpy, offset, list(args.angles_deg))
    targets = apply_taught_90_target(
        targets,
        np.asarray(args.point_90_xyz_mm, dtype=np.float64),
        np.asarray(args.point_90_rpy_deg, dtype=np.float64),
    )
    print("测试模式：" + ("实际运动" if args.execute else "仅规划，不连接机器人"))
    print_plan(pivot, targets)

    if not args.execute:
        print("未传入 --execute，机器人不会运动。")
        return 0
    try:
        execute(args)
        return 0
    except KeyboardInterrupt as exc:
        print(f"测试已取消：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
