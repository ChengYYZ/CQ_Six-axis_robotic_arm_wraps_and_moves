from __future__ import annotations

import argparse

import numpy as np

from project0714_calib.common import invert_transform, make_transform, matrix_to_rpy_xyz, rpy_xyz_to_matrix
from project0714_calib.xcore_robot import ToolsetInfo, XCoreRobotClient


def tool_transform(tool: ToolsetInfo) -> np.ndarray:
    return make_transform(rpy_xyz_to_matrix(tool.end_rpy_rad_xyz), tool.end_translation_m)


def print_tool(name: str, tool: ToolsetInfo) -> None:
    print(f"{name}:")
    print(f"  XYZ(mm) = {tool.end_translation_mm().round(3).tolist()}")
    print(f"  RPY(deg) = {tool.end_rpy_deg_xyz().round(3).tolist()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two controller TCP definitions without robot motion.")
    parser.add_argument("--robot-ip", default="192.168.2.160")
    parser.add_argument("--tool-a", default="tool4")
    parser.add_argument("--tool-b", default="tool4")
    parser.add_argument("--wobj-name", default="wobj0")
    parser.add_argument("--restore-tool", default="tool4")
    args = parser.parse_args()

    robot = XCoreRobotClient(args.robot_ip)
    try:
        robot.connect()
        tool_a = robot.set_toolset_by_name(args.tool_a, args.wobj_name)
        tool_b = robot.set_toolset_by_name(args.tool_b, args.wobj_name)
        print_tool(args.tool_a, tool_a)
        print_tool(args.tool_b, tool_b)

        transform_a = tool_transform(tool_a)
        transform_b = tool_transform(tool_b)
        delta = invert_transform(transform_a) @ transform_b
        delta_xyz_mm = delta[:3, 3] * 1000.0
        delta_rpy_deg = np.degrees(matrix_to_rpy_xyz(delta[:3, :3]))
        print(f"{args.tool_a} -> {args.tool_b} relative difference:")
        print(f"  delta XYZ(mm) = {delta_xyz_mm.round(3).tolist()}")
        print(f"  delta RPY(deg) = {delta_rpy_deg.round(3).tolist()}")
    finally:
        try:
            if robot.connected and args.restore_tool:
                robot.set_toolset_by_name(args.restore_tool, args.wobj_name)
                print(f"Restored active tool: {args.restore_tool}")
        finally:
            robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
