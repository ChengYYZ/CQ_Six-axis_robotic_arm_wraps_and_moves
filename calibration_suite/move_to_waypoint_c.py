from project0714_calib.xcore_robot import MotionOptions, XCoreRobotClient


ROBOT_IP = "192.168.2.160"
TCP_XYZ_MM = (-125.370, 87.591, 279.781)
TCP_RPY_DEG = (178.740, -32.430, -87.920)

# The same waypoint C used by surface_cluster_grasp.py.
WAYPOINT_C = (778.995, 366.266, 515.047, -0.422, 1.598, -2.162)

def main() -> None:
    print(
        "Waypoint C: "
        f"XYZ(mm)={list(WAYPOINT_C[:3])} "
        f"RPY(deg)={list(WAYPOINT_C[3:])}"
    )
    if input("确认路径及工作区安全后，输入 MOVE：").strip() != "MOVE":
        print("已取消，机械臂未运动。")
        return

    options = MotionOptions(
        motion="movej",
        speed_mm_s=200.0,
        zone_mm=0.0,
        timeout_s=60.0,
        use_current_conf_data=False,
    )

    with XCoreRobotClient(ROBOT_IP) as robot:
        robot.prepare_motion(options.speed_mm_s, options.zone_mm)
        robot.set_toolset_with_tcp_override(
            "tool4",
            "wobj0",
            TCP_XYZ_MM,
            TCP_RPY_DEG,
        )

        current = robot.read_current_pose()
        print(
            f"当前位姿: XYZ(mm)={current.translation_mm().round(3).tolist()} "
            f"RPY(deg)={current.rpy_deg_xyz().round(3).tolist()}"
        )

        final_pose = robot.move_to_pose_mm_deg(*WAYPOINT_C, options=options)
        print(
            f"已到达 C: XYZ(mm)={final_pose.translation_mm().round(3).tolist()} "
            f"RPY(deg)={final_pose.rpy_deg_xyz().round(3).tolist()}"
        )


if __name__ == "__main__":
    main()
