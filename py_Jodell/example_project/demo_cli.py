from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jodell_controller import JodellConfig, JodellController, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="钧舵 Python SDK 示例命令行")
    parser.add_argument("--config", help="配置文件路径，默认读取同目录下的 config.json")
    parser.add_argument("--model", help="临时覆盖配置中的机型")
    parser.add_argument("--port", help="临时覆盖配置中的串口")
    parser.add_argument("--baud-rate", type=int, help="临时覆盖配置中的波特率")
    parser.add_argument("--slave-id", type=int, help="临时覆盖配置中的从站 ID")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ports", help="列出本机可用串口")

    status_parser = subparsers.add_parser("status", help="查询设备状态")
    status_parser.add_argument("--pretty", action="store_true", help="格式化输出 JSON")

    enable_parser = subparsers.add_parser("enable", help="夹爪使能或去使能")
    enable_parser.add_argument("state", choices=["on", "off"])

    move_parser = subparsers.add_parser("move", help="夹持端有参运动")
    move_parser.add_argument("position", type=int)
    move_parser.add_argument("speed", type=int)
    move_parser.add_argument("torque", type=int)

    preset_parser = subparsers.add_parser("preset", help="执行夹持端预设动作")
    preset_parser.add_argument("cmd_id", type=int)

    rotate_enable_parser = subparsers.add_parser("rotate-enable", help="ERG32 旋转端使能")
    rotate_enable_parser.add_argument("state", choices=["on", "off"])

    rotate_parser = subparsers.add_parser("rotate", help="ERG32 旋转端有参运动")
    rotate_parser.add_argument("angle", type=int)
    rotate_parser.add_argument("speed", type=int)
    rotate_parser.add_argument("torque", type=int)
    rotate_parser.add_argument("--absolute", action="store_true", help="按绝对位置模式旋转")
    rotate_parser.add_argument("--cycle-num", type=int, default=0, help="绝对位置模式下的圈数")

    rotate_preset_parser = subparsers.add_parser("rotate-preset", help="ERG32 旋转端预设动作")
    rotate_preset_parser.add_argument("cmd_id", type=int)

    erg26_rotate_parser = subparsers.add_parser("rotate-erg26", help="ERG26 旋转运动")
    erg26_rotate_parser.add_argument("angle", type=int)
    erg26_rotate_parser.add_argument("speed", type=int)
    erg26_rotate_parser.add_argument("torque", type=int)

    raw_status_parser = subparsers.add_parser("raw-status", help="读取原始寄存器")
    raw_status_parser.add_argument("address", type=int)
    raw_status_parser.add_argument("read_mode", type=int)
    raw_status_parser.add_argument("count", type=int)

    return parser


def build_runtime_config(args: argparse.Namespace) -> JodellConfig:
    config = load_config(args.config)
    if args.model:
        config.model = args.model.lower()
    if args.port:
        config.port = args.port
    if args.baud_rate:
        config.baud_rate = args.baud_rate
    if args.slave_id:
        config.slave_id = args.slave_id
    return config


def dump_result(result: Any, pretty: bool = False) -> None:
    if pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ports":
        dump_result(JodellController.list_ports(), pretty=True)
        return

    config = build_runtime_config(args)

    if args.command == "status":
        with JodellController(config) as controller:
            dump_result(controller.status(), pretty=args.pretty)
        return

    with JodellController(config) as controller:
        if args.command == "enable":
            controller.enable(args.state == "on")
            print("OK")
        elif args.command == "move":
            controller.enable(True)
            controller.move(args.position, args.speed, args.torque)
            print("OK")
        elif args.command == "preset":
            controller.enable(True)
            controller.run_preset(args.cmd_id)
            print("OK")
        elif args.command == "rotate-enable":
            controller.rotate_enable(args.state == "on")
            print("OK")
        elif args.command == "rotate":
            controller.rotate_enable(True)
            controller.rotate(
                args.angle,
                args.speed,
                args.torque,
                absolute=args.absolute,
                cycle_num=args.cycle_num,
            )
            print("OK")
        elif args.command == "rotate-preset":
            controller.rotate_enable(True)
            controller.rotate_preset(args.cmd_id)
            print("OK")
        elif args.command == "rotate-erg26":
            controller.rotate_erg26(args.angle, args.speed, args.torque)
            print("OK")
        elif args.command == "raw-status":
            dump_result(
                controller.raw_status(args.address, args.read_mode, args.count),
                pretty=True,
            )
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
