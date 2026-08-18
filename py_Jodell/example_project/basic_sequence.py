from __future__ import annotations

import argparse
import json

from jodell_controller import JodellController, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="钧舵 SDK 最小示例流程")
    parser.add_argument("--config", help="配置文件路径，默认读取同目录下的 config.json")
    args = parser.parse_args()

    config = load_config(args.config)

    with JodellController(config) as controller:
        print(f"已连接: model={config.model}, port={config.port}, slave_id={config.slave_id}")

        if config.model in {"epg", "hepg", "erg32"}:
            controller.enable(True)
            print("夹爪已使能")

            if config.model == "erg32":
                controller.move(position=100, speed=80, torque=60)
                print("已执行 ERG32 夹持示例动作")
            else:
                controller.move(position=100, speed=80, torque=60)
                print("已执行夹持示例动作")

        elif config.model == "erg26":
            controller.rotate_erg26(angle=360, speed=80, torque=60)
            print("已执行 ERG26 旋转示例动作")

        elif config.model == "evs":
            print("EVS 机型请按现场工艺参数调用 SDK 原生 runWithParam()。")

        snapshot = controller.status()
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
