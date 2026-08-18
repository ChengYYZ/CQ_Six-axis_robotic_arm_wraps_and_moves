from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type


def _ensure_local_wheel_on_path() -> None:
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[1]
    wheel_path = project_root / "python版本" / "JodellTool-0.0.1-py3-none-any.whl"
    wheel_text = str(wheel_path)
    if wheel_path.exists() and wheel_text not in sys.path:
        sys.path.insert(0, wheel_text)


_ensure_local_wheel_on_path()

try:
    from jodellSdk.jodellSdkDemo import (  # type: ignore
        ClawEpgTool,
        ClawErgTool,
        ClawErgTool2,
        ClawEvsTool,
        ClawHepgTool,
        JodellSDKDemo,
    )
except ImportError as exc:  # pragma: no cover - import error message only
    raise ImportError(
        "无法导入钧舵 SDK。请先安装 pyserial、modbus-tk 和 JodellTool wheel 包。"
    ) from exc


ToolType = Type[JodellSDKDemo]


MODEL_CLASS_MAP: Dict[str, ToolType] = {
    "epg": ClawEpgTool,
    "hepg": ClawHepgTool,
    "evs": ClawEvsTool,
    "erg32": ClawErgTool,
    "erg26": ClawErgTool2,
}


ROTATION_MODELS = {"erg32", "erg26"}
ROTATION_ENABLE_MODELS = {"erg32"}
GRIP_MODELS = {"epg", "hepg", "erg32"}
STATUS_FRIENDLY_MODELS = {"epg", "hepg", "erg32"}


@dataclass
class JodellConfig:
    model: str
    port: str
    baud_rate: int
    slave_id: int

    @classmethod
    def from_file(cls, path: str | Path) -> "JodellConfig":
        config_path = Path(path)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(
            model=str(data["model"]).lower(),
            port=str(data["port"]),
            baud_rate=int(data["baud_rate"]),
            slave_id=int(data["slave_id"]),
        )


class JodellController:
    def __init__(self, config: JodellConfig):
        self.config = config
        self.tool = self._build_tool(config.model)
        self.connected = False

    def __enter__(self) -> "JodellController":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def available_models() -> List[str]:
        return sorted(MODEL_CLASS_MAP)

    @staticmethod
    def list_ports() -> List[str]:
        tool = JodellSDKDemo()
        return tool.searchCom()

    def _build_tool(self, model: str) -> JodellSDKDemo:
        normalized = model.lower()
        if normalized not in MODEL_CLASS_MAP:
            supported = ", ".join(self.available_models())
            raise ValueError(f"不支持的机型: {model}。支持的机型: {supported}")

        tool_cls = MODEL_CLASS_MAP[normalized]
        tool = tool_cls()

        # 原始 SDK 中多个子类没有调用父类初始化，这里补上基础状态，避免查询时缺属性。
        if not hasattr(tool, "clampStatus"):
            JodellSDKDemo.__init__(tool)

        return tool

    def _check_write_result(self, result: Any, action: str) -> None:
        if result != 1:
            raise RuntimeError(f"{action}失败: {result}")

    def _ensure_model(self, allowed_models: set[str], action: str) -> None:
        if self.config.model not in allowed_models:
            model_list = ", ".join(sorted(allowed_models))
            raise ValueError(f"{action}仅支持以下机型: {model_list}")

    def open(self) -> None:
        if self.connected:
            return
        result = self.tool.serialOperation(self.config.port, self.config.baud_rate, True)
        self._check_write_result(result, "连接串口")
        self.connected = True

    def close(self) -> None:
        if not self.connected:
            return
        result = self.tool.serialOperation(self.config.port, self.config.baud_rate, False)
        self._check_write_result(result, "断开串口")
        self.connected = False

    def enable(self, enabled: bool) -> None:
        self._check_write_result(
            self.tool.clawEnable(self.config.slave_id, enabled),
            "夹爪使能" if enabled else "夹爪去使能",
        )

    def rotate_enable(self, enabled: bool) -> None:
        self._ensure_model(ROTATION_ENABLE_MODELS, "旋转使能")
        self._check_write_result(
            self.tool.rotateEnable(self.config.slave_id, enabled),
            "旋转使能" if enabled else "旋转去使能",
        )

    def move(self, position: int, speed: int, torque: int) -> None:
        self._ensure_model(GRIP_MODELS, "夹持运动")
        self._check_write_result(
            self.tool.runWithParam(self.config.slave_id, position, speed, torque),
            "夹持运动",
        )

    def run_preset(self, cmd_id: int) -> None:
        self._check_write_result(
            self.tool.runWithoutParam(self.config.slave_id, cmd_id),
            "无参预设动作",
        )

    def rotate(self, angle: int, speed: int, torque: int, absolute: bool = False, cycle_num: int = 0) -> None:
        self._ensure_model({"erg32"}, "旋转运动")
        self._check_write_result(
            self.tool.rotateWithParam(
                self.config.slave_id,
                angle,
                speed,
                torque,
                absolute,
                cycle_num,
            ),
            "旋转运动",
        )

    def rotate_preset(self, cmd_id: int) -> None:
        self._ensure_model({"erg32"}, "旋转预设动作")
        self._check_write_result(
            self.tool.rotateWithoutParam(self.config.slave_id, cmd_id),
            "旋转预设动作",
        )

    def rotate_erg26(self, angle: int, speed: int, torque: int) -> None:
        self._ensure_model({"erg26"}, "ERG26 旋转运动")
        self._check_write_result(
            self.tool.rotateWithParam(self.config.slave_id, angle, speed, torque),
            "ERG26 旋转运动",
        )

    def raw_status(self, address: int, read_mode: int, count: int) -> Any:
        return self.tool.getStatus(self.config.slave_id, address, read_mode, count)

    def status(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {
            "model": self.config.model,
            "port": self.config.port,
            "baud_rate": self.config.baud_rate,
            "slave_id": self.config.slave_id,
            "software_version": self.tool.querySoftwareVersion(self.config.slave_id),
        }

        if self.config.model in STATUS_FRIENDLY_MODELS:
            snapshot["claw_status"] = self.tool.getClawCurrentStatus(self.config.slave_id)
            snapshot["position"] = self.tool.getClawCurrentLocation(self.config.slave_id)
            snapshot["speed"] = self.tool.getClawCurrentSpeed(self.config.slave_id)
            snapshot["torque"] = self.tool.getClawCurrentTorque(self.config.slave_id)
            snapshot["temperature"] = self.tool.getClawCurrentTemperature(self.config.slave_id)
            snapshot["voltage"] = self.tool.getClawCurrentVoltage(self.config.slave_id)
        elif self.config.model == "evs":
            snapshot["raw_status_2000"] = self.tool.getStatus(self.config.slave_id, 2000, 3, 2)
        elif self.config.model == "erg26":
            snapshot["raw_status_2000"] = self.tool.getStatus(self.config.slave_id, 2000, 3, 2)

        return snapshot


def load_config(path: Optional[str]) -> JodellConfig:
    target = Path(path) if path else Path(__file__).with_name("config.json")
    if not target.exists():
        raise FileNotFoundError(
            f"配置文件不存在: {target}。请先从 config.example.json 复制生成 config.json。"
        )
    return JodellConfig.from_file(target)
