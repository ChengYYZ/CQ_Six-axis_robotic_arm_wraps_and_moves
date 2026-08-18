from __future__ import annotations

import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .common import make_transform, rpy_xyz_to_matrix


MOVE_EVENT_ID_KEY = "cmdID"
MOVE_EVENT_REACH_TARGET_KEY = "reachTarget"
MOVE_EVENT_ERROR_KEY = "error"
MOVE_EVENT_REMARK_KEY = "remark"


def _sdk_root_candidates() -> list[Path]:
    workspace_root = Path(__file__).resolve().parents[2]
    return [
        workspace_root / "xCoreSDK-Python-0.7.1-win",
        workspace_root / "xCoreSDK-Python-main" / "xCoreSDK-Python-main",
    ]


def load_xcore_sdk():
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    for root in _sdk_root_candidates():
        release_dir = root / "Release"
        windows_dir = release_dir / "windows"
        stub_dir = windows_dir / "xCoreSDK_python"
        runtime_dll = windows_dir / "xCoreSDK.dll"
        runtime_pyd = windows_dir / f"xCoreSDK_python.{py_tag}-win_amd64.pyd"

        if runtime_dll.exists() and runtime_pyd.exists():
            for path in (windows_dir, stub_dir, release_dir, root):
                sys.path.insert(0, str(path))
            return importlib.import_module("xCoreSDK_python")

    searched = "\n".join(f"  - {path}" for path in _sdk_root_candidates())
    raise RuntimeError(
        "Matching xCoreSDK runtime files were not found for the current Python version.\n"
        f"Python tag: {py_tag}\n"
        f"Searched:\n{searched}"
    )


def _ensure_ok(xcore_sdk_python, ec: dict, action: str) -> None:
    code = ec.get("code", 0)
    if code in (0, "0", None):
        return
    try:
        detail = xcore_sdk_python.message(ec)
    except Exception:
        detail = str(ec)
    raise RuntimeError(f"{action} failed: {detail}")


def _sdk_call(xcore_sdk_python, action: str, ec: dict, func, *args):
    ec.clear()
    result = func(*args, ec)
    _ensure_ok(xcore_sdk_python, ec, action)
    return result


def _message_variants(message: str) -> list[str]:
    variants = {message.lower()}
    for encoding in ("gbk", "cp936"):
        try:
            variants.add(message.encode(encoding).decode("utf-8").lower())
        except UnicodeError:
            pass
    return list(variants)


def _is_unsupported_model_message(message: str) -> bool:
    for text in _message_variants(message):
        if "xmc18-r925" in text:
            return True
        if ("not support" in text or "not supported" in text or "unsupported" in text) and (
            "model" in text or "xmate" in text
        ):
            return True
        if ("不支持" in text or "暂不支持" in text) and ("模型" in text or "机型" in text or "xmate" in text):
            return True
    return False


@dataclass
class RobotPose:
    translation_m: np.ndarray
    rpy_rad_xyz: np.ndarray
    conf_data: list[float]
    external: list[float]
    has_elbow: bool
    elbow: float

    def to_transform(self) -> np.ndarray:
        rotation = rpy_xyz_to_matrix(self.rpy_rad_xyz)
        return make_transform(rotation, self.translation_m)

    def translation_mm(self) -> np.ndarray:
        return self.translation_m * 1000.0

    def rpy_deg_xyz(self) -> np.ndarray:
        return np.degrees(self.rpy_rad_xyz)

    def to_dict(self) -> dict:
        return {
            "translation_m": self.translation_m.tolist(),
            "translation_mm": self.translation_mm().tolist(),
            "rpy_rad_xyz": self.rpy_rad_xyz.tolist(),
            "rpy_deg_xyz": self.rpy_deg_xyz().tolist(),
            "conf_data": list(self.conf_data),
            "external": list(self.external),
            "has_elbow": bool(self.has_elbow),
            "elbow": float(self.elbow),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RobotPose":
        return cls(
            translation_m=np.asarray(payload["translation_m"], dtype=np.float64),
            rpy_rad_xyz=np.asarray(payload["rpy_rad_xyz"], dtype=np.float64),
            conf_data=list(payload.get("conf_data", [])),
            external=list(payload.get("external", [])),
            has_elbow=bool(payload.get("has_elbow", False)),
            elbow=float(payload.get("elbow", 0.0)),
        )


@dataclass
class MotionOptions:
    motion: str = "movej"
    speed_mm_s: float = 300.0
    zone_mm: float = 0.0
    timeout_s: float = 60.0
    use_current_conf_data: bool = False
    stop_requested: Callable[[], bool] | None = None


@dataclass
class ToolsetInfo:
    end_translation_m: np.ndarray
    end_rpy_rad_xyz: np.ndarray
    ref_translation_m: np.ndarray
    ref_rpy_rad_xyz: np.ndarray

    def end_translation_mm(self) -> np.ndarray:
        return self.end_translation_m * 1000.0

    def end_rpy_deg_xyz(self) -> np.ndarray:
        return np.degrees(self.end_rpy_rad_xyz)


class XCoreRobotClient:
    def __init__(self, robot_ip: str) -> None:
        self.robot_ip = robot_ip
        self.xcore = load_xcore_sdk()
        self.robot = self.xcore.xMateRobot(robot_ip)
        self.ec: dict = {}
        self.connected = False
        self.motion_prepared = False
        self._ik_precheck_warning_shown = False
        self._toolset = None

    def connect(self) -> None:
        if self.connected:
            return
        _sdk_call(self.xcore, "connectToRobot", self.ec, self.robot.connectToRobot)
        self.connected = True

    def disconnect(self) -> None:
        if not self.connected:
            return
        try:
            self.ec.clear()
            self.robot.disconnectFromRobot(self.ec)
        finally:
            self.connected = False
            self.motion_prepared = False

    def prepare_motion(self, speed_mm_s: float = 300.0, zone_mm: float = 0.0) -> None:
        self.connect()
        if not self.motion_prepared:
            _sdk_call(
                self.xcore,
                "setOperateMode",
                self.ec,
                self.robot.setOperateMode,
                self.xcore.OperateMode.automatic,
            )
            _sdk_call(
                self.xcore,
                "setMotionControlMode",
                self.ec,
                self.robot.setMotionControlMode,
                self.xcore.MotionControlMode.NrtCommandMode,
            )
            _sdk_call(self.xcore, "setPowerState", self.ec, self.robot.setPowerState, True)
            self.motion_prepared = True

        _sdk_call(self.xcore, "setDefaultSpeed", self.ec, self.robot.setDefaultSpeed, speed_mm_s)
        _sdk_call(self.xcore, "setDefaultZone", self.ec, self.robot.setDefaultZone, zone_mm)

    def read_current_pose(self) -> RobotPose:
        self.connect()
        posture = _sdk_call(
            self.xcore,
            "cartPosture",
            self.ec,
            self.robot.cartPosture,
            self.xcore.CoordinateType.endInRef,
        )
        return RobotPose(
            translation_m=np.asarray(posture.trans, dtype=np.float64),
            rpy_rad_xyz=np.asarray(posture.rpy, dtype=np.float64),
            conf_data=list(posture.confData),
            external=list(posture.external),
            has_elbow=bool(posture.hasElbow),
            elbow=float(posture.elbow),
        )

    def set_toolset_by_name(self, tool_name: str, wobj_name: str = "wobj0") -> ToolsetInfo:
        self.connect()
        self._toolset = _sdk_call(
            self.xcore,
            "setToolset",
            self.ec,
            self.robot.setToolset,
            tool_name,
            wobj_name,
        )
        return self._toolset_info_from_sdk(self._toolset)

    def set_toolset_with_tcp_override(
        self,
        tool_name: str,
        wobj_name: str,
        xyz_mm: list[float] | tuple[float, float, float],
        rpy_deg: list[float] | tuple[float, float, float],
    ) -> ToolsetInfo:
        """Use a session-only TCP override while preserving the named tool load and wobj."""
        self.connect()
        source = _sdk_call(
            self.xcore,
            "setToolset",
            self.ec,
            self.robot.setToolset,
            tool_name,
            wobj_name,
        )
        xyz = np.asarray(xyz_mm, dtype=np.float64)
        rpy = np.radians(np.asarray(rpy_deg, dtype=np.float64))
        if xyz.shape != (3,) or rpy.shape != (3,) or not np.all(np.isfinite(np.r_[xyz, rpy])):
            raise ValueError("TCP override requires three finite XYZ and three finite RPY values.")
        end = self.xcore.Frame((xyz / 1000.0).tolist(), rpy.tolist())
        self._toolset = self.xcore.Toolset(source.load, end, source.ref)
        _sdk_call(self.xcore, "setToolset override", self.ec, self.robot.setToolset, self._toolset)
        return self._toolset_info_from_sdk(self._toolset)

    def read_toolset(self) -> ToolsetInfo:
        self.connect()
        self._toolset = _sdk_call(self.xcore, "toolset", self.ec, self.robot.toolset)
        return self._toolset_info_from_sdk(self._toolset)

    def set_do(self, board: int, port: int, state: bool) -> None:
        self.connect()
        _sdk_call(self.xcore, "setDO", self.ec, self.robot.setDO, int(board), int(port), bool(state))

    def get_do(self, board: int, port: int) -> bool:
        self.connect()
        return bool(_sdk_call(self.xcore, "getDO", self.ec, self.robot.getDO, int(board), int(port)))

    def move_to_pose_mm_deg(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        rx_deg: float,
        ry_deg: float,
        rz_deg: float,
        options: MotionOptions | None = None,
    ) -> RobotPose:
        opts = options or MotionOptions()
        self._validate_motion_request(
            opts, (x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg)
        )
        self.prepare_motion(opts.speed_mm_s, opts.zone_mm)
        current = self.read_current_pose()
        target = self._build_target_pose(
            current,
            x_mm,
            y_mm,
            z_mm,
            rx_deg,
            ry_deg,
            rz_deg,
            opts.use_current_conf_data,
        )

        self._precheck_target_pose(target)
        _sdk_call(self.xcore, "moveReset", self.ec, self.robot.moveReset)

        if opts.motion.lower() == "movel":
            command = self.xcore.MoveLCommand(target, opts.speed_mm_s, opts.zone_mm)
        else:
            command = self.xcore.MoveJCommand(target, opts.speed_mm_s, opts.zone_mm)

        cmd_id = self.xcore.PyString()
        _sdk_call(self.xcore, "moveAppend", self.ec, self.robot.moveAppend, [command], cmd_id)
        _sdk_call(self.xcore, "moveStart", self.ec, self.robot.moveStart)

        started = self._wait_until_started(min(opts.timeout_s, 3.0), opts.stop_requested)
        self._wait_until_idle(opts.timeout_s, opts.stop_requested)
        final_pose = self.read_current_pose()
        position_error_mm = self._pose_distance_mm(final_pose, target)
        orientation_error_deg = self._pose_orientation_distance_deg(final_pose, target)
        event_info = self._query_move_event()

        if (
            (not started and position_error_mm > 2.0)
            or position_error_mm > 10.0
            or orientation_error_deg > 5.0
        ):
            raise RuntimeError(
                "Robot command did not reach the target pose.\n"
                f"Command id: {self._pystring_content(cmd_id)}\n"
                f"Started moving: {started}\n"
                f"Final position error: {position_error_mm:.2f} mm\n"
                f"Final orientation error: {orientation_error_deg:.2f} deg\n"
                f"Move event: {self._format_move_event(event_info)}"
            )
        return final_pose

    def move_linear_path_mm_deg(
        self,
        poses_mm_deg: list[tuple[float, float, float, float, float, float, float]],
        options: MotionOptions | None = None,
    ) -> RobotPose:
        if not poses_mm_deg:
            raise ValueError("No path poses were provided.")

        opts = options or MotionOptions(motion="movel")
        for pose in poses_mm_deg:
            self._validate_motion_request(opts, pose[:6])
        self.prepare_motion(opts.speed_mm_s, opts.zone_mm)
        current = self.read_current_pose()

        targets = [
            self._build_target_pose(
                current,
                x_mm,
                y_mm,
                z_mm,
                rx_deg,
                ry_deg,
                rz_deg,
                opts.use_current_conf_data,
            )
            for x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg, _zone_mm in poses_mm_deg
        ]
        for target in targets:
            self._precheck_target_pose(target)

        commands = [
            self.xcore.MoveLCommand(target, opts.speed_mm_s, zone_mm)
            for target, (*_pose, zone_mm) in zip(targets, poses_mm_deg)
        ]
        cmd_id = self.xcore.PyString()
        _sdk_call(self.xcore, "moveReset", self.ec, self.robot.moveReset)
        _sdk_call(self.xcore, "moveAppend", self.ec, self.robot.moveAppend, commands, cmd_id)
        _sdk_call(self.xcore, "moveStart", self.ec, self.robot.moveStart)

        started = self._wait_until_started(min(opts.timeout_s, 3.0), opts.stop_requested)
        self._wait_until_idle(opts.timeout_s, opts.stop_requested)
        final_pose = self.read_current_pose()
        final_target = targets[-1]
        position_error_mm = self._pose_distance_mm(final_pose, final_target)
        orientation_error_deg = self._pose_orientation_distance_deg(final_pose, final_target)
        event_info = self._query_move_event()

        if (
            (not started and position_error_mm > 2.0)
            or position_error_mm > 10.0
            or orientation_error_deg > 5.0
        ):
            raise RuntimeError(
                "Robot path command did not reach the final target pose.\n"
                f"Command id: {self._pystring_content(cmd_id)}\n"
                f"Started moving: {started}\n"
                f"Final position error: {position_error_mm:.2f} mm\n"
                f"Final orientation error: {orientation_error_deg:.2f} deg\n"
                f"Move event: {self._format_move_event(event_info)}"
            )
        return final_pose

    def stop_motion(self) -> None:
        """Software-stop current non-realtime motion and clear queued commands."""
        self.connect()
        errors: list[str] = []
        for action, func in (
            ("stop", self.robot.stop),
            ("moveReset", self.robot.moveReset),
        ):
            try:
                _sdk_call(self.xcore, action, self.ec, func)
            except Exception as exc:
                errors.append(f"{action}: {exc}")
        if errors:
            print("Warning: robot software stop reported errors: " + "; ".join(errors))

    def _build_target_pose(
        self,
        current: RobotPose,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        rx_deg: float,
        ry_deg: float,
        rz_deg: float,
        use_current_conf_data: bool,
    ):
        import math

        target = self.xcore.CartesianPosition(
            [
                x_mm / 1000.0,
                y_mm / 1000.0,
                z_mm / 1000.0,
                math.radians(rx_deg),
                math.radians(ry_deg),
                math.radians(rz_deg),
            ]
        )
        if use_current_conf_data:
            target.confData = list(current.conf_data)
            target.hasElbow = current.has_elbow
            target.elbow = current.elbow
        else:
            target.confData = []
            target.hasElbow = False
            target.elbow = 0.0
        target.external = list(current.external)
        return target

    def _precheck_target_pose(self, target) -> None:
        try:
            toolset = self._toolset
            if toolset is None:
                toolset = _sdk_call(self.xcore, "toolset", self.ec, self.robot.toolset)
                self._toolset = toolset
            _sdk_call(self.xcore, "calcIk", self.ec, self.robot.model().calcIk, target, toolset)
        except Exception as exc:
            if _is_unsupported_model_message(str(exc)):
                if not self._ik_precheck_warning_shown:
                    print(
                        "Warning: xCore SDK local IK precheck does not support this robot model; "
                        "skipping local precheck and letting the controller validate the move."
                    )
                    self._ik_precheck_warning_shown = True
                return
            raise

    def _query_move_event(self) -> dict:
        try:
            return _sdk_call(
                self.xcore,
                "queryEventInfo",
                self.ec,
                self.robot.queryEventInfo,
                self.xcore.Event.moveExecution,
            )
        except Exception:
            return {}

    def _move_event_error(self) -> object | None:
        event_info = self._query_move_event()
        error = event_info.get(MOVE_EVENT_ERROR_KEY) if event_info else None
        if not error:
            return None
        if isinstance(error, dict):
            code = error.get("ec", 0)
            if code in (0, "0", None):
                return None
        return error

    def _format_move_event(self, event_info: dict) -> str:
        if not event_info:
            return "{}"
        fields = {
            "cmdID": event_info.get(MOVE_EVENT_ID_KEY),
            "reachTarget": event_info.get(MOVE_EVENT_REACH_TARGET_KEY),
            "error": event_info.get(MOVE_EVENT_ERROR_KEY),
            "remark": event_info.get(MOVE_EVENT_REMARK_KEY),
        }
        return str(fields)

    def _pystring_content(self, value) -> str:
        content = getattr(value, "content", "")
        if callable(content):
            try:
                return str(content())
            except Exception:
                return ""
        return str(content)

    def _pose_distance_mm(self, pose: RobotPose, target) -> float:
        target_mm = np.asarray(target.trans, dtype=np.float64) * 1000.0
        return float(np.linalg.norm(pose.translation_mm() - target_mm))

    def _pose_orientation_distance_deg(self, pose: RobotPose, target) -> float:
        target_rotation = rpy_xyz_to_matrix(np.asarray(target.rpy, dtype=np.float64))
        actual_rotation = rpy_xyz_to_matrix(pose.rpy_rad_xyz)
        relative = actual_rotation.T @ target_rotation
        cos_angle = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_angle)))

    @staticmethod
    def _validate_motion_request(opts: MotionOptions, pose_values) -> None:
        if opts.motion.lower() not in ("movej", "movel"):
            raise ValueError(f"Unsupported robot motion type: {opts.motion}")
        if not np.isfinite(opts.speed_mm_s) or opts.speed_mm_s <= 0.0:
            raise ValueError("Robot speed must be a positive finite value.")
        if not np.isfinite(opts.zone_mm) or opts.zone_mm < 0.0:
            raise ValueError("Robot zone must be a non-negative finite value.")
        if not np.isfinite(opts.timeout_s) or opts.timeout_s <= 0.0:
            raise ValueError("Robot timeout must be a positive finite value.")
        values = np.asarray(tuple(pose_values), dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("Robot target pose must contain six finite values.")

    def _check_operator_stop(self, stop_requested: Callable[[], bool] | None) -> None:
        if stop_requested is not None and stop_requested():
            self.stop_motion()
            raise RuntimeError("Robot motion interrupted by operator software stop request.")

    def _wait_until_started(
        self,
        timeout_s: float,
        stop_requested: Callable[[], bool] | None = None,
    ) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self._check_operator_stop(stop_requested)
            event_error = self._move_event_error()
            if event_error is not None:
                raise RuntimeError(f"Robot move rejected before start: {event_error}")
            state = _sdk_call(self.xcore, "operationState", self.ec, self.robot.operationState)
            if state in (
                self.xcore.OperationState.moving,
                self.xcore.OperationState.jogging,
            ):
                return True
            time.sleep(0.05)
        return False

    def _wait_until_idle(
        self,
        timeout_s: float,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self._check_operator_stop(stop_requested)
            event_error = self._move_event_error()
            if event_error is not None:
                raise RuntimeError(f"Robot move failed while waiting: {event_error}")
            state = _sdk_call(self.xcore, "operationState", self.ec, self.robot.operationState)
            if state in (
                self.xcore.OperationState.idle,
                self.xcore.OperationState.unknown,
            ):
                return
            time.sleep(0.05)
        raise TimeoutError(f"Robot motion did not finish within {timeout_s:.1f} seconds.")

    def _toolset_info_from_sdk(self, toolset) -> ToolsetInfo:
        return ToolsetInfo(
            end_translation_m=np.asarray(toolset.end.trans, dtype=np.float64),
            end_rpy_rad_xyz=np.asarray(toolset.end.rpy, dtype=np.float64),
            ref_translation_m=np.asarray(toolset.ref.trans, dtype=np.float64),
            ref_rpy_rad_xyz=np.asarray(toolset.ref.rpy, dtype=np.float64),
        )

    def __enter__(self) -> "XCoreRobotClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()
