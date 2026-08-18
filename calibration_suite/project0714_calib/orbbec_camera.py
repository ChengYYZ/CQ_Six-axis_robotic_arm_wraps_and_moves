from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _load_pyorbbecsdk():
    try:
        from pyorbbecsdk import Config, OBAlignMode, OBFormat, OBSensorType, Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyorbbecsdk is not available. Run calibration_suite\\bootstrap_venv.ps1 first."
        ) from exc
    return Pipeline, Config, OBSensorType, OBFormat, OBAlignMode


@dataclass
class CameraOpenOptions:
    width: int = 1280
    height: int = 800
    fps: int = 10
    depth_width: int = 640
    depth_height: int = 400
    depth_fps: int = 5
    align_mode: str = "sw"
    wait_timeout_ms: int = 500


class OrbbecColorCamera:
    def __init__(self, options: CameraOpenOptions | None = None) -> None:
        self.options = options or CameraOpenOptions()
        self._pipeline = None
        self._config = None

    def start(self) -> None:
        Pipeline, Config, OBSensorType, OBFormat, _ = _load_pyorbbecsdk()

        self._pipeline = Pipeline()
        self._config = Config()
        profile_list = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

        selected = None
        for fmt_name in ("RGB", "BGR", "MJPG", "YUYV", "YUY2"):
            try:
                selected = profile_list.get_video_stream_profile(
                    self.options.width,
                    self.options.height,
                    getattr(OBFormat, fmt_name),
                    self.options.fps,
                )
                if selected is not None:
                    break
            except Exception:
                continue

        if selected is None:
            selected = profile_list.get_default_video_stream_profile()

        self._config.enable_stream(selected)
        self._pipeline.start(self._config)

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None

    def get_bgr_frame(self, timeout_ms: int = 100) -> np.ndarray | None:
        if self._pipeline is None:
            raise RuntimeError("Camera has not been started.")

        frames = self._pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None

        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None

        return _frame_to_bgr(color_frame)


class OrbbecRGBDCamera:
    def __init__(self, options: CameraOpenOptions | None = None) -> None:
        self.options = options or CameraOpenOptions()
        self._pipeline = None
        self._config = None

    def start(self) -> None:
        Pipeline, Config, OBSensorType, OBFormat, OBAlignMode = _load_pyorbbecsdk()

        self._pipeline = Pipeline()
        self._config = Config()

        color_profiles = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        depth_profiles = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

        color_profile = None
        for fmt_name in ("MJPG", "RGB888", "BGRA", "YUYV", "YUY2"):
            try:
                color_profile = color_profiles.get_video_stream_profile(
                    self.options.width,
                    self.options.height,
                    getattr(OBFormat, fmt_name),
                    self.options.fps,
                )
                if color_profile is not None:
                    break
            except Exception:
                continue
        if color_profile is None:
            color_profile = color_profiles.get_default_video_stream_profile()

        depth_profile = None
        for width, height, fps in (
            (self.options.depth_width, self.options.depth_height, self.options.depth_fps),
            (self.options.depth_width, self.options.depth_height, 10),
            (640, 400, 5),
            (640, 400, 10),
            (1280, 800, 5),
        ):
            try:
                depth_profile = depth_profiles.get_video_stream_profile(
                    width,
                    height,
                    OBFormat.Y16,
                    fps,
                )
                if depth_profile is not None:
                    break
            except Exception:
                continue
        if depth_profile is None:
            depth_profile = depth_profiles.get_default_video_stream_profile()

        self._config.enable_stream(color_profile)
        self._config.enable_stream(depth_profile)

        align_mode = self.options.align_mode.lower()
        if align_mode == "hw":
            self._config.set_align_mode(OBAlignMode.HW_MODE)
        elif align_mode == "off":
            self._config.set_align_mode(OBAlignMode.DISABLE)
        else:
            self._config.set_align_mode(OBAlignMode.SW_MODE)

        try:
            self._pipeline.enable_frame_sync()
        except Exception:
            pass

        self._pipeline.start(self._config)

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None

    def get_frames(
        self,
        timeout_ms: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if self._pipeline is None:
            raise RuntimeError("Camera has not been started.")

        frames = self._pipeline.wait_for_frames(timeout_ms or self.options.wait_timeout_ms)
        if frames is None:
            return None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None

        color_bgr = _frame_to_bgr(color_frame)
        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            (depth_frame.get_height(), depth_frame.get_width())
        )
        depth_mm = depth_raw.astype(np.float32) * float(depth_frame.get_depth_scale())
        depth_display = _depth_to_display(depth_mm)
        return color_bgr, depth_mm, depth_display


def _frame_to_bgr(color_frame) -> np.ndarray:
    import cv2

    width = color_frame.get_width()
    height = color_frame.get_height()
    fmt = color_frame.get_format()
    fmt_name = getattr(fmt, "name", str(fmt)).upper()
    data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

    if "MJPG" in fmt_name or "JPEG" in fmt_name:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode MJPG color frame.")
        return image

    if "BGR" in fmt_name:
        return data.reshape((height, width, 3)).copy()

    if "RGB" in fmt_name:
        rgb = data.reshape((height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if "YUYV" in fmt_name or "YUY2" in fmt_name:
        yuyv = data.reshape((height, width, 2))
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)

    raise RuntimeError(f"Unsupported Orbbec color frame format: {fmt_name}")


def _depth_to_display(depth_mm: np.ndarray) -> np.ndarray:
    import cv2

    valid = np.where(depth_mm > 0.0, depth_mm, 0.0)
    return cv2.normalize(valid, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
