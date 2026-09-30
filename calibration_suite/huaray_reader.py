from __future__ import annotations

import ctypes
import json
import os
import threading
import time
from ctypes import POINTER, Structure, byref
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageDraw


DEFAULT_SVSTUDIO_DIR = Path(r"D:\software\SVStudio")
BARCODE_RESULT_CHUNK_ID = 0x80000000
PIXEL_MONO8 = 0x01080001
PIXEL_RGB8 = 0x02180014
PIXEL_BGR8 = 0x02180015

_CALLBACK = ctypes.WINFUNCTYPE if os.name == "nt" else ctypes.CFUNCTYPE


class _Camera(Structure):
    pass


class _System(Structure):
    pass


class _Frame(Structure):
    pass


class _Stream(Structure):
    pass


_PCamera = POINTER(_Camera)
_PSystem = POINTER(_System)
_PFrame = POINTER(_Frame)
_PStream = POINTER(_Stream)

_RefCamera = _CALLBACK(ctypes.c_int32, _PCamera)
_CameraText = _CALLBACK(ctypes.c_char_p, _PCamera)

_Camera._fields_ = [
    ("priv", ctypes.c_void_p),
    ("addRef", _RefCamera),
    ("release", _RefCamera),
    ("getType", _RefCamera),
    ("getName", _CameraText),
    ("getKey", _CameraText),
    ("connect", _CALLBACK(ctypes.c_int32, _PCamera, ctypes.c_int)),
    ("disConnect", _RefCamera),
    ("isConnected", _RefCamera),
    ("getInterfaceName", _CameraText),
    ("getInterfaceType", _RefCamera),
    ("downLoadGenICamXML", _CALLBACK(ctypes.c_int32, _PCamera, ctypes.c_char_p)),
    ("getVendorName", _CameraText),
    ("getModelName", _CameraText),
    ("getSerialNumber", _CameraText),
    ("getDeviceVersion", _CameraText),
    ("getManufactureInfo", _CameraText),
    ("reserved", ctypes.c_uint32 * 15),
]

_System._fields_ = [
    ("priv", ctypes.c_void_p),
    ("addRef", _CALLBACK(ctypes.c_int32, _PSystem)),
    ("release", _CALLBACK(ctypes.c_int32, _PSystem)),
    (
        "discovery",
        _CALLBACK(
            ctypes.c_int32,
            _PSystem,
            POINTER(_PCamera),
            POINTER(ctypes.c_uint32),
            ctypes.c_int,
        ),
    ),
    ("getCamera", _CALLBACK(_PCamera, _PSystem, ctypes.c_char_p)),
    ("getVersion", _CALLBACK(ctypes.c_char_p, _PSystem)),
    ("reserved", ctypes.c_uint32 * 26),
]

_FrameRef = _CALLBACK(ctypes.c_int32, _PFrame)
_Frame._fields_ = [
    ("priv", ctypes.c_void_p),
    ("addRef", _FrameRef),
    ("release", _FrameRef),
    ("clone", _CALLBACK(_PFrame, _PFrame)),
    ("reset", _CALLBACK(None, _PFrame)),
    ("valid", _FrameRef),
    ("getImage", _CALLBACK(ctypes.c_void_p, _PFrame)),
    ("getFrameStatus", _CALLBACK(ctypes.c_uint32, _PFrame)),
    ("getImageWidth", _CALLBACK(ctypes.c_uint32, _PFrame)),
    ("getImageHeight", _CALLBACK(ctypes.c_uint32, _PFrame)),
    ("getImageSize", _CALLBACK(ctypes.c_uint32, _PFrame)),
    ("getImagePixelFormat", _CALLBACK(ctypes.c_int, _PFrame)),
    ("getImageTimeStamp", _CALLBACK(ctypes.c_uint64, _PFrame)),
    ("getBlockId", _CALLBACK(ctypes.c_uint64, _PFrame)),
    (
        "getPayLoadTypes",
        _CALLBACK(
            ctypes.c_int32,
            _PFrame,
            POINTER(ctypes.c_int),
            POINTER(ctypes.c_uint32),
        ),
    ),
    ("getChunkCount", _CALLBACK(ctypes.c_uint32, _PFrame)),
    (
        "getChunkDataByIndex",
        _CALLBACK(
            ctypes.c_int32,
            _PFrame,
            ctypes.c_uint32,
            POINTER(ctypes.c_uint32),
            POINTER(ctypes.c_char),
            POINTER(ctypes.c_uint32),
        ),
    ),
    ("getImagePaddingX", _CALLBACK(ctypes.c_uint32, _PFrame)),
    ("getImagePaddingY", _CALLBACK(ctypes.c_uint32, _PFrame)),
    ("reserved", ctypes.c_uint32 * 13),
]

_StreamRef = _CALLBACK(ctypes.c_int32, _PStream)
_Stream._fields_ = [
    ("priv", ctypes.c_void_p),
    ("addRef", _StreamRef),
    ("release", _StreamRef),
    (
        "startGrabbing",
        _CALLBACK(ctypes.c_int32, _PStream, ctypes.c_uint64, ctypes.c_int),
    ),
    ("stopGrabbing", _StreamRef),
    ("isGrabbing", _StreamRef),
    (
        "getFrame",
        _CALLBACK(ctypes.c_int32, _PStream, POINTER(_PFrame), ctypes.c_uint32),
    ),
    ("attachGrabbing", ctypes.c_void_p),
    ("detachGrabbing", ctypes.c_void_p),
    ("setBufferCount", _CALLBACK(ctypes.c_int32, _PStream, ctypes.c_uint32)),
    ("attachGrabbingEx", ctypes.c_void_p),
    ("detachGrabbingEx", ctypes.c_void_p),
    ("setInterPacketTimeout", _CALLBACK(ctypes.c_int32, _PStream, ctypes.c_uint32)),
    ("reserved", ctypes.c_uint32 * 19),
]


class _StreamInfo(Structure):
    _fields_ = [
        ("pCamera", _PCamera),
        ("channelId", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 30),
    ]


def _decode(value: bytes | None) -> str:
    if not value:
        return ""
    for encoding in ("utf-8", "gb18030", "latin-1"):
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", errors="replace")


def parse_barcode_chunk(lines: Iterable[str]) -> list[dict[str, object]]:
    """Parse the barcode chunk format used by the bundled ResultParser sample."""
    values: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.removesuffix(" Value").strip()] = value.strip()

    try:
        count = int(values.get("BarCodeNum", "0"))
    except ValueError:
        count = 0

    results: list[dict[str, object]] = []
    for index in range(max(0, count)):
        prefix = f"BarCode{index}_"
        code = values.get(prefix + "CodeData", "")
        try:
            point_count = int(values.get(prefix + "PosPointNum", "0"))
        except ValueError:
            point_count = 0
        points: list[tuple[int, int]] = []
        for point_index in range(max(0, point_count)):
            try:
                x = int(values[prefix + f"Point{point_index}_X"])
                y = int(values[prefix + f"Point{point_index}_Y"])
            except (KeyError, ValueError):
                continue
            points.append((x, y))
        results.append({"code": code, "points": points})
    return results


def _frame_to_image(frame: _PFrame) -> Image.Image:
    width = int(frame.contents.getImageWidth(frame))
    height = int(frame.contents.getImageHeight(frame))
    size = int(frame.contents.getImageSize(frame))
    pixel_format = int(frame.contents.getImagePixelFormat(frame)) & 0xFFFFFFFF
    address = frame.contents.getImage(frame)
    if width <= 0 or height <= 0 or size <= 0 or not address:
        raise RuntimeError("华睿 SDK 返回了无效图像")
    data = ctypes.string_at(address, size)
    if pixel_format == PIXEL_MONO8 or size == width * height:
        return Image.frombytes("L", (width, height), data[: width * height]).convert("RGB")
    if pixel_format == PIXEL_RGB8 and size >= width * height * 3:
        return Image.frombytes("RGB", (width, height), data[: width * height * 3])
    if pixel_format == PIXEL_BGR8 and size >= width * height * 3:
        return Image.frombytes(
            "RGB", (width, height), data[: width * height * 3], "raw", "BGR"
        )
    raise RuntimeError(
        f"暂不支持华睿像素格式 0x{pixel_format:08X}（{width}x{height}, {size} bytes）"
    )


def _frame_barcodes(frame: _PFrame) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    chunk_count = int(frame.contents.getChunkCount(frame))
    row_type = ctypes.c_char * 256
    buffer_type = row_type * 1000
    for index in range(chunk_count):
        chunk_id = ctypes.c_uint32()
        parameter_count = ctypes.c_uint32(1000)
        parameter_buffer = buffer_type()
        status = frame.contents.getChunkDataByIndex(
            frame,
            index,
            byref(chunk_id),
            ctypes.cast(parameter_buffer, POINTER(ctypes.c_char)),
            byref(parameter_count),
        )
        if status < 0 or chunk_id.value != BARCODE_RESULT_CHUNK_ID:
            continue
        lines = [
            _decode(bytes(parameter_buffer[row]).split(b"\0", 1)[0])
            for row in range(min(parameter_count.value, 1000))
        ]
        results.extend(parse_barcode_chunk(lines))
    return results


def _annotate(image: Image.Image, barcodes: list[dict[str, object]]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for item in barcodes:
        points = [tuple(point) for point in item.get("points", [])]  # type: ignore[arg-type]
        code = str(item.get("code", ""))
        if len(points) >= 2:
            draw.line(points + [points[0]], fill=(0, 255, 0), width=4, joint="curve")
            text_x = min(point[0] for point in points)
            text_y = max(0, min(point[1] for point in points) - 18)
        else:
            text_x, text_y = 8, 8
        if code:
            draw.text(
                (text_x, text_y),
                code,
                fill=(0, 255, 0),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
    return annotated


def _atomic_save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    image.save(temporary, format="JPEG", quality=85)
    _replace_with_retry(temporary, path)


def _atomic_save_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    _replace_with_retry(temporary, path)


def _replace_with_retry(temporary: Path, target: Path) -> bool:
    """Tolerate short-lived Windows locks from image readers and virus scanners."""
    for attempt in range(5):
        try:
            os.replace(temporary, target)
            return True
        except PermissionError:
            if attempt < 4:
                time.sleep(0.01 * (attempt + 1))
    try:
        temporary.unlink(missing_ok=True)
    except OSError:
        pass
    return False


class HuarayPreviewPublisher:
    """Publish annotated S3600 frames using the SDK bundled with SVStudio."""

    def __init__(
        self,
        output_path: Path,
        *,
        status_path: Path | None = None,
        sdk_root: Path = DEFAULT_SVSTUDIO_DIR,
        device_ip: str = "192.168.2.200",
        model_hint: str = "S3600MG000",
        interval_s: float = 0.20,
        status_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.status_path = Path(status_path) if status_path else self.output_path.with_suffix(".json")
        self.sdk_root = Path(sdk_root)
        self.device_ip = device_ip
        self.model_hint = model_hint.casefold()
        self.interval_s = max(0.05, float(interval_s))
        self.status_callback = status_callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_message: tuple[str, str] | None = None
        # Keep the last successful decode as a durable event.  The instantaneous
        # `barcodes` list is often empty again on the very next frame, which is
        # faster than a GUI polling loop can reliably observe.
        self._barcode_sequence = 0
        self._last_detection: dict[str, object] | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="huaray-preview",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _notify(self, state: str, detail: str) -> None:
        message = (state, detail)
        if message == self._last_message:
            return
        self._last_message = message
        if self.status_callback is not None:
            self.status_callback(state, detail)

    def _load_sdk(self) -> ctypes.WinDLL:
        application_dir = self.sdk_root / "Application"
        dll_path = application_dir / "MVSDKmd.dll"
        if not dll_path.is_file():
            raise FileNotFoundError(f"未找到华睿 SDK：{dll_path}")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(application_dir))
        os.environ["PATH"] = str(application_dir) + os.pathsep + os.environ.get("PATH", "")
        sdk = ctypes.WinDLL(str(dll_path))
        sdk.GENICAM_getSystemInstance.argtypes = [POINTER(_PSystem)]
        sdk.GENICAM_getSystemInstance.restype = ctypes.c_int32
        sdk.GENICAM_createStreamSource.argtypes = [POINTER(_StreamInfo), POINTER(_PStream)]
        sdk.GENICAM_createStreamSource.restype = ctypes.c_int32
        return sdk

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._stream_once()
            except Exception as exc:
                self._notify("连接失败", str(exc))
            self._stop.wait(3.0)

    def _stream_once(self) -> None:
        sdk = self._load_sdk()
        system = _PSystem()
        if sdk.GENICAM_getSystemInstance(byref(system)) < 0 or not system:
            raise RuntimeError("无法创建华睿 SDK 系统对象")

        camera = _PCamera()
        stream = _PStream()
        connected = False
        grabbing = False
        try:
            cameras = _PCamera()
            count = ctypes.c_uint32()
            if system.contents.discovery(system, byref(cameras), byref(count), 255) < 0:
                raise RuntimeError("华睿设备发现失败")
            if count.value == 0:
                raise RuntimeError(f"未发现华睿读码器 {self.device_ip}")

            candidates: list[tuple[_PCamera, str, str, str]] = []
            for index in range(count.value):
                candidate = ctypes.pointer(cameras[index])
                model = _decode(candidate.contents.getModelName(candidate))
                key = _decode(candidate.contents.getKey(candidate))
                serial = _decode(candidate.contents.getSerialNumber(candidate))
                candidates.append((candidate, model, key, serial))

            selected = next(
                (
                    item
                    for item in candidates
                    if self.model_hint in item[1].casefold()
                    and (self.device_ip in item[2] or len(candidates) == 1)
                ),
                None,
            )
            if selected is None:
                selected = next(
                    (item for item in candidates if self.device_ip in item[2]),
                    None,
                )
            if selected is None:
                found = ", ".join(f"{model or '?'} [{key}]" for _, model, key, _ in candidates)
                raise RuntimeError(f"未找到 {self.device_ip}/{self.model_hint}；已发现：{found}")
            camera, model, _key, serial = selected

            # SDK header documents accessPermissionControl (2) as the supported mode.
            if camera.contents.connect(camera, 2) < 0:
                raise RuntimeError("华睿读码器连接失败；请先断开 Smart Vision Studio 的设备连接")
            connected = True

            info = _StreamInfo()
            info.pCamera = camera
            info.channelId = 0
            if sdk.GENICAM_createStreamSource(byref(info), byref(stream)) < 0 or not stream:
                raise RuntimeError("华睿图像流创建失败")
            stream.contents.setBufferCount(stream, 3)
            if stream.contents.startGrabbing(stream, 0, 1) < 0:
                raise RuntimeError("华睿图像流启动失败")
            grabbing = True
            self._notify("在线", f"{model} / {serial}")

            last_publish = 0.0
            while not self._stop.is_set():
                frame = _PFrame()
                if stream.contents.getFrame(stream, byref(frame), 1000) < 0 or not frame:
                    continue
                try:
                    if frame.contents.valid(frame) < 0:
                        continue
                    now = time.monotonic()
                    if now - last_publish < self.interval_s:
                        continue
                    image = _frame_to_image(frame)
                    barcodes = _frame_barcodes(frame)
                    captured_at = time.time()
                    if any(str(item.get("code") or "").strip() for item in barcodes):
                        self._barcode_sequence += 1
                        self._last_detection = {
                            "sequence": self._barcode_sequence,
                            "captured_at": captured_at,
                            "barcodes": barcodes,
                        }
                    _atomic_save_image(_annotate(image, barcodes), self.output_path)
                    _atomic_save_json(
                        {
                            "device_ip": self.device_ip,
                            "model": model,
                            "serial": serial,
                            "captured_at": captured_at,
                            "barcodes": barcodes,
                            "last_detection": self._last_detection,
                        },
                        self.status_path,
                    )
                    last_publish = now
                finally:
                    frame.contents.release(frame)
        finally:
            if stream:
                if grabbing:
                    stream.contents.stopGrabbing(stream)
                stream.contents.release(stream)
            if camera and connected:
                camera.contents.disConnect(camera)
            if system:
                system.contents.release(system)
