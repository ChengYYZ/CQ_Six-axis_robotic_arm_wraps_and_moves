from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path
from typing import Callable

import requests
from PIL import Image
from requests.auth import HTTPDigestAuth


class HikvisionPreviewPublisher:
    """Continuously publish a Hikvision ISAPI snapshot as a GUI preview."""

    def __init__(
        self,
        output_path: Path,
        *,
        camera_ip: str,
        username: str,
        password: str,
        interval_s: float = 0.20,
        request_timeout_s: float = 1.5,
        status_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.camera_ip = camera_ip.strip()
        self.username = username
        self.password = password
        self.interval_s = max(0.10, float(interval_s))
        self.request_timeout_s = max(0.2, float(request_timeout_s))
        self.status_callback = status_callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_message: tuple[str, str] | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="hikvision-live-preview",
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

    @staticmethod
    def _decode_image(content: bytes) -> Image.Image:
        start = content.find(b"\xff\xd8")
        end = content.rfind(b"\xff\xd9")
        if start >= 0 and end > start:
            content = content[start : end + 2]
        with Image.open(io.BytesIO(content)) as source:
            return source.convert("RGB")

    def _publish(self, image: Image.Image) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # This producer can overlap the inspection pipeline, so it uses its own
        # temporary filename before atomically replacing the shared preview.
        temporary = self.output_path.with_name(self.output_path.name + ".live.tmp")
        image.save(temporary, format="JPEG", quality=85)
        for attempt in range(5):
            try:
                os.replace(temporary, self.output_path)
                return
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.01 * (attempt + 1))
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass

    def _run(self) -> None:
        session = requests.Session()
        # The robot LAN must not inherit Windows HTTP proxy settings.
        session.trust_env = False
        auth = HTTPDigestAuth(self.username, self.password)
        channel: str | None = None
        consecutive_failures = 0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                channels = (channel,) if channel is not None else ("101", "1")
                last_error: Exception | None = None
                image: Image.Image | None = None
                for candidate in channels:
                    try:
                        url = (
                            f"http://{self.camera_ip}/ISAPI/Streaming/channels/"
                            f"{candidate}/picture"
                        )
                        response = session.get(
                            url,
                            auth=auth,
                            timeout=self.request_timeout_s,
                        )
                        response.raise_for_status()
                        image = self._decode_image(response.content)
                        channel = candidate
                        break
                    except Exception as exc:
                        last_error = exc
                if image is None:
                    raise RuntimeError(str(last_error or "无法解码海康画面"))
                self._publish(image)
                consecutive_failures = 0
                self._notify("在线", f"{self.camera_ip} / 通道 {channel}")
            except Exception as exc:
                consecutive_failures += 1
                # Rediscover the working channel after a connection interruption.
                if consecutive_failures >= 3:
                    channel = None
                self._notify("连接失败", str(exc))

            remaining = self.interval_s - (time.monotonic() - started)
            self._stop.wait(max(0.02, remaining))

