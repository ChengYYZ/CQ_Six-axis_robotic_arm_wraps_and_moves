# 条形码识别部分
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable
import json
import time
from collections.abc import Iterator
import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth


@dataclass(frozen=True)
class WaybillInspectionResult:
    candidate_index: int
    has_waybill: bool
    barcode: str | None
    frame_count: int
    waybill_frame_count: int
    elapsed_s: float
    error: str | None = None
    detection_to_barcode_s: float | None = None


class AsyncWaybillInspector:
    """Capture near waypoint C and decode waybills without stopping robot motion."""

    def __init__(
        self,
        camera_ip: str,
        username: str,
        password: str,
        model_path: str | Path,
        output_dir: str | Path,
        confidence: float = 0.5,
        capture_count: int = 100,
        capture_interval_s: float = 0.1,
        capture_duration_s: float = 30.0,
        c_settle_s: float = 0.6,
        post_c_capture_s: float = 2.5,
        request_timeout_s: float = 1.0,
        waybill_class_id: int = 0,
        result_callback: Callable[[WaybillInspectionResult], None] | None = None,
    ) -> None:
        if not password:
            raise ValueError("Hikvision password is empty.")

        try:
            import zxingcpp
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Waybill inspection dependencies are missing. Install zxing-cpp and ultralytics."
            ) from exc

        self.camera_ip = camera_ip
        self.username = username
        self.password = password
        self.confidence = float(confidence)
        self.capture_count = max(1, int(capture_count))
        self.capture_interval_s = max(0.02, float(capture_interval_s))
        self.capture_duration_s = max(
            self.capture_interval_s, float(capture_duration_s)
        )
        self.c_settle_s = max(0.0, float(c_settle_s))
        self.post_c_capture_s = max(0.0, float(post_c_capture_s))
        self.request_timeout_s = max(0.1, float(request_timeout_s))
        self.waybill_class_id = int(waybill_class_id)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._zxingcpp = zxingcpp
        self._model = YOLO(str(model_path))
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="waybill-decode"
        )
        self._result_callback = result_callback or self._print_result

        self._lock = Lock()
        self._frames: deque[tuple[float, np.ndarray]] = deque(maxlen=self.capture_count)
        self._capture_stop = Event()
        self._capture_thread: Thread | None = None
        self._capture_generation = 0
        self._active_candidate: int | None = None
        self._c_arrival_time: float | None = None
        self._futures: dict[int, Future[WaybillInspectionResult]] = {}

    def begin_capture(self, candidate_index: int) -> None:
        """Capture a fixed-duration clip and submit it automatically for inspection."""
        self.cancel_capture()
        with self._lock:
            self._capture_generation += 1
            generation = self._capture_generation
            self._active_candidate = int(candidate_index)
            self._c_arrival_time = None
            self._frames.clear()
            self._capture_stop = Event()
        self._capture_thread = Thread(
            target=self._capture_loop,
            args=(generation, self._capture_stop),
            name=f"waybill-capture-{candidate_index}",
            daemon=True,
        )
        self._capture_thread.start()

    def mark_c_arrival(self, candidate_index: int) -> None:
        """Start the settle/static-capture phase and discard all travel frames."""
        with self._lock:
            if self._active_candidate != int(candidate_index):
                return
            self._c_arrival_time = time.monotonic()
            self._frames.clear()

    def submit_capture(self, candidate_index: int) -> Future[WaybillInspectionResult]:
        """Freeze buffered C-point frames and process them on a worker thread."""
        with self._lock:
            self._capture_stop.set()
            self._active_candidate = None
            self._c_arrival_time = None
            frames = [image.copy() for _timestamp, image in self._frames]
            self._frames.clear()

        return self._submit_frames(int(candidate_index), frames)

    def cancel_capture(self) -> None:
        with self._lock:
            self._capture_stop.set()
            self._active_candidate = None
            self._c_arrival_time = None
            self._frames.clear()

    def close(self) -> None:
        self.cancel_capture()
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _capture_loop(self, generation: int, stop_event: Event) -> None:
        session = requests.Session()
        # The Hikvision camera is on the robot LAN.  Never send its ISAPI
        # requests through HTTP_PROXY/HTTPS_PROXY inherited from Windows.
        session.trust_env = False
        auth = HTTPDigestAuth(self.username, self.password)
        next_capture_time = 0.0
        started = time.monotonic()
        capture_failures = 0
        camera_prewarmed = False
        while (
            not stop_event.is_set()
            and time.monotonic() - started < self.capture_duration_s
        ):
            with self._lock:
                c_arrival_time = (
                    self._c_arrival_time
                    if generation == self._capture_generation
                    else None
                )
            now = time.monotonic()
            if (
                c_arrival_time is not None
                and now - c_arrival_time >= self.c_settle_s + self.post_c_capture_s
            ):
                break

            # Start at B only to initialize the HTTP session/camera path. Do
            # not retain B->C motion frames. One successful snapshot is enough
            # to prove that capture is ready before the robot reaches C.
            if c_arrival_time is None and camera_prewarmed:
                stop_event.wait(0.05)
                continue
            if now < next_capture_time:
                stop_event.wait(next_capture_time - now)
                continue
            next_capture_time = now + self.capture_interval_s
            try:
                image = self._capture_snapshot(session, auth)
            except Exception as exc:
                capture_failures += 1
                if capture_failures == 1 or capture_failures % 5 == 0:
                    print(
                        f"C-camera snapshot failed for candidate #{self._active_candidate} "
                        f"({capture_failures} failure(s)): {exc}"
                    )
                continue
            camera_prewarmed = True
            with self._lock:
                if (
                    generation != self._capture_generation
                    or self._active_candidate is None
                ):
                    return
                # Frames acquired before C or during the post-arrival settling
                # window keep the camera warm but are deliberately discarded.
                captured_at = time.monotonic()
                elapsed_after_c_s = (
                    captured_at - c_arrival_time if c_arrival_time is not None else -1.0
                )
                if (
                    self.c_settle_s
                    <= elapsed_after_c_s
                    <= self.c_settle_s + self.post_c_capture_s
                ):
                    self._frames.append((captured_at, image))

        if not stop_event.is_set():
            with self._lock:
                if (
                    generation != self._capture_generation
                    or self._active_candidate is None
                ):
                    return
                candidate_index = self._active_candidate
                self._active_candidate = None
                self._c_arrival_time = None
                frames = [image.copy() for _timestamp, image in self._frames]
                self._frames.clear()
            self._submit_frames(candidate_index, frames)

    def _submit_frames(
        self, candidate_index: int, frames: list[np.ndarray]
    ) -> Future[WaybillInspectionResult]:
        future = self._executor.submit(
            self._inspect_frames, int(candidate_index), frames
        )
        self._futures[int(candidate_index)] = future
        future.add_done_callback(self._handle_future)
        return future

    def _capture_snapshot(
        self, session: requests.Session, auth: HTTPDigestAuth
    ) -> np.ndarray:
        last_error: Exception | None = None
        for channel in ("101", "1"):
            url = f"http://{self.camera_ip}/ISAPI/Streaming/channels/{channel}/picture"
            try:
                response = session.get(url, auth=auth, timeout=self.request_timeout_s)
                response.raise_for_status()
                return self._decode_jpeg(response.content)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Hikvision snapshot failed: {last_error}")

    @staticmethod
    def _decode_jpeg(data: bytes) -> np.ndarray:
        start = data.find(b"\xff\xd8")
        end = data.rfind(b"\xff\xd9")
        if start >= 0 and end > start:
            data = data[start : end + 2]
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Hikvision JPEG decode failed.")
        return image

    def _inspect_frames(
        self, candidate_index: int, frames: list[np.ndarray]
    ) -> WaybillInspectionResult:
        started = time.monotonic()
        if not frames:
            return WaybillInspectionResult(
                candidate_index=candidate_index,
                has_waybill=False,
                barcode=None,
                frame_count=0,
                waybill_frame_count=0,
                elapsed_s=time.monotonic() - started,
                error="No buffered frame was available at waypoint C.",
            )

        run_dir = (
            self.output_dir
            / f"candidate_{candidate_index}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        self._save_video(run_dir / "c_camera_capture.mp4", frames)
        full_frame_sharpness: list[float] = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            full_frame_sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        print(
            f"Waybill C-frame quality candidate #{candidate_index}: "
            f"captured={len(frames)} "
            f"full_frame_best={max(full_frame_sharpness):.2f} "
            f"full_frame_min={min(full_frame_sharpness):.2f}; "
            "full-frame sharpness is diagnostic only"
        )
        frame_votes: Counter[str] = Counter()
        fusion_votes: Counter[str] = Counter()
        waybill_frame_count = 0
        first_waybill_detected_at: float | None = None

        try:
            detected_rois: list[tuple[float, int, int, np.ndarray]] = []
            for frame_index, frame in enumerate(frames):
                cv2.imwrite(str(run_dir / f"frame_{frame_index}.jpg"), frame)
                boxes = self._detect_waybills(frame)
                if not boxes:
                    continue
                if first_waybill_detected_at is None:
                    first_waybill_detected_at = time.monotonic()
                waybill_frame_count += 1
                for box_index, box in enumerate(boxes):
                    roi = self._crop(frame, box)
                    if roi.size == 0:
                        continue
                    cv2.imwrite(
                        str(run_dir / f"waybill_{frame_index}_{box_index}.jpg"), roi
                    )
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi_sharpness = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())
                    detected_rois.append((roi_sharpness, frame_index, box_index, roi))

            # 每个物理帧只保留最清晰的一个 ROI，
            # 避免同一帧因为多个检测框重复投票。
            best_roi_by_frame_for_decode: dict[int, tuple[float, int, np.ndarray]] = {}

            for sharpness, frame_index, box_index, roi in detected_rois:
                previous = best_roi_by_frame_for_decode.get(frame_index)

                if previous is None or sharpness > previous[0]:
                    best_roi_by_frame_for_decode[frame_index] = (
                        sharpness,
                        box_index,
                        roi,
                    )

            selected_rois = sorted(
                [
                    (
                        sharpness,
                        frame_index,
                        box_index,
                        roi,
                    )
                    for frame_index, (
                        sharpness,
                        box_index,
                        roi,
                    ) in best_roi_by_frame_for_decode.items()
                ],
                key=lambda item: item[0],
                reverse=True,
            )[:6]
            if selected_rois:
                print(
                    f"Waybill ROI quality candidate #{candidate_index}: "
                    f"detected_frames={waybill_frame_count}/{len(frames)} "
                    f"regions={len(detected_rois)} selected={len(selected_rois)} "
                    f"best_sharpness={selected_rois[0][0]:.2f} "
                    f"selected_min={selected_rois[-1][0]:.2f}"
                )
            # Each physical frame contributes at most one vote per decoded
            # value. This prevents the many preprocessing variants of one
            # blurry frame from overwhelming agreement across real frames.
            for _sharpness, frame_index, box_index, roi in selected_rois:
                codes = self._read_barcodes(roi)

                # 同一个物理帧对同一个值最多贡献一票
                frame_votes.update(set(codes))

                if codes:
                    print(
                        f"Barcode candidates frame={frame_index} "
                        f"roi={box_index}: {codes}"
                    )
            # if codes:
            #     print(
            #         f"Barcode candidates frame={frame_index} roi={box_index}: {codes}"
            #     )

            # At C the package is stationary. Align the best ROI from each
            # frame before fusing it; travel/settling frames have already been
            # discarded by the capture loop. The original ROIs above remain
            # the fallback whenever alignment is weak or impossible.
            best_roi_by_frame: dict[int, tuple[float, np.ndarray]] = {}
            for sharpness, frame_index, _box_index, roi in detected_rois:
                previous = best_roi_by_frame.get(frame_index)
                if previous is None or sharpness > previous[0]:
                    best_roi_by_frame[frame_index] = (sharpness, roi)
            temporal_rois = [
                item[1]
                for _frame_index, item in sorted(
                    best_roi_by_frame.items(),
                    key=lambda entry: entry[1][0],
                    reverse=True,
                )[:8]
            ]
            fused_rois = self._align_and_fuse_rois(temporal_rois)
            for fusion_name, fused_roi in fused_rois:
                cv2.imwrite(
                    str(run_dir / f"waybill_fused_{fusion_name}.png"), fused_roi
                )
                codes = self._read_barcodes(fused_roi)
                fusion_votes.update(set(codes))

                if codes:
                    print(f"Barcode candidates fusion={fusion_name}: {codes}")
            barcode: str | None = None
            frame_ranked = frame_votes.most_common()
            fusion_ranked = fusion_votes.most_common()

            print(
                f"Barcode raw votes candidate #{candidate_index}: "
                f"frames={dict(frame_ranked)}, "
                f"fusion={dict(fusion_ranked)}"
            )

            # --------------------------------------------------
            # 情况 1：
            # 至少两个真实物理帧识别出同一个条码
            # --------------------------------------------------
            if frame_ranked and frame_ranked[0][1] >= 2:

                top_code = frame_ranked[0][0]
                top_votes = frame_ranked[0][1]

                second_votes = frame_ranked[1][1] if len(frame_ranked) > 1 else 0

                if top_votes > second_votes:

                    barcode = top_code

                    print(
                        f"Barcode accepted from physical-frame majority: "
                        f"value={barcode}, "
                        f"votes={top_votes}"
                    )

                else:
                    print(f"Barcode physical-frame vote tied: " f"{dict(frame_ranked)}")

            # --------------------------------------------------
            # 情况 2：
            # 没有明确的物理帧多数结果。
            #
            # 查找“真实帧 + 融合图”共同支持的结果，
            # 而不是只检查 frame_ranked[0]。
            # --------------------------------------------------

            if barcode is None and frame_votes and fusion_votes:

                agreement_codes = [
                    code for code in frame_votes if fusion_votes.get(code, 0) > 0
                ]

                if agreement_codes:

                    agreement_codes.sort(
                        key=lambda code: (
                            frame_votes[code] + fusion_votes[code],
                            frame_votes[code],
                            fusion_votes[code],
                        ),
                        reverse=True,
                    )

                    best_code = agreement_codes[0]

                    best_score = frame_votes[best_code] + fusion_votes[best_code]

                    tied_codes = [
                        code
                        for code in agreement_codes
                        if (frame_votes[code] + fusion_votes[code]) == best_score
                    ]

                    if len(tied_codes) == 1:

                        barcode = best_code

                        print(
                            f"Barcode accepted by frame+fusion agreement: "
                            f"value={barcode}, "
                            f"frame_votes={frame_votes[barcode]}, "
                            f"fusion_votes={fusion_votes[barcode]}"
                        )

                    else:

                        print(f"Barcode frame+fusion agreement tied: " f"{tied_codes}")

            # --------------------------------------------------
            # 情况 3：
            # 真实帧完全没有成功结果。
            #
            # 至少两个独立融合方式得到相同结果才接受。
            # --------------------------------------------------

            if (
                barcode is None
                and not frame_votes
                and fusion_ranked
                and fusion_ranked[0][1] >= 2
            ):
                top_code = fusion_ranked[0][0]
                top_votes = fusion_ranked[0][1]

                second_votes = fusion_ranked[1][1] if len(fusion_ranked) > 1 else 0

                if top_votes > second_votes:

                    barcode = top_code

                    print(
                        f"Barcode accepted from fusion consensus: "
                        f"value={barcode}, "
                        f"votes={top_votes}"
                    )

            # --------------------------------------------------
            # 情况 4：
            # 证据不足或结果冲突
            # --------------------------------------------------

            if barcode is None:

                print(
                    f"Barcode unresolved candidate #{candidate_index}: "
                    "no reliable agreement was found."
                )
            # barcode = decoded_votes.most_common(1)[0][0] if decoded_votes else None
            # if frame_votes:
            #     barcode = frame_votes.most_common(1)[0][0]
            #     print(
            #         f"Barcode vote candidate #{candidate_index}: "
            #         f"{dict(frame_votes.most_common())}; "
            #         f"selected={barcode}"
            #     )
            # elif fusion_votes:
            #     barcode = fusion_votes.most_common(1)[0][0]

            #     print(
            #         f"Barcode fusion fallback candidate #{candidate_index}: "
            #         f"{dict(fusion_votes.most_common())}; "
            #         f"selected={barcode}"
            #     )

            # else:
            #     barcode = None
            detection_to_barcode_s = (
                time.monotonic() - first_waybill_detected_at
                if barcode is not None and first_waybill_detected_at is not None
                else None
            )
            if detection_to_barcode_s is not None:
                print(
                    f"Waybill detection-to-barcode candidate #{candidate_index}: "
                    f"{detection_to_barcode_s:.3f}s"
                )
            has_waybill = waybill_frame_count >= 2 or barcode is not None
            if waybill_frame_count == 0 and len(frames) < 6:
                return WaybillInspectionResult(
                    candidate_index=candidate_index,
                    has_waybill=False,
                    barcode=None,
                    frame_count=len(frames),
                    waybill_frame_count=0,
                    elapsed_s=time.monotonic() - started,
                    error=(
                        f"Only {len(frames)} settled C frames were captured; at least 6 are "
                        "required for a reliable no-waybill decision."
                    ),
                )
            if waybill_frame_count == 1 and barcode is None:
                return WaybillInspectionResult(
                    candidate_index=candidate_index,
                    has_waybill=False,
                    barcode=None,
                    frame_count=len(frames),
                    waybill_frame_count=waybill_frame_count,
                    elapsed_s=time.monotonic() - started,
                    error="Waybill appeared in only one settled frame; result is uncertain, not no-waybill.",
                )
            return WaybillInspectionResult(
                candidate_index=candidate_index,
                has_waybill=has_waybill,
                barcode=barcode,
                frame_count=len(frames),
                waybill_frame_count=waybill_frame_count,
                elapsed_s=time.monotonic() - started,
                detection_to_barcode_s=detection_to_barcode_s,
            )
        except Exception as exc:
            return WaybillInspectionResult(
                candidate_index=candidate_index,
                has_waybill=waybill_frame_count > 0,
                barcode=None,
                frame_count=len(frames),
                waybill_frame_count=waybill_frame_count,
                elapsed_s=time.monotonic() - started,
                error=str(exc),
            )

    def _save_video(self, path: Path, frames: list[np.ndarray]) -> None:
        if not frames:
            print("C-camera video was not saved: no frame was captured.")
            return
        height, width = frames[0].shape[:2]
        fps = max(1.0, 1.0 / self.capture_interval_s)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            print(f"C-camera video writer could not be opened: {path}")
            return
        try:
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
        finally:
            writer.release()
        print(f"C-camera video saved: {path}")

    def _detect_waybills(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        detections: list[tuple[float, tuple[int, int, int, int]]] = []
        for result in self._model(image, conf=self.confidence, verbose=False):
            for box in result.boxes:
                if int(box.cls[0]) != self.waybill_class_id:
                    continue
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].cpu().numpy())
                clipped = (
                    max(0, min(width, x1)),
                    max(0, min(height, y1)),
                    max(0, min(width, x2)),
                    max(0, min(height, y2)),
                )
                detections.append((confidence, clipped))
        detections.sort(key=lambda item: item[0], reverse=True)
        return [box for _confidence, box in detections]

    @staticmethod
    def _crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2]

    @classmethod
    def _align_and_fuse_rois(
        cls, rois: list[np.ndarray]
    ) -> list[tuple[str, np.ndarray]]:
        """Register stationary waybill crops and create conservative fusion views."""
        if len(rois) < 2:
            return []

        rectified = [cls._rectify_waybill(roi) for roi in rois if roi.size]
        if len(rectified) < 2:
            return []
        reference = rectified[0]
        ref_height, ref_width = reference.shape[:2]
        if ref_height < 32 or ref_width < 32:
            return []

        ref_aspect = ref_width / float(ref_height)
        scale = min(1.0, 700.0 / max(ref_width, ref_height))
        small_size = (max(32, int(ref_width * scale)), max(32, int(ref_height * scale)))

        def ecc_view(image: np.ndarray) -> np.ndarray:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, small_size, interpolation=cv2.INTER_AREA)
            return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        template = ecc_view(reference)
        aligned: list[np.ndarray] = [reference]
        alignment_scores: list[float] = [1.0]
        for candidate in rectified[1:]:
            height, width = candidate.shape[:2]
            aspect = width / float(max(1, height))
            if not 0.70 <= aspect / ref_aspect <= 1.30:
                continue
            resized = cv2.resize(
                candidate, (ref_width, ref_height), interpolation=cv2.INTER_CUBIC
            )
            moving = ecc_view(resized)
            warp_small = np.eye(2, 3, dtype=np.float32)
            try:
                correlation, warp_small = cv2.findTransformECC(
                    template,
                    moving,
                    warp_small,
                    cv2.MOTION_AFFINE,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-4),
                    None,
                    5,
                )
            except cv2.error:
                continue
            if not np.isfinite(correlation) or correlation < 0.55:
                continue
            warp_full = warp_small.copy()
            warp_full[:, 2] /= max(scale, 1e-6)
            registered = cv2.warpAffine(
                resized,
                warp_full,
                (ref_width, ref_height),
                flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT101,
            )
            aligned.append(registered)
            alignment_scores.append(float(correlation))

        if len(aligned) < 2:
            print(
                "Waybill temporal fusion skipped: fewer than 2 ROIs aligned reliably."
            )
            return []

        image_stack = np.stack(aligned).astype(np.float32)
        sharpness = np.array(
            [
                max(
                    1.0,
                    float(
                        cv2.Laplacian(
                            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                        ).var()
                    ),
                )
                for image in aligned
            ],
            dtype=np.float32,
        )
        weights = sharpness * np.asarray(alignment_scores, dtype=np.float32)
        weights /= max(float(weights.sum()), 1e-6)
        weighted = np.clip(
            np.sum(image_stack * weights[:, None, None, None], axis=0), 0, 255
        ).astype(np.uint8)

        # A median view suppresses JPEG/noise artefacts; a local-focus view
        # preserves whichever registered frame contains the strongest bar edge.
        median = np.median(image_stack, axis=0).astype(np.uint8)
        local_focus: list[np.ndarray] = []
        for image in aligned:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edge_energy = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
            local_focus.append(cv2.GaussianBlur(edge_energy, (0, 0), 5.0))
        focus_index = np.argmax(np.stack(local_focus), axis=0)
        focus = np.empty_like(aligned[0])
        for index, image in enumerate(aligned):
            mask = focus_index == index
            focus[mask] = image[mask]

        print(
            "Waybill temporal fusion: "
            f"input={len(rois)} aligned={len(aligned)} "
            f"ecc_min={min(alignment_scores):.3f}"
        )
        return [("weighted", weighted), ("median", median), ("focus", focus)]

    @staticmethod
    def _enhance_images(image: np.ndarray) -> Iterator[np.ndarray]:
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3
            else image.astype(np.uint8, copy=False)
        )
        # 1. 原图
        yield image

        # 2. 灰度图
        yield gray
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(gray)

        yield clahe
        # 4. 锐化
        blur = cv2.GaussianBlur(
            clahe,
            (0, 0),
            1.0,
        )

        sharp = cv2.addWeighted(
            clahe,
            1.9,
            blur,
            -0.9,
            0,
        )

        yield sharp
        # --------------------------------------------------

        # 根据 ROI 尺寸动态决定是否放大
        # --------------------------------------------------

        height, width = gray.shape[:2]
        max_side = max(height, width)

        # 很小的条码 ROI
        if max_side < 450:

            clahe_3x = cv2.resize(
                clahe,
                None,
                fx=3.0,
                fy=3.0,
                interpolation=cv2.INTER_CUBIC,
            )

            yield clahe_3x

            sharp_3x = cv2.resize(
                sharp,
                None,
                fx=3.0,
                fy=3.0,
                interpolation=cv2.INTER_CUBIC,
            )

            yield sharp_3x

        # 中等大小条码 ROI
        elif max_side < 700:

            clahe_2x = cv2.resize(
                clahe,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_CUBIC,
            )

            yield clahe_2x

            sharp_2x = cv2.resize(
                sharp,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_CUBIC,
            )

            yield sharp_2x

        # --------------------------------------------------
        # 前面的常规方案全部失败以后
        # 再尝试去噪、二值化等更强处理
        # --------------------------------------------------

        # 5. 双边滤波
        denoised = cv2.bilateralFilter(
            gray,
            5,
            30,
            30,
        )

        yield denoised

        # 6. Otsu 二值化
        _, binary = cv2.threshold(
            sharp,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        yield binary

        # 7. 自适应二值化
        adaptive = cv2.adaptiveThreshold(
            clahe,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )

        yield adaptive

        # 8. Sauvola 风格局部阈值
        normalized = gray.astype(np.float32) / 255.0

        local_mean = cv2.boxFilter(
            normalized,
            cv2.CV_32F,
            (31, 31),
            normalize=True,
        )

        local_square = cv2.boxFilter(
            normalized * normalized,
            cv2.CV_32F,
            (31, 31),
            normalize=True,
        )

        local_std = np.sqrt(
            np.maximum(
                local_square - local_mean * local_mean,
                0.0,
            )
        )

        sauvola_threshold = local_mean * (1.0 + 0.22 * (local_std / 0.5 - 1.0))

        sauvola = np.where(
            normalized > sauvola_threshold,
            255,
            0,
        ).astype(np.uint8)

        yield sauvola

    @staticmethod
    def _order_quad(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(4, 2)
        ordered = np.zeros((4, 2), dtype=np.float32)
        sums = points.sum(axis=1)
        differences = np.diff(points, axis=1).ravel()
        ordered[0] = points[np.argmin(sums)]  # top-left
        ordered[2] = points[np.argmax(sums)]  # bottom-right
        ordered[1] = points[np.argmin(differences)]  # top-right
        ordered[3] = points[np.argmax(differences)]  # bottom-left
        return ordered

    @classmethod
    def _four_point_transform(cls, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        top_left, top_right, bottom_right, bottom_left = cls._order_quad(points)
        width = int(
            max(
                np.linalg.norm(bottom_right - bottom_left),
                np.linalg.norm(top_right - top_left),
            )
        )
        height = int(
            max(
                np.linalg.norm(top_right - bottom_right),
                np.linalg.norm(top_left - bottom_left),
            )
        )
        if width < 8 or height < 8:
            return image
        destination = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(
            np.array(
                [top_left, top_right, bottom_right, bottom_left], dtype=np.float32
            ),
            destination,
        )
        return cv2.warpPerspective(
            image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
        )

    @classmethod
    def _rectify_waybill(cls, image: np.ndarray) -> np.ndarray:
        """Deskew the paper inside a loose YOLO crop when its outline is visible."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 130)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        )
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image_area = float(image.shape[0] * image.shape[1])
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(contour) < image_area * 0.30:
                break
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            if len(polygon) == 4 and cv2.isContourConvex(polygon):
                return cls._four_point_transform(image, polygon[:, 0, :])
        return image

    @staticmethod
    def _add_quiet_zone(image: np.ndarray) -> np.ndarray:
        """Add the white margin required by linear barcode start/stop patterns."""
        height, width = image.shape[:2]
        horizontal = max(12, int(width * 0.12))
        vertical = max(8, int(height * 0.08))
        return cv2.copyMakeBorder(
            image,
            vertical,
            vertical,
            horizontal,
            horizontal,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    @classmethod
    def _find_linear_barcode_regions(cls, image: np.ndarray) -> list[np.ndarray]:
        """Locate dense vertical-line groups and return deskewed barcode crops."""
        regions: list[tuple[float, np.ndarray]] = []

        # OpenCV's barcode detector is useful as a locator even when its own
        # decoder has insufficient pixels. Its quadrilaterals also give us a
        # more accurate deskew than an axis-aligned contour crop.
        try:
            detected, points = cv2.barcode_BarcodeDetector().detect(image)
        except (AttributeError, cv2.error):
            detected, points = False, None
        if detected and points is not None:
            for points_item in np.asarray(points).reshape(-1, 4, 2):
                crop = cls._four_point_transform(image, points_item)
                if crop.shape[0] > crop.shape[1]:
                    crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                regions.append(
                    (float(crop.shape[0] * crop.shape[1]), cls._add_quiet_zone(crop))
                )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale = max(1.0, min(image.shape[:2]) / 500.0)
        blackhat_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(15, int(25 * scale)) | 1, max(3, int(7 * scale)) | 1)
        )
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
        gradient = cv2.Scharr(blackhat, cv2.CV_32F, 1, 0)
        gradient = cv2.convertScaleAbs(gradient)
        gradient = cv2.GaussianBlur(gradient, (5, 5), 0)
        _, mask = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(17, int(31 * scale)) | 1, max(3, int(7 * scale)) | 1)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = float(image.shape[0] * image.shape[1])
        for contour in contours:
            rectangle = cv2.minAreaRect(contour)
            rect_width, rect_height = rectangle[1]
            long_side, short_side = max(rect_width, rect_height), min(
                rect_width, rect_height
            )
            area = rect_width * rect_height
            if short_side < 10 or long_side / max(short_side, 1.0) < 1.8:
                continue
            if area < image_area * 0.002 or area > image_area * 0.45:
                continue
            box = cv2.boxPoints(rectangle)
            center = box.mean(axis=0)
            box = center + (box - center) * np.array([1.10, 1.35], dtype=np.float32)
            box[:, 0] = np.clip(box[:, 0], 0, image.shape[1] - 1)
            box[:, 1] = np.clip(box[:, 1], 0, image.shape[0] - 1)
            crop = cls._four_point_transform(image, box)
            if crop.shape[0] > crop.shape[1]:
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            regions.append((area, cls._add_quiet_zone(crop)))
        regions.sort(key=lambda item: item[0], reverse=True)
        return [region for _area, region in regions[:8]]

    @staticmethod
    def _normalize_barcode_text(value: object) -> str | None:
        text = str(value or "").strip()
        if len(text) < 13 or "\ufffd" in text:
            return None
        return text

    # def _read_barcodes(self, image: np.ndarray) -> list[str]:
    #     rectified = self._rectify_waybill(image)

    #     # --------------------------------------------------
    #     # 第一阶段：
    #     # OpenCV BarcodeDetector 快速尝试
    #     # --------------------------------------------------

    #     try:
    #         decoded, _points, _straight = cv2.barcode_BarcodeDetector().detectAndDecode(
    #             rectified
    #         )

    #         if isinstance(decoded, str):
    #             text = self._normalize_barcode_text(decoded)

    #             if text:
    #                 return [text]

    #         if isinstance(decoded, (tuple, list)):
    #             codes = {
    #                 normalized
    #                 for value in decoded
    #                 if (normalized := self._normalize_barcode_text(value)) is not None
    #             }

    #             if codes:
    #                 return sorted(codes)

    #     except (AttributeError, cv2.error):
    #         pass

    #     # --------------------------------------------------
    #     # 第二阶段：
    #     # 定位面单中的一维条码区域
    #     # --------------------------------------------------

    #     regions = self._find_linear_barcode_regions(rectified)

    #     # --------------------------------------------------
    #     # 如果定位算法没有找到或定位不准确，
    #     # 完整面单 ROI 继续作为兜底
    #     # --------------------------------------------------

    #     regions.extend(
    #         [
    #             self._add_quiet_zone(rectified),
    #             self._add_quiet_zone(
    #                 cv2.rotate(
    #                     rectified,
    #                     cv2.ROTATE_90_CLOCKWISE,
    #                 )
    #             ),
    #         ]
    #     )

    #     formats = self._zxingcpp.BarcodeFormat.AllLinear

    #     # --------------------------------------------------
    #     # 第三阶段：
    #     # 每个条码区域按逐级增强方式尝试
    #     # --------------------------------------------------
    #     for region_index, region in enumerate(regions):
    #         for enhance_index, candidate in enumerate(self._enhance_images(region)):
    #             codes: set[str] = set()
    #             try:
    #                 results = self._zxingcpp.read_barcodes(
    #                     candidate,
    #                     formats=formats,
    #                     try_rotate=True,
    #                     try_downscale=False,
    #                     try_invert=True,
    #                 )

    #             except Exception as exc:
    #                 print(
    #                     f"ZXingCPP decode failed "
    #                     f"region={region_index} "
    #                     f"enhance={enhance_index}: {exc}"
    #                 )
    #                 continue

    #             for result in results:

    #                 text = self._normalize_barcode_text(result.text)

    #                 if text:
    #                     codes.add(text)

    #             # 一旦当前增强版本成功，
    #             # 立即停止后续增强和放大
    #             if codes:

    #                 print(
    #                     f"Barcode decoded: "
    #                     f"region={region_index} "
    #                     f"enhance={enhance_index} "
    #                     f"value={sorted(codes)}"
    #                 )
    #                 return sorted(codes)

    #     return []
    def _read_barcodes(self, image: np.ndarray) -> list[str]:
        rectified = self._rectify_waybill(image)

        # --------------------------------------------------
        # 第一阶段：
        # 定位可能的一维条码区域
        #
        # _find_linear_barcode_regions() 内部仍然可以使用
        # OpenCV BarcodeDetector.detect()，但只用于定位，
        # 不使用 OpenCV 进行最终解码。
        # --------------------------------------------------

        regions = self._find_linear_barcode_regions(rectified)

        # --------------------------------------------------
        # 第二阶段：
        # 如果条码定位失败或定位不准确，
        # 完整面单 ROI 作为兜底。
        #
        # 同时加入旋转 90° 的版本。
        # --------------------------------------------------

        regions.extend(
            [
                self._add_quiet_zone(rectified),
                self._add_quiet_zone(
                    cv2.rotate(
                        rectified,
                        cv2.ROTATE_90_CLOCKWISE,
                    )
                ),
            ]
        )

        # 当前业务读取一维条形码
        formats = self._zxingcpp.BarcodeFormat.Code128

        # --------------------------------------------------
        # 第三阶段：
        # ZXingCPP 逐区域、逐增强版本尝试。
        #
        # _enhance_images() 是 yield 生成器，
        # 一旦识别成功，后面的增强不会继续计算。
        # --------------------------------------------------

        for region_index, region in enumerate(regions):

            for enhance_index, candidate in enumerate(self._enhance_images(region)):

                try:
                    results = self._zxingcpp.read_barcodes(
                        candidate,
                        formats=formats,
                        try_rotate=True,
                        try_downscale=False,
                        try_invert=True,
                    )

                except Exception as exc:
                    print(
                        f"ZXingCPP decode failed "
                        f"region={region_index} "
                        f"enhance={enhance_index}: {exc}"
                    )
                    continue

                codes: set[str] = set()

                for result in results:
                    text = self._normalize_barcode_text(result.text)

                    if text:
                        codes.add(text)

                if codes:
                    decoded = sorted(codes)

                    print(
                        f"Barcode decoded by ZXingCPP: "
                        f"region={region_index} "
                        f"enhance={enhance_index} "
                        f"value={decoded}"
                    )

                    return decoded

        return []

    def _read_barcode(self, image: np.ndarray) -> str | None:
        """Compatibility wrapper for callers that only need one value."""
        codes = self._read_barcodes(image)
        return codes[0] if codes else None

    def _handle_future(self, future: Future[WaybillInspectionResult]) -> None:
        try:
            result = future.result()
        except Exception as exc:
            print(f"Waybill inspection worker failed: {exc}")
            return
        self._append_result(result)
        self._result_callback(result)

    def _append_result(self, result: WaybillInspectionResult) -> None:
        payload = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "candidate_index": result.candidate_index,
            "has_waybill": result.has_waybill,
            "barcode": result.barcode,
            "frame_count": result.frame_count,
            "waybill_frame_count": result.waybill_frame_count,
            "elapsed_s": round(result.elapsed_s, 3),
            "detection_to_barcode_s": (
                round(result.detection_to_barcode_s, 3)
                if result.detection_to_barcode_s is not None
                else None
            ),
            "error": result.error,
        }
        with (self.output_dir / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _print_result(result: WaybillInspectionResult) -> None:
        prefix = f"Waybill inspection candidate #{result.candidate_index}"
        if result.error:
            print(f"{prefix}: ERROR {result.error}")
        elif not result.has_waybill:
            print(
                f"{prefix}: no waybill ({result.frame_count} frame(s), {result.elapsed_s:.2f}s)."
            )
        elif result.barcode:
            detection_to_barcode = (
                f", detection-to-barcode={result.detection_to_barcode_s:.3f}s"
                if result.detection_to_barcode_s is not None
                else ""
            )
            print(
                f"{prefix}: waybill detected, barcode={result.barcode} "
                f"({result.waybill_frame_count}/{result.frame_count} frame(s), "
                f"total={result.elapsed_s:.2f}s{detection_to_barcode})."
            )
        else:
            print(
                f"{prefix}: waybill detected, but barcode was unreadable "
                f"({result.waybill_frame_count}/{result.frame_count} frame(s), {result.elapsed_s:.2f}s)."
            )
