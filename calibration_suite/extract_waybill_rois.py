from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Detect waybills with YOLO and save their cropped ROIs."
    )
    parser.add_argument("input",
                        default=r"G:\CQ\datasets\barcode",
                        help="Input image or directory")
    parser.add_argument(
        "--model",
        default=str(project_root / "weights" / "best.pt"),
        help="YOLO waybill detector weights",
    )
    parser.add_argument(
        "--output-dir",
        default=r"G:\CQ\datasets\barcode\barcode9.1",
        help="Directory used for ROI images and reports",
    )
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Padding around the detected box as a fraction of box size",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--all-detections",
        action="store_true",
        help="Save every matching detection instead of only the most confident one",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Also save source images with detection boxes",
    )
    return parser


def find_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_SUFFIXES else []
    if not input_path.is_dir():
        return []
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def padded_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    padding = max(0.0, float(padding))
    pad_x = int(round(max(1, x2 - x1) * padding))
    pad_y = int(round(max(1, y2 - y1) * padding))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    )


def unique_output_path(directory: Path, source: Path, detection_index: int) -> Path:
    suffix = source.suffix.lower() if source.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
    base = f"{source.stem}_waybill"
    if detection_index > 1:
        base += f"_{detection_index}"
    candidate = directory / f"{base}{suffix}"
    duplicate_index = 2
    while candidate.exists():
        candidate = directory / f"{base}_{duplicate_index}{suffix}"
        duplicate_index += 1
    return candidate


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fieldnames = [
        "source_filename",
        "source_path",
        "status",
        "confidence",
        "class_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "crop_width",
        "crop_height",
        "roi_path",
        "elapsed_s",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    image_paths = find_images(input_path, args.recursive)
    if not image_paths:
        raise SystemExit(f"No supported images found: {input_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    roi_dir = output_dir / "rois"
    annotated_dir = output_dir / "annotated"
    roi_dir.mkdir(parents=True, exist_ok=True)
    if args.save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(Path(args.model).expanduser().resolve()))
    records: list[dict[str, object]] = []
    detected_source_files: set[str] = set()
    failed_filenames: list[str] = []

    for image_index, path in enumerate(image_paths, start=1):
        started = time.perf_counter()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            failed_filenames.append(path.name)
            records.append(
                {
                    "source_filename": path.name,
                    "source_path": str(path),
                    "status": "unreadable",
                    "confidence": "",
                    "class_id": "",
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "crop_width": "",
                    "crop_height": "",
                    "roi_path": "",
                    "elapsed_s": round(time.perf_counter() - started, 4),
                    "error": "OpenCV could not decode the image",
                }
            )
            continue

        height, width = image.shape[:2]
        result = model.predict(image, conf=args.conf, verbose=False)[0]
        detections: list[tuple[float, tuple[int, int, int, int]]] = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id != args.class_id:
                continue
            confidence = float(box.conf[0])
            values = box.xyxy[0].cpu().numpy()
            raw_box = tuple(int(round(float(value))) for value in values)
            detections.append((confidence, raw_box))
        detections.sort(key=lambda item: item[0], reverse=True)
        if not args.all_detections:
            detections = detections[:1]

        if not detections:
            failed_filenames.append(path.name)
            records.append(
                {
                    "source_filename": path.name,
                    "source_path": str(path),
                    "status": "not_detected",
                    "confidence": "",
                    "class_id": args.class_id,
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "crop_width": "",
                    "crop_height": "",
                    "roi_path": "",
                    "elapsed_s": round(time.perf_counter() - started, 4),
                    "error": "No waybill detection passed the confidence threshold",
                }
            )
            print(f"[{image_index}/{len(image_paths)}] MISS {path.name}")
            continue

        detected_source_files.add(path.name)
        annotated = image.copy() if args.save_annotated else None
        for detection_index, (confidence, raw_box) in enumerate(detections, start=1):
            x1, y1, x2, y2 = padded_box(
                raw_box, width, height, args.padding
            )
            roi = image[y1:y2, x1:x2]
            roi_path = unique_output_path(roi_dir, path, detection_index)
            saved = bool(roi.size) and cv2.imwrite(str(roi_path), roi)
            status = "saved" if saved else "save_failed"
            error = "" if saved else "ROI was empty or OpenCV failed to save it"
            if not saved:
                failed_filenames.append(path.name)
            if annotated is not None:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"waybill {confidence:.3f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            records.append(
                {
                    "source_filename": path.name,
                    "source_path": str(path),
                    "status": status,
                    "confidence": round(confidence, 6),
                    "class_id": args.class_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "crop_width": x2 - x1,
                    "crop_height": y2 - y1,
                    "roi_path": str(roi_path) if saved else "",
                    "elapsed_s": round(time.perf_counter() - started, 4),
                    "error": error,
                }
            )
        if annotated is not None:
            cv2.imwrite(str(annotated_dir / path.name), annotated)
        print(
            f"[{image_index}/{len(image_paths)}] OK {path.name}: "
            f"{len(detections)} ROI(s)"
        )

    successful_records = [item for item in records if item["status"] == "saved"]
    elapsed_values = [float(item["elapsed_s"]) for item in records]
    report = {
        "input": str(input_path),
        "model": str(Path(args.model).expanduser().resolve()),
        "confidence_threshold": args.conf,
        "padding": args.padding,
        "total_images": len(image_paths),
        "detected_images": len(detected_source_files),
        "missed_images": len(image_paths) - len(detected_source_files),
        "saved_rois": len(successful_records),
        "detection_rate_percent": round(
            len(detected_source_files) * 100.0 / len(image_paths), 2
        ),
        "average_time_s_per_record": round(
            sum(elapsed_values) / len(elapsed_values), 4
        )
        if elapsed_values
        else None,
        "failed_filenames": sorted(set(failed_filenames)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "records.csv", records)

    print("\nWaybill ROI extraction result")
    print(f"  Total images: {report['total_images']}")
    print(f"  Detected images: {report['detected_images']}")
    print(f"  Detection rate: {report['detection_rate_percent']:.2f}%")
    print(f"  Saved ROIs: {report['saved_rois']}")
    print(f"  ROI directory: {roi_dir}")
    print(f"  Report: {output_dir / 'report.json'}")
    return 0 if successful_records else 1


if __name__ == "__main__":
    raise SystemExit(main())
