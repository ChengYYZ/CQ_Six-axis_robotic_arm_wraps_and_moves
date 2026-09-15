from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from project0714_grasp.waybill_inspection import AsyncWaybillInspector


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the existing waybill detector and barcode decoder on local images."
    )
    parser.add_argument("input", help="Image file or directory containing test images")
    parser.add_argument(
        "--model",
        default=str(project_root / "weights" / "best.pt"),
        help="YOLO waybill detector weights",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")
    parser.add_argument(
        "--barcode-model",
        default=str(project_root / "weights" / "barcode_weights" / "best.pt"),
        help="YOLO OBB weights used to locate Code128 barcodes inside the waybill ROI",
    )
    parser.add_argument(
        "--barcode-conf",
        type=float,
        default=0.25,
        help="Barcode OBB confidence threshold",
    )
    parser.add_argument(
        "--mode",
        choices=("sequence", "single"),
        default="sequence",
        help=(
            "sequence: treat all images as frames of one package and use multi-frame voting; "
            "single: test every image independently"
        ),
    )
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories")
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "calibration_suite" / "workspace" / "offline_waybill_test"),
        help="Directory for annotated evidence, crops, fused images and summary.json",
    )
    return parser


def find_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_SUFFIXES else []
    if not input_path.is_dir():
        return []
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def result_payload(label: str, image_paths: list[Path], result: object) -> dict[str, object]:
    return {
        "label": label,
        "images": [str(path.resolve()) for path in image_paths],
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


def build_batch_report(summaries: list[dict[str, object]]) -> dict[str, object]:
    total = len(summaries)
    successful = [item for item in summaries if item.get("barcode")]
    failed = [item for item in summaries if not item.get("barcode")]
    elapsed_values = [float(item["elapsed_s"]) for item in summaries]
    successful_elapsed = [float(item["elapsed_s"]) for item in successful]
    detection_to_barcode = [
        float(item["detection_to_barcode_s"])
        for item in successful
        if item.get("detection_to_barcode_s") is not None
    ]

    def average(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    return {
        "total_images": total,
        "successful_images": len(successful),
        "failed_images": len(failed),
        "success_rate_percent": round(len(successful) * 100.0 / total, 2)
        if total
        else 0.0,
        "average_processing_time_s_all_images": average(elapsed_values),
        "average_processing_time_s_successful_images": average(successful_elapsed),
        "average_detection_to_barcode_s_successful_images": average(
            detection_to_barcode
        ),
        "failed_filenames": [str(item["label"]) for item in failed],
    }


def write_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    fieldnames = [
        "label",
        "barcode",
        "has_waybill",
        "frame_count",
        "waybill_frame_count",
        "elapsed_s",
        "detection_to_barcode_s",
        "error",
        "image_path",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            images = item.get("images") or []
            writer.writerow(
                {
                    "label": item.get("label"),
                    "barcode": item.get("barcode"),
                    "has_waybill": item.get("has_waybill"),
                    "frame_count": item.get("frame_count"),
                    "waybill_frame_count": item.get("waybill_frame_count"),
                    "elapsed_s": item.get("elapsed_s"),
                    "detection_to_barcode_s": item.get(
                        "detection_to_barcode_s"
                    ),
                    "error": item.get("error"),
                    "image_path": images[0] if images else "",
                }
            )


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    image_paths = find_images(input_path, args.recursive)
    if not image_paths:
        raise SystemExit(f"No supported images found: {input_path}")

    frames: list[tuple[Path, object]] = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue
        frames.append((path, image))
    if not frames:
        raise SystemExit("No image could be decoded by OpenCV.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    inspector = AsyncWaybillInspector(
        camera_ip="offline",
        username="offline",
        password="offline",
        model_path=Path(args.model).expanduser().resolve(),
        barcode_model_path=Path(args.barcode_model).expanduser().resolve(),
        output_dir=output_dir,
        confidence=args.conf,
        barcode_confidence=args.barcode_conf,
        allow_single_frame_result=args.mode == "single",
    )
    summaries: list[dict[str, object]] = []
    try:
        if args.mode == "sequence":
            paths = [item[0] for item in frames]
            result = inspector._inspect_frames(1, [item[1] for item in frames])
            inspector._print_result(result)
            summaries.append(result_payload("sequence", paths, result))
        else:
            for index, (path, image) in enumerate(frames, start=1):
                result = inspector._inspect_frames(index, [image])
                inspector._print_result(result)
                summaries.append(result_payload(path.name, [path], result))
    finally:
        inspector.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = build_batch_report(summaries)
    report_path = output_dir / "batch_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_path = output_dir / "results.csv"
    write_csv(csv_path, summaries)

    print(f"Summary written to: {summary_path}")
    print(f"CSV written to: {csv_path}")
    print(f"Batch report written to: {report_path}")
    print("\nBatch result")
    print(f"  Total images: {report['total_images']}")
    print(f"  Successful: {report['successful_images']}")
    print(f"  Failed: {report['failed_images']}")
    print(f"  Success rate: {report['success_rate_percent']:.2f}%")
    print(
        "  Average processing time (all images): "
        f"{report['average_processing_time_s_all_images']}s"
    )
    print(
        "  Average detection-to-barcode time (successful images): "
        f"{report['average_detection_to_barcode_s_successful_images']}s"
    )
    if report["failed_filenames"]:
        print("  Failed filenames:")
        for filename in report["failed_filenames"]:
            print(f"    {filename}")
    return 0 if any(item["barcode"] for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
