from __future__ import annotations

import sys
import unittest
from pathlib import Path


CALIBRATION_SUITE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CALIBRATION_SUITE.parent
for path in (PROJECT_ROOT, CALIBRATION_SUITE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from system_desktop import (
    DEFAULT_MAIN_OPTIONS,
    NO_BARCODE_TEXT,
    barcode_event_from_huaray_payload,
    barcode_from_huaray_payload,
    barcode_from_log_line,
    resets_barcode_display,
)


class BarcodeLogParsingTests(unittest.TestCase):
    def test_robot_error_is_not_a_barcode(self) -> None:
        line = (
            "Controller error: Robot move rejected before start: "
            "{'ec': -50021, 'message': '指定conf参数下目标点无解'}"
        )
        self.assertIsNone(barcode_from_log_line(line))

    def test_tcp_reader_event_is_accepted(self) -> None:
        self.assertEqual(
            barcode_from_log_line("Top/front barcode received: 434857762912875"),
            ("D点TCP条码", "434857762912875"),
        )

    def test_waybill_result_event_is_accepted(self) -> None:
        line = "Waybill inspection package pkg-1 combined: final=barcode=CODE-128, errors=none."
        self.assertEqual(barcode_from_log_line(line), ("条码", "CODE-128"))

    def test_unrelated_barcode_assignment_is_ignored(self) -> None:
        self.assertIsNone(barcode_from_log_line("debug barcode=temporary-value"))

    def test_suction_on_resets_barcode_display(self) -> None:
        self.assertTrue(
            resets_barcode_display(
                "Suction command ON -> DO3_6 ON."
            )
        )
        self.assertFalse(
            resets_barcode_display(
                "Barcode reader window remains open at A*: accepting late results."
            )
        )
        self.assertFalse(
            resets_barcode_display(
                "Barcode display cycle reset at A*: awaiting barcode for the next package."
            )
        )

    def test_frontend_rejects_degraded_placement_by_default(self) -> None:
        self.assertNotIn("--allow-degraded-placement", DEFAULT_MAIN_OPTIONS)

    def test_no_barcode_placeholder_is_explicit(self) -> None:
        self.assertEqual(NO_BARCODE_TEXT, "暂未识别出条形码")

    def test_huaray_sdk_payload_supplies_detected_code(self) -> None:
        payload = {
            "barcodes": [
                {"code": "SF5152983800051", "points": [[1, 2], [3, 4]]},
            ]
        }
        self.assertEqual(barcode_from_huaray_payload(payload), "SF5152983800051")

    def test_huaray_sdk_payload_without_code_is_ignored(self) -> None:
        self.assertIsNone(barcode_from_huaray_payload({"barcodes": []}))
        self.assertIsNone(barcode_from_huaray_payload({"barcodes": [{"code": ""}]}))

    def test_huaray_persisted_detection_survives_empty_current_frame(self) -> None:
        payload = {
            "barcodes": [],
            "last_detection": {
                "sequence": 7,
                "captured_at": 123.0,
                "barcodes": [{"code": "465108210636573", "points": []}],
            },
        }
        self.assertEqual(
            barcode_event_from_huaray_payload(payload),
            (7, "465108210636573"),
        )


if __name__ == "__main__":
    unittest.main()
