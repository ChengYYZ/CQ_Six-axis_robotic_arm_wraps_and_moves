from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


CALIBRATION_SUITE = Path(__file__).resolve().parents[1]
if str(CALIBRATION_SUITE) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_SUITE))

from project0714_grasp.waybill_inspection import (  # noqa: E402
    AsyncWaybillInspector,
    WaybillInspectionResult,
)


class WaybillAsyncPipelineTests(unittest.TestCase):
    def test_missing_waybill_roi_falls_back_to_sharpest_full_frame(self) -> None:
        inspector = AsyncWaybillInspector.__new__(AsyncWaybillInspector)
        inspector.fast_single_frame_barcode = True
        flat = np.full((80, 120, 3), 127, dtype=np.uint8)
        checker = np.indices((80, 120)).sum(axis=0) % 2
        sharp = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        decoded_inputs: list[np.ndarray] = []

        with tempfile.TemporaryDirectory() as temporary:
            inspector.output_dir = Path(temporary)
            inspector._save_and_extract_video = lambda _run_dir, frames: frames
            inspector._detect_waybills = lambda _frame: []

            def decode(image: np.ndarray) -> list[str]:
                decoded_inputs.append(image)
                return ["CODE-128"] if np.array_equal(image, sharp) else []

            inspector._read_barcodes_fast_full_frame = decode
            with patch.object(cv2, "imwrite", return_value=True):
                result = inspector._inspect_frames(1, [flat, sharp])

        self.assertEqual(result.barcode, "CODE-128")
        self.assertTrue(result.has_waybill)
        self.assertEqual(len(decoded_inputs), 1)
        np.testing.assert_array_equal(decoded_inputs[0], sharp)

    def test_main_and_side_results_are_combined_once(self) -> None:
        inspector = AsyncWaybillInspector.__new__(AsyncWaybillInspector)
        inspector._lock = threading.Lock()
        inspector._view_results = {}
        main = WaybillInspectionResult(
            1, False, None, 3, 0, 0.2,
            inspection_id="pkg-1-test", view_name="main",
        )
        side = WaybillInspectionResult(
            1, True, "CODE-128", 3, 1, 0.3,
            inspection_id="pkg-1-test", view_name="side",
        )

        with patch("builtins.print") as print_mock:
            inspector._record_view_result(main)
            inspector._record_view_result(side)

        combined_lines = [
            str(call.args[0]) for call in print_mock.call_args_list
            if "combined:" in str(call.args[0])
        ]
        self.assertEqual(len(combined_lines), 1)
        self.assertIn("final=barcode=CODE-128", combined_lines[0])


if __name__ == "__main__":
    unittest.main()
