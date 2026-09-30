from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


CALIBRATION_SUITE = Path(__file__).resolve().parents[1]
if str(CALIBRATION_SUITE) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_SUITE))

from runtime_run_log import (
    analyze_runtime_log,
    analyze_runtime_log_directory,
    close_active_run_log,
    start_run_log,
)


class RuntimeRunLogTests(unittest.TestCase):
    def test_analyzer_counts_operational_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "run.log"
            log_path.write_text(
                "\n".join(
                    (
                        "Completed batch candidate #1.",
                        "CONTROLLER_ERROR_EVENT code=-50021 motion=movej use_current_conf=1",
                        "PLAN_REJECTION_SUMMARY candidate=1 accepted=2 rejected=4 "
                        "reasons=pickup_rotation=3,unused_cup_collision=1 examples=none",
                        "At functional placement D; placement_quality=degraded",
                        "No new top/front barcode matched this D-point cycle before timeout.",
                        "Batch motion stopped at candidate #2: failed",
                    )
                ),
                encoding="utf-8",
            )

            summary_path = analyze_runtime_log(log_path)
            summary = summary_path.read_text(encoding="utf-8")

        self.assertIn("| `completed_packages` | 1 |", summary)
        self.assertIn("| `motion_stops` | 1 |", summary)
        self.assertIn("| `controller_conf_rejections` | 1 |", summary)
        self.assertIn("| `large_pickup_rotations_rejected` | 3 |", summary)
        self.assertIn("| `degraded_placements` | 1 |", summary)
        self.assertIn("| `barcode_timeouts` | 1 |", summary)

    def test_session_saves_stdout_stderr_and_redacts_password(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = start_run_log(
                directory,
                argv=["demo.py", "--waybill-camera-password", "secret-value"],
            )
            try:
                print("stdout marker")
                print("stderr marker", file=sys.stderr)
            finally:
                close_active_run_log(0)
            content = session.log_path.read_text(encoding="utf-8")

        self.assertIn("stdout marker", content)
        self.assertIn("stderr marker", content)
        self.assertIn("<redacted>", content)
        self.assertNotIn("secret-value", content)

    def test_directory_analyzer_tracks_cross_run_trends(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "surface_grasp_001.log").write_text(
                "Completed batch candidate #1.\nRobot move rejected: -50021\n",
                encoding="utf-8",
            )
            (root / "surface_grasp_002.log").write_text(
                "Batch motion stopped at candidate #2: failed\n"
                "Rejecting suction plan with excessive A*-to-pickup rotation\n",
                encoding="utf-8",
            )

            trend_path = analyze_runtime_log_directory(root)
            trend = trend_path.read_text(encoding="utf-8")

        self.assertIn("Runs included: 2", trend)
        self.assertIn("Completed packages: 1", trend)
        self.assertIn("Motion stops: 1", trend)
        self.assertIn("surface_grasp_001", trend)
        self.assertIn("surface_grasp_002", trend)


if __name__ == "__main__":
    unittest.main()
