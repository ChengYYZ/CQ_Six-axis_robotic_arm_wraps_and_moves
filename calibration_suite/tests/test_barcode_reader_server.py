from __future__ import annotations

import socket
import sys
import unittest
from pathlib import Path


CALIBRATION_SUITE = Path(__file__).resolve().parents[1]
if str(CALIBRATION_SUITE) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_SUITE))

from surface_cluster_grasp import BarcodeReaderServer


class BarcodeReaderServerTests(unittest.TestCase):
    def setUp(self) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        self.port = int(probe.getsockname()[1])
        probe.close()
        self.server = BarcodeReaderServer("127.0.0.1", self.port)
        self.assertTrue(self.server.start())

    def tearDown(self) -> None:
        self.server.stop()

    def _send(self, payload: bytes) -> None:
        with socket.create_connection(("127.0.0.1", self.port), timeout=1.0) as client:
            client.sendall(payload)

    def test_receives_messages_before_d_point_wait_begins(self) -> None:
        marker = self.server.current_sequence()
        self._send(b"PKG-123\r\n")

        message = self.server.wait_for_message(marker, 1.0)

        self.assertIsNotNone(message)
        assert message is not None
        self.assertEqual(message.text, "PKG-123")

    def test_marker_prevents_reusing_previous_package_result(self) -> None:
        self._send(b"OLD\n")
        old = self.server.wait_for_message(0, 1.0)
        self.assertIsNotNone(old)
        marker = self.server.current_sequence()
        self._send("新条码".encode("gb18030") + b"\x00")

        new = self.server.wait_for_message(marker, 1.0)

        self.assertIsNotNone(new)
        assert new is not None
        self.assertEqual(new.text, "新条码")
        self.assertGreater(new.sequence, marker)


if __name__ == "__main__":
    unittest.main()
