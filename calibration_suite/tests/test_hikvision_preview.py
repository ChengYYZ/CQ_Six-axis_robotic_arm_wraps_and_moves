from __future__ import annotations

import io
import unittest

from PIL import Image

from calibration_suite.hikvision_preview import HikvisionPreviewPublisher


class HikvisionPreviewTests(unittest.TestCase):
    def test_decodes_jpeg_surrounded_by_transport_bytes(self) -> None:
        source = Image.new("RGB", (8, 6), (30, 60, 90))
        encoded = io.BytesIO()
        source.save(encoded, format="JPEG")

        decoded = HikvisionPreviewPublisher._decode_image(
            b"transport-prefix" + encoded.getvalue() + b"transport-suffix"
        )

        self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(decoded.size, (8, 6))


if __name__ == "__main__":
    unittest.main()
