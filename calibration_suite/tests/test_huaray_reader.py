from __future__ import annotations

import unittest

from calibration_suite.huaray_reader import parse_barcode_chunk


class HuarayChunkParserTests(unittest.TestCase):
    def test_parses_code_and_polygon_points(self) -> None:
        result = parse_barcode_chunk(
            [
                "BarCodeNum Value:1",
                "BarCode0_CodeData Value:434857762912875",
                "BarCode0_PosPointNum Value:4",
                "BarCode0_Point0_X Value:10",
                "BarCode0_Point0_Y Value:20",
                "BarCode0_Point1_X Value:30",
                "BarCode0_Point1_Y Value:20",
                "BarCode0_Point2_X Value:30",
                "BarCode0_Point2_Y Value:40",
                "BarCode0_Point3_X Value:10",
                "BarCode0_Point3_Y Value:40",
            ]
        )

        self.assertEqual(
            result,
            [
                {
                    "code": "434857762912875",
                    "points": [(10, 20), (30, 20), (30, 40), (10, 40)],
                }
            ],
        )

    def test_keeps_colons_inside_barcode_data(self) -> None:
        result = parse_barcode_chunk(
            [
                "BarCodeNum Value:1",
                "BarCode0_CodeData Value:PKG:123",
                "BarCode0_PosPointNum Value:0",
            ]
        )

        self.assertEqual(result[0]["code"], "PKG:123")


if __name__ == "__main__":
    unittest.main()
