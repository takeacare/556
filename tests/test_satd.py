"""satd 模块单元测试（技术文档 3.1.3）。

覆盖：hadamard4x4 纯整数、全零块/常量块 SATD、satd16x16 等于子块累加。
说明：本参考模型的 SATD 保留 DC 分量，故非常量块的常量值 v 的 SATD 为
8*|v|（仅在 4x4 哈德曼变换的 DC 位置有能量），而非 0——以源码实际行为为准。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h264_iframe_model import satd


class TestHadamard(unittest.TestCase):
    """哈德曼变换仅使用加减法，输出为整数 4x4。"""

    def test_shape_and_integer(self):
        block = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        out = satd.hadamard4x4(block)
        self.assertEqual(len(out), 4)
        for row in out:
            self.assertEqual(len(row), 4)
            for v in row:
                self.assertIsInstance(v, int)

    def test_zero_block_zero_output(self):
        out = satd.hadamard4x4([[0] * 4 for _ in range(4)])
        self.assertEqual(out, [[0] * 4 for _ in range(4)])


class TestSatd4x4(unittest.TestCase):
    """4x4 SATD：全零为 0，常量块为 8*|v|。"""

    def test_all_zero(self):
        self.assertEqual(satd.satd4x4([[0] * 4 for _ in range(4)]), 0)

    def test_constant_block(self):
        # 常量块的哈德曼变换仅在 DC 位置有能量：|16*v| // 2 = 8*|v|
        for v in (1, 5, -7):
            block = [[v] * 4 for _ in range(4)]
            self.assertEqual(satd.satd4x4(block), 8 * abs(v))

    def test_non_negative(self):
        res = [[(r * 5 + c * 3) % 7 - 3 for c in range(4)] for r in range(4)]
        self.assertGreaterEqual(satd.satd4x4(res), 0)


class TestSatd16x16(unittest.TestCase):
    """satd16x16 等于 16 个 4x4 子块 SATD 累加之和。"""

    def test_equals_subblock_sum(self):
        res = [[(r * 7 + c * 3) % 9 - 4 for c in range(16)] for r in range(16)]
        manual = 0
        for by in range(0, 16, 4):
            for bx in range(0, 16, 4):
                sub = [row[bx:bx + 4] for row in res[by:by + 4]]
                manual += satd.satd4x4(sub)
        self.assertEqual(satd.satd16x16(res), manual)

    def test_satd8_subblock_sum(self):
        res = [[(r * 3 + c * 5) % 11 - 5 for c in range(8)] for r in range(8)]
        manual = 0
        for by in range(0, 8, 4):
            for bx in range(0, 8, 4):
                sub = [row[bx:bx + 4] for row in res[by:by + 4]]
                manual += satd.satd4x4(sub)
        self.assertEqual(satd.satd8x8(res), manual)


if __name__ == "__main__":
    unittest.main()
