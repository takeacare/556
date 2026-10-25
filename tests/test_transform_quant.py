"""transform / quant 模块单元测试（技术文档 3.2.2 / 3.2.3）。

覆盖：正向变换整数 4x4、量化/反量化在 Qp 0 与 51 下可用、Qp=0 近无损
往返、quant_step 随 Qp 单调不减。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h264_iframe_model import quant
from h264_iframe_model import transform


def _residual():
    """构造一个幅值适中的 4x4 残差块。"""
    return [
        [10, -3, 4, 0],
        [2, 7, -1, 5],
        [-4, 1, 6, -2],
        [3, 0, -5, 8],
    ]


class TestForwardTransform(unittest.TestCase):
    """正向变换输出为整数 4x4。"""

    def test_shape_and_integer(self):
        out = transform.forward_transform_4x4(_residual())
        self.assertEqual(len(out), 4)
        for row in out:
            self.assertEqual(len(row), 4)
            for v in row:
                self.assertIsInstance(v, int)

    def test_zero_residual(self):
        out = transform.forward_transform_4x4([[0] * 4 for _ in range(4)])
        self.assertEqual(out, [[0] * 4 for _ in range(4)])


class TestQuantDequant(unittest.TestCase):
    """量化/反量化在 Qp 0 与 51 下均可用，输出 4x4。"""

    def test_supported_qp(self):
        fwd = transform.forward_transform_4x4(_residual())
        for qp in (0, 51):
            levels = quant.quantize_4x4(fwd, qp)
            self.assertEqual(len(levels), 4)
            for row in levels:
                self.assertEqual(len(row), 4)
            dq = quant.dequantize_4x4(levels, qp)
            self.assertEqual(len(dq), 4)
            for row in dq:
                self.assertEqual(len(row), 4)

    def test_qp0_near_lossless(self):
        # Qp=0 时量化误差应极小，重建残差与原始残差误差 <= 2
        res = _residual()
        fwd = transform.forward_transform_4x4(res)
        levels = quant.quantize_4x4(fwd, 0)
        dq = quant.dequantize_4x4(levels, 0)
        inv = transform.inverse_transform_4x4(dq)
        max_err = max(abs(inv[i][j] - res[i][j]) for i in range(4) for j in range(4))
        self.assertLessEqual(max_err, 2)


class TestQuantStep(unittest.TestCase):
    """quant_step 随 Qp 单调不减。"""

    def test_monotonic(self):
        steps = [quant.quant_step(qp) for qp in range(0, 52)]
        for i in range(len(steps) - 1):
            self.assertLessEqual(steps[i], steps[i + 1])

    def test_clip(self):
        # Qp 超范围时被裁剪到 0..51
        self.assertEqual(quant.quant_step(-5), quant.quant_step(0))
        self.assertEqual(quant.quant_step(100), quant.quant_step(51))


if __name__ == "__main__":
    unittest.main()
