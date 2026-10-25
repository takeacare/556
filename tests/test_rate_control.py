"""rate_control 模块单元测试（技术文档 3.5.2）。

覆盖：frame_complexity 随 satd_sum 单调不减、adjust_qp 关于 complexity 单调
不减且在 0..51、crf_base_qp 关闭时返回 base_qp。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h264_iframe_model import rate_control


class TestFrameComplexity(unittest.TestCase):
    """复杂度随 satd_sum 单调不减。"""

    def test_monotonic(self):
        num_mbs = 12
        prev = -1.0
        for satd_sum in range(0, 200000, 5000):
            c = rate_control.frame_complexity(satd_sum, num_mbs)
            self.assertGreaterEqual(c, prev)
            prev = c

    def test_zero_mbs(self):
        self.assertEqual(rate_control.frame_complexity(1000, 0), 0.0)


class TestAdjustQp(unittest.TestCase):
    """adjust_qp 关于 complexity 单调不减且在 0..51。"""

    def test_monotonic_and_range(self):
        base = 28
        prev = -1
        for complexity in (0, 50, 100, 500, 2000, 8000, 20000, 100000):
            qp = rate_control.adjust_qp(base, complexity)
            self.assertGreaterEqual(qp, 0)
            self.assertLessEqual(qp, 51)
            self.assertGreaterEqual(qp, prev)
            prev = qp

    def test_low_complexity_no_offset(self):
        self.assertEqual(rate_control.adjust_qp(28, 0.0), 28)


class TestCrfBaseQp(unittest.TestCase):
    """crf_base_qp 关闭时返回 base_qp。"""

    def test_disabled(self):
        self.assertEqual(rate_control.crf_base_qp(30, 99999, 12, enable=False), 30)

    def test_enabled_matches_adjust(self):
        got = rate_control.crf_base_qp(28, 120000, 12, enable=True)
        expect = rate_control.adjust_qp(
            28, rate_control.frame_complexity(120000, 12)
        )
        self.assertEqual(got, expect)


if __name__ == "__main__":
    unittest.main()
