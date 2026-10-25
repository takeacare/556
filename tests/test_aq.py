"""aq 模块单元测试（技术文档 3.2.3）。

覆盖：AQ_OFFSET_TABLE 长度与取值范围、关闭 AQ 时返回 base_qp、实际 Qp 范围、
Qp 偏移随 SATD 单调不减。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h264_iframe_model import aq


class TestAqTable(unittest.TestCase):
    """偏移表长度 256，取值均在 [-6, 6]。"""

    def test_length_and_range(self):
        self.assertEqual(len(aq.AQ_OFFSET_TABLE), 256)
        for v in aq.AQ_OFFSET_TABLE:
            self.assertGreaterEqual(v, -6)
            self.assertLessEqual(v, 6)


class TestActualQp(unittest.TestCase):
    """actual_qp 关 AQ 返回 base_qp，开启时结果在 0..51。"""

    def test_disabled_returns_base(self):
        for base in (0, 20, 51):
            self.assertEqual(aq.actual_qp(base, 12345, aq_enable=False), base)

    def test_range(self):
        for base in (0, 25, 51):
            for satd in (0, 500, 5000, 20000, 10 ** 7):
                qp = aq.actual_qp(base, satd)
                self.assertGreaterEqual(qp, 0)
                self.assertLessEqual(qp, 51)

    def test_qp_offset_monotonic(self):
        # SATD 越大，Qp 偏移不小于 SATD 越小者（非严格单调，但单调不减）
        satds = list(range(0, 20000, 137))
        offsets = [aq.qp_offset(s) for s in satds]
        for i in range(len(offsets) - 1):
            self.assertLessEqual(offsets[i], offsets[i + 1])

    def test_larger_satd_not_smaller_offset(self):
        self.assertLessEqual(aq.qp_offset(100), aq.qp_offset(15000))


if __name__ == "__main__":
    unittest.main()
