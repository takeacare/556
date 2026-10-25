"""psy_rd 模块单元测试（技术文档 3.5.1）。

覆盖：HVS 权重表为 4x4 且低频权重不小于高频、加权失真相对于不加权有差异、
RD 代价在开启/关闭 Psy-RD（不同强度）下结果不同、lambda 随 Qp 单调不减。
说明：rd_cost 在 psy_rd_strength=1.0 时开启与关闭等价（lambda 乘以 1.0），
故此处用 strength != 1.0 验证开启/关闭的差异——以源码实际行为为准。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h264_iframe_model import psy_rd


class TestHvsWeightTable(unittest.TestCase):
    """HVS 权重表 4x4，低频 (0,0) 不小于高频 (3,3)。"""

    def test_shape_and_order(self):
        tab = psy_rd.HVS_WEIGHT_TABLE_4X4
        self.assertEqual(len(tab), 4)
        for row in tab:
            self.assertEqual(len(row), 4)
        self.assertGreaterEqual(tab[0][0], tab[3][3])


class TestHvsWeightedDistortion(unittest.TestCase):
    """非零输入时加权与不加权结果不同。"""

    def test_enable_changes_result(self):
        orig = [[0] * 4 for _ in range(4)]
        recon = [[0] * 4 for _ in range(4)]
        orig[1][1] = 4  # (1,1) 权重 11/16 != 1
        on = psy_rd.hvs_weighted_distortion(orig, recon, True)
        off = psy_rd.hvs_weighted_distortion(orig, recon, False)
        self.assertNotEqual(on, off)

    def test_identical_zero(self):
        z = [[0] * 4 for _ in range(4)]
        self.assertEqual(psy_rd.hvs_weighted_distortion(z, z, True), 0.0)


class TestRdCost(unittest.TestCase):
    """rd_cost 开启/关闭 Psy-RD（不同强度）结果不同。"""

    def test_enable_changes_result(self):
        on = psy_rd.rd_cost(10.0, 5, 28, True, 2.0)
        off = psy_rd.rd_cost(10.0, 5, 28, False)
        self.assertNotEqual(on, off)


class TestLambda(unittest.TestCase):
    """lambda_for_qp 随 Qp 单调不减。"""

    def test_monotonic(self):
        lams = [psy_rd.lambda_for_qp(qp) for qp in range(0, 52)]
        for i in range(len(lams) - 1):
            self.assertLessEqual(lams[i], lams[i + 1])


if __name__ == "__main__":
    unittest.main()
