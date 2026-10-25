"""intra_pred 模块单元测试（技术文档 §2.3 / §4）。

覆盖：16x16 四模式与 4x4 九模式的输出尺寸/取值，select_candidate_modes
候选数量、SATD 升序与候选预测块尺寸。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h264_iframe_model import intra_pred
from h264_iframe_model import io_yuv


class TestPredict16x16(unittest.TestCase):
    """16x16 四种模式均返回 16x16 合法像素。"""

    def setUp(self):
        self.plane = io_yuv.synthesize_frame(64, 64, "gradient").y
        self.ref = intra_pred.gather_mb_reference(self.plane, 1, 1)

    def test_four_modes(self):
        for mode in (intra_pred.I16_VERTICAL, intra_pred.I16_HORIZONTAL,
                     intra_pred.I16_DC, intra_pred.I16_PLANE):
            pred = intra_pred.predict_16x16(
                mode, self.ref["top"], self.ref["left"], self.ref["top_left"]
            )
            self.assertEqual(len(pred), 16)
            for row in pred:
                self.assertEqual(len(row), 16)
                for v in row:
                    self.assertIsInstance(v, int)
                    self.assertGreaterEqual(v, 0)
                    self.assertLessEqual(v, 255)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            intra_pred.predict_16x16(99, self.ref["top"], self.ref["left"],
                                     self.ref["top_left"])


class TestPredict4x4(unittest.TestCase):
    """4x4 九种模式均返回 4x4 合法像素。"""

    def setUp(self):
        self.plane = io_yuv.synthesize_frame(64, 64, "gradient").y
        self.ref = intra_pred.gather_4x4_reference(self.plane, 8, 8)

    def test_nine_modes(self):
        for mode in range(9):
            pred = intra_pred.predict_4x4(
                mode, self.ref["top"], self.ref["left"], self.ref["top_left"]
            )
            self.assertEqual(len(pred), 4)
            for row in pred:
                self.assertEqual(len(row), 4)
                for v in row:
                    self.assertGreaterEqual(v, 0)
                    self.assertLessEqual(v, 255)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            intra_pred.predict_4x4(9, self.ref["top"], self.ref["left"],
                                   self.ref["top_left"])


class TestSelectCandidateModes(unittest.TestCase):
    """候选粗筛：长度等于 num_candidates、按 SATD 升序、预测块为 16x16。"""

    def setUp(self):
        self.plane = io_yuv.synthesize_frame(64, 64, "edges").y

    def test_length_and_order(self):
        for num in (1, 2, 3, 5):
            cands = intra_pred.select_candidate_modes(self.plane, 1, 1, num)
            self.assertEqual(len(cands), num)
            self.assertGreaterEqual(cands[0].satd, 0)
            # 按 SATD 升序
            for i in range(len(cands) - 1):
                self.assertLessEqual(cands[i].satd, cands[i + 1].satd)

    def test_candidate_pred_is_16x16(self):
        cands = intra_pred.select_candidate_modes(self.plane, 0, 0, 2)
        for cand in cands:
            self.assertEqual(len(cand.pred), 16)
            for row in cand.pred:
                self.assertEqual(len(row), 16)
            self.assertIn(cand.mode_type, ("I16", "I4"))


if __name__ == "__main__":
    unittest.main()
