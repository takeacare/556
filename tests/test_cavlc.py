"""cavlc 模块单元测试（技术文档 3.4）。

覆盖：zig-zag 扫描往返一致、全零块 0 比特、非零块比特 > 0、非零系数更多时
比特数不小于更少时。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h264_iframe_model import cavlc


def _block(coords_vals):
    """由 {(r, c): value} 构造 4x4 块。"""
    blk = [[0] * 4 for _ in range(4)]
    for (r, c), v in coords_vals.items():
        blk[r][c] = v
    return blk


class TestZigzag(unittest.TestCase):
    """zigzag_scan / inverse_zigzag 往返一致。"""

    def test_round_trip(self):
        blk = [[(r * 4 + c + 1) for c in range(4)] for r in range(4)]
        seq = cavlc.zigzag_scan(blk)
        self.assertEqual(len(seq), 16)
        self.assertEqual(cavlc.inverse_zigzag(seq), blk)

    def test_scan_order(self):
        blk = [[0] * 4 for _ in range(4)]
        blk[0][0] = 1
        blk[0][1] = 2
        blk[1][0] = 3
        seq = cavlc.zigzag_scan(blk)
        self.assertEqual(seq[:3], [1, 2, 3])


class TestBlockBits(unittest.TestCase):
    """全零块 0 比特；非零块比特 > 0；非零系数越多比特不减。"""

    def test_zero_block(self):
        self.assertEqual(cavlc.block_bits([[0] * 4 for _ in range(4)]), 0)

    def test_nonzero_block(self):
        blk = _block({(0, 0): 5})
        self.assertGreater(cavlc.block_bits(blk), 0)

    def test_more_coeffs_not_less_bits(self):
        b1 = _block({(0, 0): 1})
        b2 = _block({(0, 0): 1, (0, 1): 1})
        b3 = _block({(0, 0): 1, (0, 1): 1, (0, 2): 1, (0, 3): 1,
                     (1, 0): 1, (1, 1): 1})
        bits1 = cavlc.block_bits(b1)
        bits2 = cavlc.block_bits(b2)
        bits3 = cavlc.block_bits(b3)
        self.assertLessEqual(bits1, bits2)
        self.assertLessEqual(bits2, bits3)

    def test_mb_bits_sum(self):
        blocks = [_block({(0, 0): 3}), [[0] * 4 for _ in range(4)]]
        self.assertEqual(cavlc.mb_bits(blocks), cavlc.block_bits(blocks[0]))


if __name__ == "__main__":
    unittest.main()
