"""deblock 模块单元测试（技术文档 3.3）。

覆盖：boundary_strength 取值范围与色度档位、平坦/强边缘档位、关闭滤波不
改变像素、开启滤波对可平滑跳变减小边界梯度。
说明：本模型采用内容自适应策略——差异极大的边界被视为真实边缘（BS=0）而
不滤波；因此“明显跳变”需用可平滑的较小台阶来验证梯度减小，同时另行断言
大台阶被保留（以源码实际行为为准）。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h264_iframe_model import deblock
from h264_iframe_model import io_yuv


def _two_block_plane(diff, left=100, width=32, height=16):
    """构造左右两块（各 16 列）亮度相差 diff 的平面。"""
    right = left + diff
    return [[left if c < 16 else right for c in range(width)]
            for _ in range(height)]


def _zero_chroma(width=32, height=16):
    return [[0] * (width // 2) for _ in range(height // 2)]


class TestBoundaryStrength(unittest.TestCase):
    """BS 在 0~4，色度在 {0,2}。"""

    def test_range(self):
        cases = [
            ([100, 100, 100, 100], [100, 100, 100, 100]),
            ([0, 0, 0, 0], [255, 255, 255, 255]),
            ([10, 12, 20, 40], [60, 30, 10, 5]),
        ]
        for p, q in cases:
            bs = deblock.boundary_strength(p, q, is_chroma=False)
            self.assertGreaterEqual(bs, 0)
            self.assertLessEqual(bs, 4)

    def test_chroma_levels(self):
        for p, q in (([10, 10, 10, 10], [10, 10, 10, 10]),
                     ([0, 0, 0, 0], [255, 255, 255, 255])):
            bs = deblock.boundary_strength(p, q, is_chroma=True)
            self.assertIn(bs, (0, 2))

    def test_flat_high_strength(self):
        # 全平坦边界 -> 高档（强滤波）
        bs = deblock.boundary_strength([100] * 4, [100] * 4, is_chroma=False)
        self.assertGreaterEqual(bs, 3)

    def test_large_diff_low_strength(self):
        # 差异极大的边界 -> 低档（弱滤波/不滤波）
        bs = deblock.boundary_strength([0] * 4, [255] * 4, is_chroma=False)
        self.assertLessEqual(bs, 1)


class TestDeblockFrame(unittest.TestCase):
    """整帧去块：关闭不改变像素；开启对可平滑台阶减小边界梯度。"""

    def test_disabled_no_change(self):
        plane = [[(r * 13 + c * 7) % 256 for c in range(32)] for r in range(16)]
        frame = io_yuv.YUV420PFrame(32, 16, plane, _zero_chroma(), _zero_chroma())
        before = io_yuv.clone_frame(frame)
        deblock.deblock_frame(frame, 40, enable=False)
        self.assertEqual(frame.y, before.y)
        self.assertEqual(frame.u, before.u)
        self.assertEqual(frame.v, before.v)

    def test_enabled_reduces_small_step(self):
        # 左右相差 4 的平坦两块：边界梯度应减小
        plane = _two_block_plane(4)
        frame = io_yuv.YUV420PFrame(32, 16, plane, _zero_chroma(), _zero_chroma())
        before = abs(frame.y[0][15] - frame.y[0][16])
        deblock.deblock_frame(frame, 51, enable=True)
        after = abs(frame.y[0][15] - frame.y[0][16])
        self.assertLess(after, before)

    def test_enabled_preserves_large_step(self):
        # 相差 40 的强边缘：内容自适应不滤波，边界梯度保持
        plane = _two_block_plane(40)
        frame = io_yuv.YUV420PFrame(32, 16, plane, _zero_chroma(), _zero_chroma())
        before = abs(frame.y[0][15] - frame.y[0][16])
        deblock.deblock_frame(frame, 51, enable=True)
        after = abs(frame.y[0][15] - frame.y[0][16])
        self.assertEqual(after, before)

    def test_enabled_does_not_increase_gradient(self):
        plane = _two_block_plane(6)
        frame = io_yuv.YUV420PFrame(32, 16, plane, _zero_chroma(), _zero_chroma())
        before = abs(frame.y[0][15] - frame.y[0][16])
        deblock.deblock_frame(frame, 51, enable=True)
        after = abs(frame.y[0][15] - frame.y[0][16])
        self.assertLessEqual(after, before)


if __name__ == "__main__":
    unittest.main()
