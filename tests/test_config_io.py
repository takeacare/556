"""config 与 io_yuv 模块单元测试（技术文档 §5.3 / §6）。

覆盖：EncoderConfig.validate 非法参数、寄存器往返映射、合成帧尺寸与取值、
YUV420P 读写往返、extract_block 越界。
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h264_iframe_model import io_yuv
from h264_iframe_model.config import EncoderConfig


class TestEncoderConfigValidate(unittest.TestCase):
    """配置校验：16 非对齐 / 非法 Qp / 非法预测模式应抛 ValueError。"""

    def test_width_not_aligned(self):
        with self.assertRaises(ValueError):
            EncoderConfig(width=100, height=48).validate()

    def test_height_not_aligned(self):
        with self.assertRaises(ValueError):
            EncoderConfig(width=64, height=50).validate()

    def test_zero_or_negative_dimension(self):
        with self.assertRaises(ValueError):
            EncoderConfig(width=0, height=48).validate()
        with self.assertRaises(ValueError):
            EncoderConfig(width=64, height=-16).validate()

    def test_illegal_qp(self):
        for qp in (-1, 52, 100):
            with self.assertRaises(ValueError):
                EncoderConfig(width=64, height=48, base_qp=qp).validate()

    def test_illegal_predict_mode(self):
        with self.assertRaises(ValueError):
            EncoderConfig(width=64, height=48, predict_mode=2).validate()

    def test_valid_config(self):
        EncoderConfig(width=64, height=48, base_qp=28).validate()


class TestRegisterRoundTrip(unittest.TestCase):
    """寄存器 <-> 配置对象往返一致（仅寄存器映射字段）。"""

    def test_round_trip(self):
        cfg = EncoderConfig(
            width=64,
            height=48,
            base_qp=30,
            aq_enable=False,
            psy_rd_enable=False,
            deblock_enable=True,
            predict_mode=1,
        )
        back = EncoderConfig.from_registers(cfg.to_registers())
        for field in ("width", "height", "base_qp", "aq_enable",
                      "psy_rd_enable", "deblock_enable", "predict_mode"):
            self.assertEqual(getattr(back, field), getattr(cfg, field), field)

    def test_from_registers_defaults(self):
        # 缺失字段取默认值（width/height 默认 1920/1088）
        cfg = EncoderConfig.from_registers({})
        self.assertEqual(cfg.width, EncoderConfig().width)
        self.assertEqual(cfg.height, EncoderConfig().height)


class TestSynthesizeFrame(unittest.TestCase):
    """合成帧各 pattern 的尺寸与取值范围。"""

    def test_all_patterns(self):
        w, h = 64, 48
        for pattern in ("gradient", "flat", "random", "edges"):
            frame = io_yuv.synthesize_frame(w, h, pattern)
            self.assertEqual(len(frame.y), h)
            self.assertEqual(len(frame.y[0]), w)
            self.assertEqual(len(frame.u), h // 2)
            self.assertEqual(len(frame.u[0]), w // 2)
            self.assertEqual(len(frame.v), h // 2)
            self.assertEqual(len(frame.v[0]), w // 2)
            for plane in (frame.y, frame.u, frame.v):
                for row in plane:
                    for v in row:
                        self.assertGreaterEqual(v, 0)
                        self.assertLessEqual(v, 255)

    def test_unknown_pattern(self):
        with self.assertRaises(ValueError):
            io_yuv.synthesize_frame(64, 48, "unknown")


class TestYuvRoundTrip(unittest.TestCase):
    """YUV420P 读写往返数据一致（临时文件，测试后删除）。"""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".yuv")
        os.close(fd)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_round_trip(self):
        frame = io_yuv.synthesize_frame(64, 48, "gradient")
        io_yuv.write_yuv420p(self.path, frame)
        back = io_yuv.read_yuv420p(self.path, 64, 48)
        self.assertEqual(back.y, frame.y)
        self.assertEqual(back.u, frame.u)
        self.assertEqual(back.v, frame.v)


class TestExtractBlock(unittest.TestCase):
    """extract_block 正常提取与越界行为。"""

    def setUp(self):
        self.plane = [[r * 10 + c for c in range(8)] for r in range(8)]

    def test_valid_extract(self):
        # plane[r][c] = r*10 + c；提取 x0=2, y0=3, size=4 -> 行 3..6、列 2..5
        block = io_yuv.extract_block(self.plane, 2, 3, 4)
        self.assertEqual(len(block), 4)
        self.assertEqual(block[0], [32, 33, 34, 35])
        self.assertEqual(block[3], [62, 63, 64, 65])

    def test_out_of_bounds(self):
        with self.assertRaises(IndexError):
            io_yuv.extract_block(self.plane, 6, 0, 4)
        with self.assertRaises(IndexError):
            io_yuv.extract_block(self.plane, -1, 0, 4)
        with self.assertRaises(IndexError):
            io_yuv.extract_block(self.plane, 0, 6, 4)


if __name__ == "__main__":
    unittest.main()
