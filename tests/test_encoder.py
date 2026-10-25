"""encoder 端到端测试（技术文档 §3/§4/§5/§7）。

覆盖：两种尺寸 x 两种图案 encode_frame 跑通、宏块数、比特数和、候选/第二级
长度、重建帧尺寸、Qp=0 高 PSNR，以及关闭 AQ / Psy-RD 时能出结果且比特不同。
说明：`edges`/`flat` 图案关闭 AQ/Psy-RD 时总比特可能与默认相同（预测/量化
结果恰好一致），此处以 `random` 图案验证开关确实改变总比特——以源码实际
行为为准。
"""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import metrics
from h264_iframe_model import encoder
from h264_iframe_model import io_yuv
from h264_iframe_model.config import EncoderConfig

_SIZES = ((64, 48), (128, 64))
_PATTERNS = ("edges", "gradient")


class TestEncodeFrameEndToEnd(unittest.TestCase):
    """两种尺寸、两种图案的端到端编码一致性。"""

    def _encode(self, width, height, pattern, qp=28, **kwargs):
        frame = io_yuv.synthesize_frame(width, height, pattern)
        cfg = EncoderConfig(width=width, height=height, base_qp=qp, **kwargs)
        return frame, encoder.encode_frame(frame, cfg)

    def test_runs_and_counts(self):
        for (w, h) in _SIZES:
            for pattern in _PATTERNS:
                frame, result = self._encode(w, h, pattern)
                num_mb = (w // 16) * (h // 16)
                self.assertEqual(len(result.macroblocks), num_mb)
                self.assertEqual(
                    sum(mb.bits for mb in result.macroblocks), result.total_bits
                )
                self.assertGreater(result.total_bits, 0)

    def test_candidate_and_stage2_lengths(self):
        for (w, h) in _SIZES:
            for pattern in _PATTERNS:
                _frame, result = self._encode(w, h, pattern)
                for mb in result.macroblocks:
                    self.assertGreaterEqual(len(mb.candidates), 1)
                    self.assertEqual(len(mb.candidates), len(mb.stage2))

    def test_recon_dimensions_match(self):
        for (w, h) in _SIZES:
            for pattern in _PATTERNS:
                frame, result = self._encode(w, h, pattern)
                recon = result.recon_frame
                self.assertEqual(recon.width, w)
                self.assertEqual(recon.height, h)
                self.assertEqual(len(recon.y), h)
                self.assertEqual(len(recon.y[0]), w)
                self.assertEqual(len(recon.u), h // 2)
                self.assertEqual(len(recon.u[0]), w // 2)
                self.assertEqual(len(recon.v), h // 2)
                self.assertEqual(len(recon.v[0]), w // 2)
                # 与输入帧尺寸一致
                self.assertEqual(len(frame.y), len(recon.y))
                self.assertEqual(len(frame.u), len(recon.u))
                self.assertEqual(len(frame.v), len(recon.v))

    def test_qp0_high_psnr(self):
        for (w, h) in _SIZES:
            for pattern in _PATTERNS:
                frame, result = self._encode(w, h, pattern, qp=0)
                psnr = metrics.frame_psnr(frame, result.recon_frame)
                self.assertGreater(psnr, 30.0)


class TestToggles(unittest.TestCase):
    """关闭 AQ / Psy-RD 能正常出结果且总比特与默认配置不同。"""

    def test_aq_and_psy_off_change_bits(self):
        frame = io_yuv.synthesize_frame(128, 64, "random")
        base = encoder.encode_frame(
            frame, EncoderConfig(width=128, height=64, base_qp=28)
        )
        noaq = encoder.encode_frame(
            frame, EncoderConfig(width=128, height=64, base_qp=28, aq_enable=False)
        )
        nopsy = encoder.encode_frame(
            frame, EncoderConfig(width=128, height=64, base_qp=28, psy_rd_enable=False)
        )
        # 均能正常出结果
        for res in (noaq, nopsy):
            self.assertEqual(len(res.macroblocks), (128 // 16) * (64 // 16))
            self.assertGreater(res.total_bits, 0)
        # 与默认配置总比特不同
        self.assertNotEqual(base.total_bits, noaq.total_bits)
        self.assertNotEqual(base.total_bits, nopsy.total_bits)


if __name__ == "__main__":
    unittest.main()
