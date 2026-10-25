"""指标脚本（技术文档 §7）。

无外部素材时合成一帧（默认 128x96 gradient），流程：
    合成/读取 YUV -> encode_frame -> 打印 PSNR(Y)、SSIM(Y)、总码率估算(bit)、
    等效 bpp、宏块数、有效基准 Qp，并断言重建帧尺寸与原图一致。

用法示例：
    python tests/run_metrics.py
    python tests/run_metrics.py --width 192 --height 128 --qp 24 --pattern edges
    python tests/run_metrics.py --input in.yuv --width 128 --height 96
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import metrics
from h264_iframe_model import encoder
from h264_iframe_model import io_yuv
from h264_iframe_model.config import EncoderConfig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="H.264 I 帧编码器参考模型质量指标脚本"
    )
    parser.add_argument("--width", type=int, default=128, help="图像宽度（16 对齐）")
    parser.add_argument("--height", type=int, default=96, help="图像高度（16 对齐）")
    parser.add_argument("--qp", type=int, default=28, help="基准 Qp（0..51）")
    parser.add_argument("--pattern", default="gradient",
                        help="合成图案：gradient / flat / random / edges")
    parser.add_argument("--input", default=None,
                        help="输入 YUV420P 文件（提供时忽略 --pattern）")
    args = parser.parse_args(argv)

    if args.input:
        frame = io_yuv.read_yuv420p(args.input, args.width, args.height)
    else:
        frame = io_yuv.synthesize_frame(args.width, args.height, args.pattern)

    cfg = EncoderConfig(width=args.width, height=args.height, base_qp=args.qp)
    result = encoder.encode_frame(frame, cfg)

    # 断言重建帧尺寸与原图一致
    recon = result.recon_frame
    assert recon.width == frame.width and recon.height == frame.height
    assert len(recon.y) == len(frame.y) and len(recon.y[0]) == len(frame.y[0])
    assert len(recon.u) == len(frame.u) and len(recon.v) == len(frame.v)

    psnr_y = metrics.frame_psnr(frame, recon)
    ssim_y = metrics.frame_ssim(frame, recon)
    total_pixels = frame.width * frame.height
    bpp = result.total_bits / float(total_pixels) if total_pixels else 0.0

    print("图像尺寸: %dx%d" % (frame.width, frame.height))
    print("PSNR(Y): %.4f dB" % psnr_y)
    print("SSIM(Y): %.6f" % ssim_y)
    print("总码率估算: %d bit" % result.total_bits)
    print("等效 bpp: %.6f" % bpp)
    print("宏块数: %d" % len(result.macroblocks))
    print("有效基准 Qp: %d" % result.effective_base_qp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
