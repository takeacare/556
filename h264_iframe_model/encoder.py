"""H.264 I 帧编码器顶层入口（技术文档 §3/§4/§5）。

串联三段流程：
    1. 帧级复杂度预分析（CRF）：遍历所有宏块做第一级预测+SATD，累加得
       satd_sum，据此由 rate_control.crf_base_qp 调整基准 Qp；
    2. 宏块三级流水（pipeline.encode_macroblock）：逐宏块编码并收集结果；
    3. 帧级去块滤波（deblock.deblock_frame）：对重建帧去块（可开关）。

输入 YUV420P 帧 + 配置，输出重建帧、每宏块模式与总码率估算比特数。
"""

import argparse
from dataclasses import dataclass

from . import deblock
from . import intra_pred
from . import io_yuv
from . import rate_control
from .config import EncoderConfig
from .pipeline import encode_macroblock


@dataclass
class EncodeResult:
    """一帧的编码结果。"""

    recon_frame: object      # YUV420PFrame 重建帧
    macroblocks: list        # MacroblockResult 列表
    total_bits: int          # 全帧码率估算比特数
    base_qp: int             # 配置中的基准 Qp
    effective_base_qp: int   # CRF 调整后的基准 Qp
    satd_sum: int            # 全帧 SATD 累加（复杂度度量）


def encode_frame(frame, cfg) -> EncodeResult:
    """编码一帧：CRF 预分析 → 宏块三级流水 → 帧级去块。"""
    cfg.validate()

    recon = io_yuv.clone_frame(frame)
    indices = io_yuv.iter_macroblock_indices(cfg.width, cfg.height)
    num_mbs = len(indices)

    # ---- 1. 帧级复杂度预分析（复用第一级预测 + SATD）----
    satd_sum = 0
    for (mb_x, mb_y) in indices:
        cands = intra_pred.select_candidate_modes(frame.y, mb_x, mb_y, 2)
        satd_sum += cands[0].satd

    effective_base_qp = rate_control.crf_base_qp(
        cfg.base_qp, satd_sum, num_mbs, cfg.crf_enable
    )

    # ---- 2. 宏块三级流水（按光栅顺序）----
    macroblocks = []
    total_bits = 0
    for (mb_x, mb_y) in indices:
        mb = encode_macroblock(frame, recon, mb_x, mb_y, cfg, effective_base_qp, 2)
        macroblocks.append(mb)
        total_bits += mb.bits

    # ---- 3. 帧级去块滤波 ----
    if cfg.deblock_enable:
        deblock.deblock_frame(recon, effective_base_qp, True)

    return EncodeResult(
        recon_frame=recon,
        macroblocks=macroblocks,
        total_bits=total_bits,
        base_qp=cfg.base_qp,
        effective_base_qp=effective_base_qp,
        satd_sum=satd_sum,
    )


def main(argv=None) -> int:
    """简易命令行入口。

    无 --input 时用 io_yuv.synthesize_frame 合成一帧（edges 图案）便于自测；
    打印总码率估算比特数与等效 bpp。
    """
    parser = argparse.ArgumentParser(
        description="H.264 I 帧编码器 Python 参考模型"
    )
    parser.add_argument("--input", default=None, help="输入 YUV420P 文件（可选）")
    parser.add_argument("--width", type=int, default=64, help="图像宽度（16 对齐）")
    parser.add_argument("--height", type=int, default=48, help="图像高度（16 对齐）")
    parser.add_argument("--qp", type=int, default=28, help="基准 Qp（0..51）")
    parser.add_argument("--output", default=None, help="输出重建 YUV420P 文件（可选）")
    parser.add_argument("--no-aq", action="store_true", help="关闭自适应量化 AQ")
    parser.add_argument("--no-psy", action="store_true", help="关闭 Psy-RD")
    parser.add_argument("--no-deblock", action="store_true", help="关闭去块滤波")
    parser.add_argument("--no-crf", action="store_true", help="关闭 CRF 码率控制")
    args = parser.parse_args(argv)

    cfg = EncoderConfig(
        width=args.width,
        height=args.height,
        base_qp=args.qp,
        aq_enable=not args.no_aq,
        psy_rd_enable=not args.no_psy,
        deblock_enable=not args.no_deblock,
        crf_enable=not args.no_crf,
    )

    if args.input:
        frame = io_yuv.read_yuv420p(args.input, args.width, args.height)
    else:
        frame = io_yuv.synthesize_frame(args.width, args.height, "edges")

    result = encode_frame(frame, cfg)

    if args.output:
        io_yuv.write_yuv420p(args.output, result.recon_frame)

    total_pixels = args.width * args.height
    bpp = result.total_bits / float(total_pixels) if total_pixels else 0.0

    print("宏块数: %d" % len(result.macroblocks))
    print("总码率估算比特数: %d" % result.total_bits)
    print("等效 bpp: %.4f" % bpp)
    print("基准 Qp: %d, CRF 调整后 Qp: %d" % (result.base_qp, result.effective_base_qp))
    print("帧 SATD 累加: %d" % result.satd_sum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
