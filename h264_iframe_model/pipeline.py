"""三级宏块流水线（技术文档 §2.3 阶段划分，§4/§5 编码流程）。

实现单个宏块（亮度 16x16 + 色度 8x8）的“预测级 → 变换量化级 → 决策熵编码级”
三级流水，并暴露各阶段中间产物（第一级 SATD 候选、第二级候选变换量化重建结果）
以便验证。

参考模型简化说明（重要）：
    1. 帧内预测的参考像素取自 **源平面**（沿用 intra_pred 现有行为），重建帧
       仅作为输出，不参与后续宏块的参考闭环（本模型不做重建闭环参考）；
    2. **色度采用 DC 预测简化**：统一使用 I4_DC（=2）模式，不做色度模式遍历；
    3. 熵编码仅按 CAVLC 估算比特数，不写出实际 NALU 码流。
"""

from dataclasses import dataclass, field

from . import aq
from . import cavlc
from . import intra_pred
from . import psy_rd
from . import quant
from . import transform

_MB_SIZE = 16            # 亮度宏块边长
_CHROMA_MB_SIZE = 8      # 色度宏块边长
_SUB_SIZE = 4            # 变换/量化子块边长
_MODE_OVERHEAD_BITS = 4  # 每宏块固定模式开销（如 mb_type / 预测模式语法比特）
_CHROMA_PRED_MODE = intra_pred.I4_DC  # 色度简化：统一 DC 预测（I4_DC=2）


@dataclass
class CandidateResult:
    """第二级产物：某个候选模式的变换量化重建结果。"""

    mode_type: str       # "I16" 或 "I4"
    mode: int            # I16 模式号；I4 整宏块候选为 -1
    sub_modes: list      # I4 候选的 4x4 子模式矩阵（I16 为 None）
    satd: int            # 第一级 SATD 代价
    quant_blocks: list   # 16 个 4x4 量化系数块
    recon_block: list    # 16x16 重建像素
    distortion: float    # Psy-RD 加权失真（变换域）
    rate_bits: int       # 该候选的亮度码率估算（含固定模式开销）
    rd_cost: float       # RD 代价


@dataclass
class MacroblockResult:
    """单个宏块的完整编码结果。"""

    mb_x: int
    mb_y: int
    mode_type: str       # 最优模式类型 "I16"/"I4"
    mode: int
    sub_modes: list
    qp: int              # AQ 后实际 Qp
    bits: int            # 亮度 + 色度码率估算
    pred: list           # 最优预测 16x16
    recon: list          # 最优重建 16x16
    candidates: list     # 第一级 SATD 候选（ModeCandidate 列表）
    stage2: list         # 第二级候选结果（CandidateResult 列表）


def _clip(v: int) -> int:
    """像素裁剪到 0..255。"""
    if v < 0:
        return 0
    if v > 255:
        return 255
    return v


def _transform_quant_roundtrip(residual_sub: list, qp: int):
    """对单个 4x4 残差做 正向变换 → 量化 → 反量化 → 反向变换。

    返回 (正向变换系数, 反量化系数, 反变换残差, 量化 levels)。
    正向系数作为 Psy-RD 的 orig_coeffs，反量化系数作为 recon_coeffs。
    """
    fwd = transform.forward_transform_4x4(residual_sub)
    levels = quant.quantize_4x4(fwd, qp)
    dq = quant.dequantize_4x4(levels, qp)
    inv = transform.inverse_transform_4x4(dq)
    return fwd, dq, inv, levels


def _process_luma_candidate(cand, source_frame, mb_x: int, mb_y: int, qp: int):
    """第二级：对单个亮度候选做 16 个子块的变换量化重建。

    返回 (quant_blocks, fwd_blocks, dq_blocks, recon_block)：
        quant_blocks 为 16 个 4x4 量化系数块；fwd_blocks/dq_blocks 分别为 16 个
        正向变换系数与反量化系数（供失真计算）；recon_block 为 16x16 重建像素。
    """
    px = mb_x * _MB_SIZE
    py = mb_y * _MB_SIZE
    pred = cand.pred

    quant_blocks = []
    fwd_blocks = []
    dq_blocks = []
    recon_block = [[0] * _MB_SIZE for _ in range(_MB_SIZE)]

    for sr in range(4):
        for sc in range(4):
            x0 = px + sc * _SUB_SIZE
            y0 = py + sr * _SUB_SIZE
            sub_src = [
                [source_frame.y[y0 + r][x0 + c] for c in range(_SUB_SIZE)]
                for r in range(_SUB_SIZE)
            ]
            sub_pred = [
                [pred[sr * _SUB_SIZE + r][sc * _SUB_SIZE + c] for c in range(_SUB_SIZE)]
                for r in range(_SUB_SIZE)
            ]
            residual = [
                [sub_src[r][c] - sub_pred[r][c] for c in range(_SUB_SIZE)]
                for r in range(_SUB_SIZE)
            ]
            fwd, dq, inv, levels = _transform_quant_roundtrip(residual, qp)
            quant_blocks.append(levels)
            fwd_blocks.append(fwd)
            dq_blocks.append(dq)
            for r in range(_SUB_SIZE):
                for c in range(_SUB_SIZE):
                    recon_block[sr * _SUB_SIZE + r][sc * _SUB_SIZE + c] = _clip(
                        sub_pred[r][c] + inv[r][c]
                    )

    return quant_blocks, fwd_blocks, dq_blocks, recon_block


def _encode_chroma(source_frame, recon_frame, mb_x: int, mb_y: int, qp: int) -> int:
    """色度 U/V 平面编码（DC 预测简化）。

    宏块覆盖的色度区域位于 (mb_x*8, mb_y*8)、大小 8x8，拆成 4 个 4x4 子块。
    对每个子块：gather_4x4_reference 取参考像素 → predict_4x4(DC) → 残差 →
    正向变换 → 量化 → 反量化 → 反向变换 → 与预测相加写回 recon 的 U/V 平面，
    全程使用同一实际 Qp。返回色度码率估算比特数。
    """
    total_bits = 0
    cx = mb_x * _CHROMA_MB_SIZE
    cy = mb_y * _CHROMA_MB_SIZE

    for src_plane, rec_plane in (
        (source_frame.u, recon_frame.u),
        (source_frame.v, recon_frame.v),
    ):
        for sr in range(2):
            for sc in range(2):
                x0 = cx + sc * _SUB_SIZE
                y0 = cy + sr * _SUB_SIZE
                ref = intra_pred.gather_4x4_reference(src_plane, x0, y0)
                pred = intra_pred.predict_4x4(
                    _CHROMA_PRED_MODE, ref["top"], ref["left"], ref["top_left"]
                )
                sub_src = [
                    [src_plane[y0 + r][x0 + c] for c in range(_SUB_SIZE)]
                    for r in range(_SUB_SIZE)
                ]
                residual = [
                    [sub_src[r][c] - pred[r][c] for c in range(_SUB_SIZE)]
                    for r in range(_SUB_SIZE)
                ]
                _fwd, _dq, inv, levels = _transform_quant_roundtrip(residual, qp)
                for r in range(_SUB_SIZE):
                    for c in range(_SUB_SIZE):
                        rec_plane[y0 + r][x0 + c] = _clip(pred[r][c] + inv[r][c])
                total_bits += cavlc.block_bits(levels)

    return total_bits


def encode_macroblock(source_frame, recon_frame, mb_x: int, mb_y: int, cfg,
                      eff_base_qp: int, num_candidates: int = 2) -> MacroblockResult:
    """三级流水编码单个宏块，返回 MacroblockResult。

    第一级（预测级）：select_candidate_modes 得到按 SATD 升序的候选，取最小
    SATD 供 AQ 使用，得到该宏块实际 Qp（同一宏块所有候选共用同一 Qp）。
    第二级（变换量化级）：对每个候选做 16x16 残差的逐 4x4 子块 正向变换→量化→
    反量化→反向变换，与预测相加得到重建块，记录量化块与重建块。
    第三级（决策熵编码级）：以 Psy-RD 加权失真 + CAVLC 码率构成 RD 代价，取
    rd_cost 最小的候选作为最优，写回 recon_frame.y，并附加色度重建与码率。

    参考模型简化：预测参考取自源平面、色度用 DC 预测（见模块 docstring）。
    """
    px = mb_x * _MB_SIZE
    py = mb_y * _MB_SIZE

    # ---- 第一级：预测级（SATD 粗筛）----
    candidates = intra_pred.select_candidate_modes(
        source_frame.y, mb_x, mb_y, num_candidates
    )
    min_satd = candidates[0].satd  # 候选已按 SATD 升序
    qp = aq.actual_qp(eff_base_qp, min_satd, cfg.aq_enable)

    # ---- 第二级 + 第三级：逐候选变换量化并计算 RD 代价 ----
    stage2 = []
    best_idx = 0
    best_rd = None
    for idx, cand in enumerate(candidates):
        quant_blocks, fwd_blocks, dq_blocks, recon_block = _process_luma_candidate(
            cand, source_frame, mb_x, mb_y, qp
        )

        # 失真：16 个子块的 Psy-RD 加权失真累加
        distortion = 0.0
        for i in range(len(quant_blocks)):
            distortion += psy_rd.hvs_weighted_distortion(
                fwd_blocks[i], dq_blocks[i],
                cfg.psy_rd_enable, cfg.psy_rd_strength,
            )

        # 码率：16 个子块 CAVLC 比特累加 + 固定模式开销
        rate_bits = 0
        for blk in quant_blocks:
            rate_bits += cavlc.block_bits(blk)
        rate_bits += _MODE_OVERHEAD_BITS

        rd = psy_rd.rd_cost(
            distortion, rate_bits, qp,
            cfg.psy_rd_enable, cfg.psy_rd_strength,
        )
        stage2.append(
            CandidateResult(
                mode_type=cand.mode_type,
                mode=cand.mode,
                sub_modes=cand.sub_modes,
                satd=cand.satd,
                quant_blocks=quant_blocks,
                recon_block=recon_block,
                distortion=distortion,
                rate_bits=rate_bits,
                rd_cost=rd,
            )
        )
        if best_rd is None or rd < best_rd:
            best_rd = rd
            best_idx = idx

    best = stage2[best_idx]
    best_cand = candidates[best_idx]

    # ---- 写回亮度重建像素 ----
    for r in range(_MB_SIZE):
        for c in range(_MB_SIZE):
            recon_frame.y[py + r][px + c] = best.recon_block[r][c]

    # ---- 色度：DC 预测简化，码率并入 bits ----
    chroma_bits = _encode_chroma(source_frame, recon_frame, mb_x, mb_y, qp)
    bits = best.rate_bits + chroma_bits

    return MacroblockResult(
        mb_x=mb_x,
        mb_y=mb_y,
        mode_type=best.mode_type,
        mode=best.mode,
        sub_modes=best_cand.sub_modes,
        qp=qp,
        bits=bits,
        pred=best_cand.pred,
        recon=best.recon_block,
        candidates=candidates,
        stage2=stage2,
    )
