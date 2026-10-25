"""简化版 CRF 恒定质量码率控制（技术文档 3.5.2 节）。

对齐 x264 的 CRF 逻辑：以全帧 SATD 累加值作为帧复杂度度量，复用该度量查表
调整基准 Qp——纹理/细节越复杂（SATD 越大）分配码率越多、Qp 越高，从而实现
“恒定质量”而非恒定码率。
"""

# 复杂度（每宏块平均 SATD）到 Qp 偏移量的查表阈值。
# 复用全帧 SATD 累加值作为复杂度度量，查表调整 Qp，对齐 x264 CRF 逻辑。
_COMPLEXITY_TABLE = (
    (20000, 8),
    (8000, 6),
    (2000, 4),
    (500, 2),
    (100, 1),
)


def frame_complexity(satd_sum: int, num_mbs: int) -> float:
    """由全帧 SATD 累加值与宏块数得到归一化复杂度（>= 0）。

    以“每宏块平均 SATD”作为归一化复杂度：值越大说明帧纹理越复杂。
    """
    if num_mbs <= 0:
        return 0.0
    avg = float(satd_sum) / float(num_mbs)
    if avg < 0.0:
        return 0.0
    return avg


def adjust_qp(base_qp: int, complexity: float) -> int:
    """由复杂度查表调整基准 Qp，结果 clamp 到 0..51；复杂度越高 Qp 越高。

    复杂度越大，查表得到越大的正向偏移量，故本函数关于 complexity 单调不减。
    """
    if complexity < 0.0:
        complexity = 0.0

    delta = 0
    for threshold, offset in _COMPLEXITY_TABLE:
        if complexity >= threshold:
            delta = offset
            break

    qp = base_qp + delta
    if qp < 0:
        return 0
    if qp > 51:
        return 51
    return qp


def crf_base_qp(base_qp: int, satd_sum: int, num_mbs: int, enable: bool = True) -> int:
    """便捷入口：CRF 码率控制下求实际使用的 Qp。

    enable=False 时直接返回 base_qp（关闭 CRF）。
    """
    if not enable:
        return base_qp
    return adjust_qp(base_qp, frame_complexity(satd_sum, num_mbs))
