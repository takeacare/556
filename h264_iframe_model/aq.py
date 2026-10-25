"""自适应量化 AQ（技术文档 3.2.3 节）。

复用帧内预测阶段已算出的 SATD 作为纹理复杂度度量，无需新增方差/梯度
计算单元：SATD 越大表示纹理越复杂，Qp 偏移越正（量化更粗、省码率）；平坦
区域 SATD 小，Qp 偏移为负或 0（量化更细、保留细节）。

Qp 偏移表 AQ_OFFSET_TABLE 固化为 256 项 ROM，索引为归一化后的 SATD。
"""

# SATD 归一化参考上限：SATD >= 该值即视为最复杂纹理，映射到索引 255。
_SATD_INDEX_MAX = 16384

# Qp 偏移 ROM：长度 256，取值均在 [-6, 6]。
# 索引 0 对应最平坦（偏移 -6），索引 255 对应最复杂（偏移 +6），
# 中间随纹理复杂度线性过渡；256 项由列表推导生成，等价于固化常量表。
AQ_OFFSET_TABLE = [
    max(-6, min(6, round((i - 128) * 6 / 128))) for i in range(256)
]


def qp_offset(satd: int) -> int:
    """由 SATD 映射 Qp 偏移，返回范围 [-6, 6]。

    先把 SATD 归一化/截断到 0..255 的索引（SATD 可能较大，此处按
    _SATD_INDEX_MAX 缩放并截断），再查 AQ_OFFSET_TABLE。
    """
    if satd < 0:
        satd = 0
    idx = (satd * 256) // _SATD_INDEX_MAX
    if idx > 255:
        idx = 255
    if idx < 0:
        idx = 0
    return AQ_OFFSET_TABLE[idx]


def actual_qp(base_qp: int, satd: int, aq_enable: bool = True) -> int:
    """计算实际 Qp = base_qp + qp_offset(satd)，并 clamp 到 0..51。

    aq_enable=False 时直接返回 base_qp（关闭 AQ）。
    """
    if not aq_enable:
        return base_qp
    qp = base_qp + qp_offset(satd)
    if qp < 0:
        return 0
    if qp > 51:
        return 51
    return qp
