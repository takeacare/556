"""Psy-RD 频域 HVS 加权与 RD 代价（技术文档 3.5.1 节）。

RD 计算前对每个变换系数乘以对应 HVS 权重：低频（人眼更敏感）权重大、高频
权重小。权重为定点数（放大 16 倍的整数），实际用移位 + 加法实现（乘权重
后右移 4 位），避免浮点乘法器。
"""

# 4x4 HVS 权重表（定点，放大 16 倍：表中 16 表示权重 1.0）。
# 位置 (0,0) 为低频 DC，权重最大；越往右下高频，权重越小。
HVS_WEIGHT_TABLE_4X4 = [
    [16, 13, 11, 10],
    [13, 11, 10, 9],
    [11, 10, 9, 8],
    [10, 9, 8, 8],
]

_HVS_SHIFT = 4  # 权重定点小数位数（放大 16 倍，故右移 4 位还原）


def hvs_weighted_distortion(orig_coeffs: list, recon_coeffs: list,
                            psy_rd_enable: bool = True,
                            strength: float = 1.0) -> float:
    """变换域失真。

    Psy-RD 开启时：按 HVS 权重加权的平方误差和，每项为
    (w/16 · strength) · (orig - recon)^2，w/16 由右移 4 位实现；
    关闭时：退化为普通平方误差和（SSE，不加权）。
    """
    total = 0.0
    w_tab = HVS_WEIGHT_TABLE_4X4
    for i in range(4):
        for j in range(4):
            d = orig_coeffs[i][j] - recon_coeffs[i][j]
            if psy_rd_enable:
                w = (w_tab[i][j] * strength) / (1 << _HVS_SHIFT)
                total += w * d * d
            else:
                total += d * d
    return total


def lambda_for_qp(qp: int) -> float:
    """由 Qp 计算拉格朗日乘子（x264 风格，随 Qp 单调递增）。"""
    return 0.85 * (2.0 ** ((qp - 12) / 3.0))


def rd_cost(distortion: float, rate_bits: int, qp: int,
            psy_rd_enable: bool = True,
            psy_rd_strength: float = 1.0) -> float:
    """RD 代价 = distortion + lambda · rate_bits。

    lambda 由 Qp 决定（lambda_for_qp）。psy_rd_enable=True 时 lambda 再按
    psy_rd_strength 缩放（强度越大越偏重失真项）；关闭时使用普通 lambda。
    """
    lam = lambda_for_qp(qp)
    if psy_rd_enable:
        lam *= psy_rd_strength
    return distortion + lam * rate_bits
