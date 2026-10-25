"""图像质量评价指标：MSE / PSNR / SSIM（技术文档 §7）。

测试与指标脚本共用；输入均为二维 int 列表（亮度平面）。SSIM 采用全局
（整幅窗）简化实现，不引入第三方依赖。尺寸不一致时抛出 ValueError。
"""

import math


def _check_same_shape(a, b) -> None:
    """校验两个二维平面尺寸一致且非空，否则抛 ValueError。"""
    if len(a) != len(b):
        raise ValueError("两平面行数不一致: %d vs %d" % (len(a), len(b)))
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            raise ValueError("两平面列数不一致: %d vs %d" % (len(ra), len(rb)))
    if len(a) == 0 or len(a[0]) == 0:
        raise ValueError("空平面无法计算指标")


def mse(a, b) -> float:
    """两个二维平面的均方误差。"""
    _check_same_shape(a, b)
    h = len(a)
    w = len(a[0])
    n = h * w
    acc = 0.0
    for r in range(h):
        row_a = a[r]
        row_b = b[r]
        for c in range(w):
            d = row_a[c] - row_b[c]
            acc += d * d
    return acc / n


def psnr(a, b, max_val: int = 255) -> float:
    """峰值信噪比（dB）。MSE 为 0 时返回 float('inf')。"""
    m = mse(a, b)
    if m == 0.0:
        return float("inf")
    return 10.0 * math.log10((max_val * max_val) / m)


def ssim(a, b, max_val: int = 255) -> float:
    """全局 SSIM（整幅窗简化实现），取值范围约 [-1, 1]，相同图像为 1。"""
    _check_same_shape(a, b)
    h = len(a)
    w = len(a[0])
    n = h * w
    sum_a = sum_b = sum_aa = sum_bb = sum_ab = 0.0
    for r in range(h):
        row_a = a[r]
        row_b = b[r]
        for c in range(w):
            va = row_a[c]
            vb = row_b[c]
            sum_a += va
            sum_b += vb
            sum_aa += va * va
            sum_bb += vb * vb
            sum_ab += va * vb
    mu_a = sum_a / n
    mu_b = sum_b / n
    var_a = sum_aa / n - mu_a * mu_a
    var_b = sum_bb / n - mu_b * mu_b
    if var_a < 0.0:
        var_a = 0.0
    if var_b < 0.0:
        var_b = 0.0
    cov = sum_ab / n - mu_a * mu_b
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    return num / den


def frame_psnr(orig_frame, recon_frame) -> float:
    """对 Y（亮度）平面求 PSNR。"""
    return psnr(orig_frame.y, recon_frame.y, 255)


def frame_ssim(orig_frame, recon_frame) -> float:
    """对 Y（亮度）平面求 SSIM。"""
    return ssim(orig_frame.y, recon_frame.y, 255)
