"""内容自适应去块滤波（技术文档 3.3 节）。

去块滤波用于消除基于块的预测/变换量化引入的块效应。本模块采用内容自适应
策略：由边界两侧像素差推导边界强度 BS（平坦边界 BS 大、强滤波；细节/边缘
边界 BS 小、弱滤波/不滤波），再依据由 Qp 推导的 alpha/beta 阈值控制是否滤波
与滤波幅度，仅处理宏块网格的左边界与上边界（符合标准的光栅扫描顺序），避免
模糊真实细节。

H.264 去块滤波的基本思路：
    - bs == 0      : 不滤波
    - 0 < bs < 4   : 弱滤波（仅修正 p0/p1/q0/q1，必要时不动 p1/q1）
    - bs == 4      : 强滤波（修正 p0..p2 / q0..q2）
其中 alpha/beta 阈值随 Qp 增大而增大，Qp 越大允许滤波的像素差越大。
"""

# 由 Qp 索引的阈值表（Table 8-16/8-17 的近似，长度 52，随 Qp 单调不减）。
# alpha 控制“是否滤波”的允许像素差，beta 控制邻域平坦度判定。
_ALPHA_TABLE = (
    [0] * 16            # Qp 0..15
    + [4] * 12          # Qp 16..27
    + [6] * 8           # Qp 28..35
    + [7] * 4           # Qp 36..39
    + [8] * 4           # Qp 40..43
    + [9] * 4           # Qp 44..47
    + [10] * 4          # Qp 48..51
)
_BETA_TABLE = (
    [0] * 16            # Qp 0..15
    + [2] * 12          # Qp 16..27
    + [3] * 8           # Qp 28..35
    + [3] * 4           # Qp 36..39
    + [4] * 4           # Qp 40..43
    + [4] * 4           # Qp 44..47
    + [5] * 4           # Qp 48..51
)
# 弱滤波的裁剪幅度上限 tc0（Table 8-16 中 tc0 列的近似）。
_TC0_TABLE = (
    [0] * 12            # Qp 0..11
    + [0] * 8           # Qp 12..19
    + [1] * 8           # Qp 20..27
    + [1] * 8           # Qp 28..35
    + [2] * 8           # Qp 36..43
    + [2] * 4           # Qp 44..47
    + [3] * 4           # Qp 48..51
)

# 色度平坦判定阈值：总差值不超过该值视为平坦（返回 BS=2）
_CHROMA_FLAT_THRESHOLD = 3


def _clip(value: int, low: int, high: int) -> int:
    """把 value 裁剪到 [low, high] 区间。"""
    if value < low:
        return low
    if value > high:
        return high
    return value


def _clip_pixel(value: int) -> int:
    """把像素值裁剪到 [0, 255]。"""
    return _clip(value, 0, 255)


def boundary_strength(p: list, q: list, is_chroma: bool = False) -> int:
    """由边界两侧像素差推导边界强度 BS（0~4）。

    参数:
        p: 边界一侧（左/上）靠近边界的 4 个像素，p[0] 最靠近边界。
        q: 边界另一侧（右/下）靠近边界的 4 个像素，q[0] 最靠近边界。
        is_chroma: 是否为色度，色度只有 {0, 2} 两档低强度。

    说明:
        采用内容自适应判据：以 |p0-q0| 及邻域差 |p1-p0|、|q1-q0| 之和作为
        “边界平坦度”度量。差值越小说明边界越平坦（多为块效应），BS 越大、
        越倾向强滤波；差值越大说明此处是真实细节/边缘，BS 越小、弱滤波甚至
        不滤波，从而保护细节不被模糊。
    """
    d0 = abs(p[0] - q[0])       # 边界两侧中心像素差
    d1 = abs(p[1] - p[0])       # 边界侧邻域梯度
    d2 = abs(q[1] - q[0])
    total = d0 + d1 + d2

    if is_chroma:
        # 色度只有 {0, 2} 两档：平坦 -> 强（色度意义上的 2 级），否则不滤波
        return 2 if total <= _CHROMA_FLAT_THRESHOLD else 0

    # 差值小 -> BS 高（强滤波）；差值大 -> BS 低（弱滤波/不滤波）
    if total <= 2:
        return 4
    if total <= 6:
        return 3
    if total <= 12:
        return 2
    if total <= 24:
        return 1
    return 0


def filter_edge(p: list, q: list, bs: int, qp: int, is_chroma: bool = False) -> tuple:
    """按 BS 与 Qp 阈值对一条边界实施滤波。

    参数:
        p: 边界一侧 4 个像素（p[0] 最靠近边界）。
        q: 边界另一侧 4 个像素（q[0] 最靠近边界）。
        bs: 边界强度，0 表示不滤波。
        qp: 量化参数，用于推导 alpha/beta/tc0 阈值。
        is_chroma: 是否为色度（上限 BS 为 2）。

    返回:
        (p_out, q_out) 两个长度 4 的列表，为滤波后的像素段。
    """
    p_out = list(p)
    q_out = list(q)

    if bs <= 0:
        return p_out, q_out
    if is_chroma and bs > 2:
        bs = 2

    idx = _clip(qp, 0, 51)
    alpha = _ALPHA_TABLE[idx]
    beta = _BETA_TABLE[idx]
    tc0 = _TC0_TABLE[idx]

    p0, p1, p2, p3 = p_out[0], p_out[1], p_out[2], p_out[3]
    q0, q1, q2, q3 = q_out[0], q_out[1], q_out[2], q_out[3]

    # 由 Qp 推导的阈值控制是否滤波：差值过大（真实边缘）则跳过，保护细节
    if abs(p0 - q0) >= alpha:
        return p_out, q_out
    if abs(p1 - p0) >= beta or abs(q1 - q0) >= beta:
        return p_out, q_out

    if bs < 4:
        # 弱滤波：修正 p0/q0，邻域足够平坦时再修正 p1/q1
        delta = ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3
        delta = _clip(delta, -tc0, tc0)
        p0n = _clip_pixel(p0 + delta)
        q0n = _clip_pixel(q0 - delta)

        if abs(p2 - p0) < beta:
            dp1 = (p2 + ((p0 + q0 + 1) >> 1) - 2 * p1) >> 1
            dp1 = _clip(dp1, -tc0, tc0)
            p1n = _clip_pixel(p1 + dp1)
        else:
            p1n = p1
        if abs(q2 - q0) < beta:
            dq1 = (q2 + ((p0 + q0 + 1) >> 1) - 2 * q1) >> 1
            dq1 = _clip(dq1, -tc0, tc0)
            q1n = _clip_pixel(q1 + dq1)
        else:
            q1n = q1

        p_out[0], p_out[1] = p0n, p1n
        q_out[0], q_out[1] = q0n, q1n
        return p_out, q_out

    # 强滤波（bs == 4）：对 p0..p2 / q0..q2 做平滑
    p0n = _clip_pixel((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3)
    p1n = _clip_pixel((p2 + p1 + p0 + q0 + 2) >> 2)
    p2n = _clip_pixel((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3)
    q0n = _clip_pixel((q2 + 2 * q1 + 2 * q0 + 2 * p0 + p1 + 4) >> 3)
    q1n = _clip_pixel((q2 + q1 + q0 + p0 + 2) >> 2)
    q2n = _clip_pixel((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3)

    p_out[0], p_out[1], p_out[2] = p0n, p1n, p2n
    q_out[0], q_out[1], q_out[2] = q0n, q1n, q2n
    return p_out, q_out


def deblock_plane(plane: list, qp: int, is_chroma: bool = False,
                  enable: bool = True, mb_size: int = 16) -> None:
    """就地对整平面按宏块网格滤波。

    仅处理每个宏块的左边界（列 x = mb_x*mb_size）与上边界（行 y = mb_y*mb_size），
    跳过帧最左/最上边界（符合标准扫描顺序，边界外无参考像素）。
    """
    if not enable:
        return
    ph = len(plane)
    pw = len(plane[0]) if ph else 0
    if ph < 4 or pw < 4:
        return

    for mb_y in range(0, ph, mb_size):
        for mb_x in range(0, pw, mb_size):
            # 垂直边界（左边界）：p 在左、q 在右，逐行处理
            x = mb_x
            if x >= 4 and x + 4 <= pw:
                for y in range(mb_y, min(mb_y + mb_size, ph)):
                    p = [plane[y][x - 1], plane[y][x - 2],
                         plane[y][x - 3], plane[y][x - 4]]
                    q = [plane[y][x], plane[y][x + 1],
                         plane[y][x + 2], plane[y][x + 3]]
                    bs = boundary_strength(p, q, is_chroma)
                    np_, nq_ = filter_edge(p, q, bs, qp, is_chroma)
                    plane[y][x - 1], plane[y][x - 2] = np_[0], np_[1]
                    plane[y][x - 3], plane[y][x - 4] = np_[2], np_[3]
                    plane[y][x], plane[y][x + 1] = nq_[0], nq_[1]
                    plane[y][x + 2], plane[y][x + 3] = nq_[2], nq_[3]

            # 水平边界（上边界）：p 在上、q 在下，逐列处理
            yb = mb_y
            if yb >= 4 and yb + 4 <= ph:
                for x in range(mb_x, min(mb_x + mb_size, pw)):
                    p = [plane[yb - 1][x], plane[yb - 2][x],
                         plane[yb - 3][x], plane[yb - 4][x]]
                    q = [plane[yb][x], plane[yb + 1][x],
                         plane[yb + 2][x], plane[yb + 3][x]]
                    bs = boundary_strength(p, q, is_chroma)
                    np_, nq_ = filter_edge(p, q, bs, qp, is_chroma)
                    plane[yb - 1][x], plane[yb - 2][x] = np_[0], np_[1]
                    plane[yb - 3][x], plane[yb - 4][x] = np_[2], np_[3]
                    plane[yb][x], plane[yb + 1][x] = nq_[0], nq_[1]
                    plane[yb + 2][x], plane[yb + 3][x] = nq_[2], nq_[3]


def deblock_frame(frame, qp: int, enable: bool = True) -> None:
    """对整帧去块滤波：Y 用 16x16 网格自适应滤波，U/V 用 8x8 网格且按色度规则。"""
    deblock_plane(frame.y, qp, is_chroma=False, enable=enable, mb_size=16)
    deblock_plane(frame.u, qp, is_chroma=True, enable=enable, mb_size=8)
    deblock_plane(frame.v, qp, is_chroma=True, enable=enable, mb_size=8)
