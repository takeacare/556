"""4x4 整数 DCT 正反变换（技术文档 3.2.2 节）。

采用 H.264 标准整数变换，全程定点、只含加法与移位，无浮点运算：

    正向核 Cf = [[ 1, 1, 1, 1],
                 [ 2, 1,-1,-2],
                 [ 1,-1,-1, 1],
                 [ 1,-2, 2,-1]]，   Y = Cf · X · Cfᵀ

    反向核 Ci 由含 1/2 因子的整数蝶形实现，最后加 32 右移 6 位归一化。

正、反变换共用同一套加法器阵列结构，硬件上分时复用（同一组蝶形加法器
先串行算正向、再串行算反向），以节省面积，对应文档 3.2.2 的复用描述。
"""


def forward_transform_4x4(residual: list) -> list:
    """4x4 整数正向变换：Y = Cf · X · Cfᵀ（先行变换再列变换）。

    输入 residual 为 4x4 二维 int 列表，输出 4x4 整数系数。全部由加减法
    与移位实现，无乘法器。单方向蝶形等价于与 Cf 相乘：

        t0 = a0 + a3, t1 = a1 + a2, t2 = a1 - a2, t3 = a0 - a3
        y0 = t0 + t1, y1 = t2 + (t3 << 1), y2 = t0 - t1, y3 = t3 - (t2 << 1)
    """
    tmp = [[0] * 4 for _ in range(4)]
    # 行变换：X · Cfᵀ
    for r in range(4):
        a0, a1, a2, a3 = residual[r][0], residual[r][1], residual[r][2], residual[r][3]
        t0 = a0 + a3
        t1 = a1 + a2
        t2 = a1 - a2
        t3 = a0 - a3
        tmp[r][0] = t0 + t1
        tmp[r][1] = t2 + (t3 << 1)
        tmp[r][2] = t0 - t1
        tmp[r][3] = t3 - (t2 << 1)

    out = [[0] * 4 for _ in range(4)]
    # 列变换：Cf · (中间结果)
    for c in range(4):
        a0, a1, a2, a3 = tmp[0][c], tmp[1][c], tmp[2][c], tmp[3][c]
        t0 = a0 + a3
        t1 = a1 + a2
        t2 = a1 - a2
        t3 = a0 - a3
        out[0][c] = t0 + t1
        out[1][c] = t2 + (t3 << 1)
        out[2][c] = t0 - t1
        out[3][c] = t3 - (t2 << 1)
    return out


def inverse_transform_4x4(coeffs: list) -> list:
    """4x4 整数反向变换，返回 4x4 近似残差。

    采用标准反向核（含 1/2 因子的 Ci）的整数蝶形，先行后列；最后加 32
    右移 6 位做归一化，使与正向变换的往返误差落在整数变换固有范围内。
    只含加法与移位，与正向变换共用同一蝶形加法器阵列（分时复用）。
    """
    tmp = [[0] * 4 for _ in range(4)]
    # 列反向蝶形：Ciᵀ · (中间结果)
    for c in range(4):
        a0 = coeffs[0][c] + coeffs[2][c]
        a1 = coeffs[0][c] - coeffs[2][c]
        a2 = (coeffs[1][c] >> 1) - coeffs[3][c]
        a3 = coeffs[1][c] + (coeffs[3][c] >> 1)
        tmp[0][c] = a0 + a3
        tmp[1][c] = a1 + a2
        tmp[2][c] = a1 - a2
        tmp[3][c] = a0 - a3

    out = [[0] * 4 for _ in range(4)]
    # 行反向蝶形 + 归一化移位
    for r in range(4):
        a0 = tmp[r][0] + tmp[r][2]
        a1 = tmp[r][0] - tmp[r][2]
        a2 = (tmp[r][1] >> 1) - tmp[r][3]
        a3 = tmp[r][1] + (tmp[r][3] >> 1)
        out[r][0] = (a0 + a3 + 32) >> 6
        out[r][1] = (a1 + a2 + 32) >> 6
        out[r][2] = (a1 - a2 + 32) >> 6
        out[r][3] = (a0 - a3 + 32) >> 6
    return out
