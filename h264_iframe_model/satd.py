"""SATD（哈德曼变换绝对差和）代价计算。

采用 4x4 哈德曼变换，仅使用加减法实现，无乘法器，对齐技术文档 3.1.3 节
的 SATD 计算单元描述。
"""


def hadamard4x4(block: list) -> list:
    """对 4x4 块做哈德曼变换，先行后列，仅使用加减法。

    变换矩阵为:
        [ 1  1  1  1]
        [ 1 -1  1 -1]
        [ 1  1 -1 -1]
        [ 1 -1 -1  1]
    """
    tmp = [[0] * 4 for _ in range(4)]
    for r in range(4):
        a0, a1, a2, a3 = block[r][0], block[r][1], block[r][2], block[r][3]
        s0 = a0 + a1
        s1 = a0 - a1
        s2 = a2 + a3
        s3 = a2 - a3
        tmp[r][0] = s0 + s2
        tmp[r][1] = s1 + s3
        tmp[r][2] = s0 - s2
        tmp[r][3] = s1 - s3
    out = [[0] * 4 for _ in range(4)]
    for c in range(4):
        x0, x1, x2, x3 = tmp[0][c], tmp[1][c], tmp[2][c], tmp[3][c]
        s0 = x0 + x1
        s1 = x0 - x1
        s2 = x2 + x3
        s3 = x2 - x3
        out[0][c] = s0 + s2
        out[1][c] = s1 + s3
        out[2][c] = s0 - s2
        out[3][c] = s1 - s3
    return out


def satd4x4(residual: list) -> int:
    """4x4 残差块的 SATD：变换后绝对值之和整除 2。"""
    t = hadamard4x4(residual)
    total = 0
    for row in t:
        for v in row:
            total += v if v >= 0 else -v
    return total // 2


def satd8x8(residual: list) -> int:
    """8x8 残差块：拆成 4 个 4x4 子块累加 SATD。"""
    total = 0
    for by in range(0, 8, 4):
        for bx in range(0, 8, 4):
            sub = [row[bx:bx + 4] for row in residual[by:by + 4]]
            total += satd4x4(sub)
    return total


def satd16x16(residual: list) -> int:
    """16x16 残差块：拆成 16 个 4x4 子块累加 SATD。"""
    total = 0
    for by in range(0, 16, 4):
        for bx in range(0, 16, 4):
            sub = [row[bx:bx + 4] for row in residual[by:by + 4]]
            total += satd4x4(sub)
    return total
