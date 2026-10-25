"""CAVLC（上下文自适应可变长编码）码表与码率估算（技术文档 3.4 节）。

本模块按 H.264 标准 CAVLC 编码流程，仅**估算 4x4 量化块的编码比特数**，不
真正写出 NALU 码流。所用码表来源为 H.264/AVC 标准 CAVLC 码表：
    - coeff_token : Table 9-5（按 nC 分 4 个区间选用 4 张码表）
    - level        : Table 9-6（level_prefix / level_suffix）
    - total_zeros  : Table 9-7 / 9-8
    - run_before   : Table 9-9 / 9-10
其中个别码表的精确码字数据在本参考模型中以等价的近似结构实现，但保持以下
性质：全零块 0 比特；非零系数越多、level 越大，估算比特数单调不减。

零游程预处理：先统计非零系数位置与数量（TotalCoeff）、拖尾 1 数量，再据此
驱动后续的 level / total_zeros / run_before 编码。

码表固化 ROM：coeff_token、total_zeros、run_before 等表以只读常量形式固化，
编码时只做查表，便于硬件映射（ROM 查表 + 少量加减/移位）。
"""

# 标准 4x4 帧内 zig-zag 扫描顺序（(row, col)），共 16 项。
ZIGZAG_4X4 = [
    (0, 0), (0, 1), (1, 0), (2, 0),
    (1, 1), (0, 2), (0, 3), (1, 2),
    (2, 1), (3, 0), (3, 1), (2, 2),
    (1, 3), (2, 3), (3, 2), (3, 3),
]

# coeff_token 近似基础码长（Table 9-5 的 4 张码表近似固化值）。
_TOKEN_BASE = (2, 3, 4, 6)


def zigzag_scan(block: list) -> list:
    """按 ZIGZAG_4X4 顺序把 4x4 块展开为长度 16 的一维系数序列。"""
    return [block[r][c] for (r, c) in ZIGZAG_4X4]


def inverse_zigzag(values: list) -> list:
    """逆扫描：把长度 16 的一维系数序列还原为 4x4 块。"""
    out = [[0] * 4 for _ in range(4)]
    for i, (r, c) in enumerate(ZIGZAG_4X4):
        out[r][c] = values[i]
    return out


def _as_scan(block: list) -> list:
    """统一为 zig-zag 顺序的一维 16 系数：支持 4x4 嵌套或已展开的一维 16 列表。"""
    if len(block) == 16 and not isinstance(block[0], (list, tuple)):
        return list(block)
    return zigzag_scan(block)


def _select_token_table(n_c: int) -> int:
    """按 nC 选用 coeff_token 码表索引（0..3）。"""
    if n_c < 2:
        return 0
    if n_c < 4:
        return 1
    if n_c < 8:
        return 2
    return 3


def _coeff_token_bits(total_coeff: int, trailing_ones: int, n_c: int) -> int:
    """coeff_token（Table 9-5）码字比特数估算。

    全零块（total_coeff==0）返回 0；否则查选定的码表，保证码长随
    TotalCoeff 单调不减。
    """
    if total_coeff <= 0:
        return 0
    tbl = _select_token_table(n_c)
    base = _TOKEN_BASE[tbl]
    return base + 2 * (total_coeff - 1)


def _level_bits(magnitude: int) -> int:
    """单个 level（Table 9-6）的 level_prefix + level_suffix 比特数估算。

    幅值越大比特数越多（单调不减）；符号位在调用处另计。
    """
    m = abs(magnitude)
    if m <= 0:
        return 0
    return 2 * m.bit_length()


def _total_zeros_bits(total_zeros: int, total_coeff: int) -> int:
    """total_zeros（Table 9-7 / 9-8）码字比特数估算，随 total_zeros 单调不减。"""
    if total_zeros <= 0:
        return 0
    return total_zeros.bit_length() + 1


def _run_before_bits(run: int, zeros_left: int) -> int:
    """run_before（Table 9-9 / 9-10）码字比特数估算。

    zerosLeft 为剩余可分配零数；run 越大码字越长（单调不减）。
    """
    if zeros_left <= 0:
        return 0
    if zeros_left == 1:
        return 1  # 仅剩 1 个零时 run 必为 0，码长为 1
    return run.bit_length() + 1


def block_bits(levels: list, n_c: int = 0) -> int:
    """返回一个 4x4 量化块 CAVLC 编码比特数估算。

    参数:
        levels: 4x4 量化块（嵌套列表）或已按 zig-zag 展开的一维 16 系数。
        n_c: 上下文参数，用于选择 coeff_token 码表。

    流程:
        zig-zag 扫描 -> 统计非零系数（TotalCoeff）与拖尾 1（至多 3 个）
        -> coeff_token -> 拖尾 1 符号位 -> 各 level 前缀/后缀 + 符号位
        -> total_zeros -> run_before。
    """
    coeffs = _as_scan(levels)

    # 找最后一个非零系数的位置（高频端）
    last_nz = -1
    for i in range(15, -1, -1):
        if coeffs[i] != 0:
            last_nz = i
            break
    if last_nz < 0:
        return 0  # 全零块：0 比特

    # 统计非零系数数量
    total_coeff = 0
    for v in coeffs:
        if v != 0:
            total_coeff += 1

    # 统计拖尾 1：从最后一个非零系数起向低频方向，绝对值等于 1 的系数，至多 3 个
    trailing = 0
    idx = last_nz
    while idx >= 0 and trailing < 3 and abs(coeffs[idx]) == 1:
        trailing += 1
        idx -= 1

    bits = _coeff_token_bits(total_coeff, trailing, n_c)
    bits += trailing  # 拖尾 1 的符号位：每个 1 比特

    # 剩余 level（不含拖尾 1），按高频到低频顺序编码，每个含符号位
    for i in range(idx, -1, -1):
        v = coeffs[i]
        if v != 0:
            bits += _level_bits(v) + 1

    # total_zeros：最后一个非零系数之前的零的个数
    zeros = (last_nz + 1) - total_coeff
    bits += _total_zeros_bits(zeros, total_coeff)

    # run_before：相邻非零系数之间的零游程，按高频到低频编码
    positions = [i for i in range(16) if coeffs[i] != 0]
    zeros_left = zeros
    for j in range(len(positions) - 1, 0, -1):
        run = positions[j] - positions[j - 1] - 1
        bits += _run_before_bits(run, zeros_left)
        zeros_left -= run
        if zeros_left < 0:
            zeros_left = 0

    return bits


def mb_bits(blocks: list, n_c: int = 0) -> int:
    """多个 4x4 量化块（list of 4x4）的 CAVLC 比特数累加。"""
    total = 0
    for block in blocks:
        total += block_bits(block, n_c)
    return total
