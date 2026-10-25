"""帧内预测生成与粗筛模式决策。

实现 16x16（4 种模式）与 4x4（9 种模式）帧内预测，并对齐 x264 的两级
SATD 粗筛逻辑：16x16 各模式 + 4x4 逐子块最优模式拼装，共 5 个候选中按
SATD 升序取前 num_candidates 个。4x4 预测公式对齐 x264/FFmpeg 标准实现。
"""

from dataclasses import dataclass

from . import satd as _satd

# ---- 16x16 预测模式 ----
I16_VERTICAL = 0
I16_HORIZONTAL = 1
I16_DC = 2
I16_PLANE = 3

# ---- 4x4 预测模式（与 H.264 一致的九模式）----
I4_VERTICAL = 0        # 垂直
I4_HORIZONTAL = 1      # 水平
I4_DC = 2              # DC
I4_DOWN_LEFT = 3       # 下左（Diagonal Down Left）
I4_DOWN_RIGHT = 4      # 下右（Diagonal Down Right）
I4_VERTICAL_RIGHT = 5  # 垂直右（Vertical Right）
I4_HORIZONTAL_DOWN = 6  # 水平下（Horizontal Down）
I4_VERTICAL_LEFT = 7   # 垂直左（Vertical Left）
I4_HORIZONTAL_UP = 8   # 水平上（Horizontal Up）


@dataclass
class ModeCandidate:
    """一个候选预测模式及其代价。"""

    mode_type: str        # "I16" 或 "I4"
    mode: int             # I16 模式号；I4 整宏块候选为 -1
    satd: int             # 16x16 残差 SATD 代价
    pred: list            # 16x16 预测像素
    sub_modes: list = None  # I4 候选的 4x4 子模式矩阵（4x4）


def _clip(v: int) -> int:
    """裁剪到 0..255 像素范围。"""
    if v < 0:
        return 0
    if v > 255:
        return 255
    return v


def _f1(a: int, b: int) -> int:
    """(a + b + 1) >> 1 线性插值。"""
    return (a + b + 1) >> 1


def _f2(a: int, b: int, c: int) -> int:
    """(a + 2b + c + 2) >> 2 三点插值滤波。"""
    return (a + 2 * b + c + 2) >> 2


def gather_mb_reference(plane: list, mb_x: int, mb_y: int) -> dict:
    """采集宏块预测参考像素。

    返回 {"top": 16 像素, "left": 16 像素, "top_left": int}。
    mb_x/mb_y 为宏块坐标；越界参考像素取 128。
    """
    px = mb_x * 16
    py = mb_y * 16
    ph = len(plane)
    pw = len(plane[0]) if ph else 0
    top = []
    for c in range(16):
        x = px + c
        y = py - 1
        if 0 <= y < ph and 0 <= x < pw:
            top.append(plane[y][x])
        else:
            top.append(128)
    left = []
    for r in range(16):
        x = px - 1
        y = py + r
        if 0 <= y < ph and 0 <= x < pw:
            left.append(plane[y][x])
        else:
            left.append(128)
    if 0 <= py - 1 < ph and 0 <= px - 1 < pw:
        top_left = plane[py - 1][px - 1]
    else:
        top_left = 128
    return {"top": top, "left": left, "top_left": top_left}


def predict_16x16(mode: int, top: list, left: list, top_left: int) -> list:
    """按 H.264 标准生成 16x16 预测块，返回 16x16 二维列表。"""
    if mode == I16_VERTICAL:
        return [[top[c] for c in range(16)] for _ in range(16)]
    if mode == I16_HORIZONTAL:
        return [[left[r] for c in range(16)] for r in range(16)]
    if mode == I16_DC:
        dc = (sum(top) + sum(left) + 16) >> 5
        return [[dc] * 16 for _ in range(16)]
    if mode == I16_PLANE:
        # 标准平面模式整数近似（对齐 x264 predict_16x16_p）
        h = 0
        v = 0
        for i in range(8):
            h += (i + 1) * (top[8 + i] - top[6 - i])
            v += (i + 1) * (left[8 + i] - left[6 - i])
        a = 16 * (top[15] + left[15])
        b = (5 * h + 32) >> 6
        c = (5 * v + 32) >> 6
        i00 = a - b * 7 - c * 7 + 16
        pred = []
        for y in range(16):
            row = []
            pix = i00
            for x in range(16):
                row.append(_clip(pix >> 5))
                pix += b
            pred.append(row)
            i00 += c
        return pred
    raise ValueError("未知 16x16 模式: %r" % mode)


def gather_4x4_reference(plane: list, x0: int, y0: int) -> dict:
    """采集 4x4 预测参考像素。

    参考模型简化处理：直接从 plane 按宏块内 4x4 子块光栅顺序取上方/左方
    像素，越界取 128（不区分相邻块可用性）。x0/y0 为绝对像素坐标。
    返回 {"top": 8 像素, "left": 4 像素, "top_left": int}。
    """
    ph = len(plane)
    pw = len(plane[0]) if ph else 0
    top = []
    for c in range(8):
        x = x0 + c
        y = y0 - 1
        if 0 <= y < ph and 0 <= x < pw:
            top.append(plane[y][x])
        else:
            top.append(128)
    left = []
    for r in range(4):
        x = x0 - 1
        y = y0 + r
        if 0 <= y < ph and 0 <= x < pw:
            left.append(plane[y][x])
        else:
            left.append(128)
    if 0 <= y0 - 1 < ph and 0 <= x0 - 1 < pw:
        top_left = plane[y0 - 1][x0 - 1]
    else:
        top_left = 128
    return {"top": top, "left": left, "top_left": top_left}


def predict_4x4(mode: int, top: list, left: list, top_left: int) -> list:
    """按 H.264 标准生成 4x4 预测块。

    top 为 8 个参考像素（上方 4 + 右上 4），left 为 4 个，top_left 为角点。
    公式对齐 x264 的 predict_4x4 实现。
    """
    t0, t1, t2, t3, t4, t5, t6, t7 = top[0:8]
    l0, l1, l2, l3 = left[0:4]
    lt = top_left
    p = [[0] * 4 for _ in range(4)]

    if mode == I4_VERTICAL:
        for r in range(4):
            for c in range(4):
                p[r][c] = top[c]
    elif mode == I4_HORIZONTAL:
        for r in range(4):
            for c in range(4):
                p[r][c] = left[r]
    elif mode == I4_DC:
        dc = (t0 + t1 + t2 + t3 + l0 + l1 + l2 + l3 + 4) >> 3
        for r in range(4):
            for c in range(4):
                p[r][c] = dc
    elif mode == I4_DOWN_LEFT:
        p[0][0] = _f2(t0, t1, t2)
        p[0][1] = p[1][0] = _f2(t1, t2, t3)
        p[0][2] = p[1][1] = p[2][0] = _f2(t2, t3, t4)
        p[0][3] = p[1][2] = p[2][1] = p[3][0] = _f2(t3, t4, t5)
        p[1][3] = p[2][2] = p[3][1] = _f2(t4, t5, t6)
        p[2][3] = p[3][2] = _f2(t5, t6, t7)
        p[3][3] = _f2(t6, t7, t7)
    elif mode == I4_DOWN_RIGHT:
        p[0][3] = _f2(t3, t2, t1)
        p[0][2] = p[1][3] = _f2(t2, t1, t0)
        p[0][1] = p[1][2] = p[2][3] = _f2(t1, t0, lt)
        p[0][0] = p[1][1] = p[2][2] = p[3][3] = _f2(t0, lt, l0)
        p[1][0] = p[2][1] = p[3][2] = _f2(lt, l0, l1)
        p[2][0] = p[3][1] = _f2(l0, l1, l2)
        p[3][0] = _f2(l1, l2, l3)
    elif mode == I4_VERTICAL_RIGHT:
        p[3][0] = _f2(l2, l1, l0)
        p[2][0] = _f2(l1, l0, lt)
        p[1][0] = p[3][1] = _f2(l0, lt, t0)
        p[0][0] = p[2][1] = _f1(lt, t0)
        p[1][1] = p[3][2] = _f2(lt, t0, t1)
        p[0][1] = p[2][2] = _f1(t0, t1)
        p[1][2] = p[3][3] = _f2(t0, t1, t2)
        p[0][2] = p[2][3] = _f1(t1, t2)
        p[1][3] = _f2(t1, t2, t3)
        p[0][3] = _f1(t2, t3)
    elif mode == I4_HORIZONTAL_DOWN:
        p[3][0] = _f1(l2, l3)
        p[3][1] = _f2(l1, l2, l3)
        p[2][0] = p[3][2] = _f1(l1, l2)
        p[2][1] = p[3][3] = _f2(l0, l1, l2)
        p[1][0] = p[2][2] = _f1(l0, l1)
        p[1][1] = p[2][3] = _f2(lt, l0, l1)
        p[0][0] = p[1][2] = _f1(lt, l0)
        p[0][1] = p[1][3] = _f2(t0, lt, l0)
        p[0][2] = _f2(t1, t0, lt)
        p[0][3] = _f2(t2, t1, t0)
    elif mode == I4_VERTICAL_LEFT:
        p[0][0] = _f1(t0, t1)
        p[0][1] = _f1(t1, t2)
        p[0][2] = _f1(t2, t3)
        p[0][3] = _f1(t3, t4)
        p[1][0] = _f2(t0, t1, t2)
        p[1][1] = _f2(t1, t2, t3)
        p[1][2] = _f2(t2, t3, t4)
        p[1][3] = _f2(t3, t4, t5)
        p[2][0] = _f1(t1, t2)
        p[2][1] = _f1(t2, t3)
        p[2][2] = _f1(t3, t4)
        p[2][3] = _f1(t4, t5)
        p[3][0] = _f2(t1, t2, t3)
        p[3][1] = _f2(t2, t3, t4)
        p[3][2] = _f2(t3, t4, t5)
        p[3][3] = _f2(t4, t5, t6)
    elif mode == I4_HORIZONTAL_UP:
        p[0][0] = _f1(l0, l1)
        p[0][1] = _f2(l0, l1, l2)
        p[0][2] = _f1(l1, l2)
        p[0][3] = _f2(l1, l2, l3)
        p[1][0] = _f1(l1, l2)
        p[1][1] = _f2(l1, l2, l3)
        p[1][2] = _f1(l2, l3)
        p[1][3] = _f2(l2, l3, l3)
        p[2][0] = _f1(l2, l3)
        p[2][1] = _f2(l2, l3, l3)
        p[2][2] = l3
        p[2][3] = l3
        p[3][0] = l3
        p[3][1] = l3
        p[3][2] = l3
        p[3][3] = l3
    else:
        raise ValueError("未知 4x4 模式: %r" % mode)

    return [[_clip(p[r][c]) for c in range(4)] for r in range(4)]


def select_candidate_modes(plane: list, mb_x: int, mb_y: int, num_candidates: int = 2) -> list:
    """粗筛候选模式，按 SATD 升序返回前 num_candidates 个。

    流程：①16x16 四模式各生成预测块并算 satd16x16，得 4 个 I16 候选；
    ②16 个 4x4 子块各遍历 9 模式取 SATD 最小者，拼装整宏块预测并算
    satd16x16，得 1 个 I4 候选；③合并 5 个候选后按 satd 升序截断。
    """
    px = mb_x * 16
    py = mb_y * 16
    src = [[plane[py + r][px + c] for c in range(16)] for r in range(16)]

    def _residual(pred):
        return [[src[r][c] - pred[r][c] for c in range(16)] for r in range(16)]

    candidates = []

    # ① 16x16 四种模式
    ref16 = gather_mb_reference(plane, mb_x, mb_y)
    for mode in (I16_VERTICAL, I16_HORIZONTAL, I16_DC, I16_PLANE):
        pred = predict_16x16(mode, ref16["top"], ref16["left"], ref16["top_left"])
        cost = _satd.satd16x16(_residual(pred))
        candidates.append(ModeCandidate("I16", mode, cost, pred))

    # ② 16 个 4x4 子块逐个选择最优方向模式并拼装
    sub_modes = [[0] * 4 for _ in range(4)]
    pred4 = [[0] * 16 for _ in range(16)]
    for sr in range(4):
        for sc in range(4):
            x0 = px + sc * 4
            y0 = py + sr * 4
            ref = gather_4x4_reference(plane, x0, y0)
            sub_src = [
                [plane[y0 + r][x0 + c] for c in range(4)] for r in range(4)
            ]
            best_mode = I4_DC
            best_cost = None
            best_pred = None
            for m in range(9):
                bp = predict_4x4(m, ref["top"], ref["left"], ref["top_left"])
                res = [[sub_src[r][c] - bp[r][c] for c in range(4)] for r in range(4)]
                cost = _satd.satd4x4(res)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_mode = m
                    best_pred = bp
            sub_modes[sr][sc] = best_mode
            for r in range(4):
                for c in range(4):
                    pred4[sr * 4 + r][sc * 4 + c] = best_pred[r][c]
    cost4 = _satd.satd16x16(_residual(pred4))
    candidates.append(ModeCandidate("I4", -1, cost4, pred4, sub_modes))

    # ③ 合并按 satd 升序取前 num_candidates
    candidates.sort(key=lambda mc: mc.satd)
    return candidates[:num_candidates]
