"""YUV420P 帧的读写与合成。

YUV420P 平面顺序为 Y(w*h) U(w*h/4) V(w*h/4)，其中亮度平面为二维
height x width 列表，色度平面为 (height//2) x (width//2) 列表。
"""

import random
from dataclasses import dataclass


@dataclass
class YUV420PFrame:
    """一帧 YUV420P 图像。"""

    width: int
    height: int
    y: list  # height x width 的二维 int 列表
    u: list  # (height//2) x (width//2) 的二维 int 列表
    v: list  # (height//2) x (width//2) 的二维 int 列表


def validate_dimensions(width: int, height: int) -> None:
    """校验图像尺寸：必须为正整数且能被 16 整除。"""
    for name, value in (("width", width), ("height", height)):
        if type(value) is not int or value <= 0:
            raise ValueError("%s 必须为正整数，当前为 %r" % (name, value))
        if value % 16 != 0:
            raise ValueError("%s 必须能被 16 整除，当前为 %d" % (name, value))


def _new_plane(h: int, w: int, fill: int = 0) -> list:
    """创建 h x w 的二维列表平面。"""
    return [[fill] * w for _ in range(h)]


def read_yuv420p(path: str, width: int, height: int, frame_index: int = 0) -> YUV420PFrame:
    """从 YUV420P 文件读取指定帧。"""
    validate_dimensions(width, height)
    y_size = width * height
    c_size = (width // 2) * (height // 2)
    frame_size = y_size + 2 * c_size
    with open(path, "rb") as fp:
        fp.seek(frame_index * frame_size)
        data = fp.read(frame_size)
    if len(data) < frame_size:
        raise ValueError("文件数据不足，无法读取第 %d 帧" % frame_index)
    cw = width // 2
    ch = height // 2
    y = [[data[r * width + c] for c in range(width)] for r in range(height)]
    u_off = y_size
    v_off = y_size + c_size
    u = [[data[u_off + r * cw + c] for c in range(cw)] for r in range(ch)]
    v = [[data[v_off + r * cw + c] for c in range(cw)] for r in range(ch)]
    return YUV420PFrame(width, height, y, u, v)


def write_yuv420p(path: str, frame: YUV420PFrame) -> None:
    """将一帧 YUV420P 图像写入文件。"""
    validate_dimensions(frame.width, frame.height)
    buf = bytearray()
    for row in frame.y:
        buf.extend(row)
    for row in frame.u:
        buf.extend(row)
    for row in frame.v:
        buf.extend(row)
    with open(path, "wb") as fp:
        fp.write(buf)


def synthesize_frame(width: int, height: int, pattern: str = "gradient") -> YUV420PFrame:
    """合成一帧测试图像，支持 gradient / flat / random / edges 四种图案。"""
    validate_dimensions(width, height)
    cw = width // 2
    ch = height // 2
    y = _new_plane(height, width)
    u = _new_plane(ch, cw, 128)
    v = _new_plane(ch, cw, 128)

    if pattern == "gradient":
        # 平滑双线性渐变
        for r in range(height):
            for c in range(width):
                val = (c * 255 // (width - 1) + r * 255 // (height - 1)) // 2
                y[r][c] = val
    elif pattern == "flat":
        # 平坦区域，Y=U=V=128
        for r in range(height):
            for c in range(width):
                y[r][c] = 128
    elif pattern == "random":
        # 固定种子的随机噪声（保证可复现）
        rng = random.Random(1234)
        for r in range(height):
            for c in range(width):
                y[r][c] = rng.randrange(0, 256)
        for r in range(ch):
            for c in range(cw):
                u[r][c] = rng.randrange(0, 256)
                v[r][c] = rng.randrange(0, 256)
    elif pattern == "edges":
        # 含边缘纹理：棋盘式高低电平块
        for r in range(height):
            for c in range(width):
                y[r][c] = 235 if ((c // 4) + (r // 4)) % 2 == 0 else 16
    else:
        raise ValueError("未知图案类型: %r" % pattern)
    return YUV420PFrame(width, height, y, u, v)


def iter_macroblock_indices(width: int, height: int) -> list:
    """按光栅顺序返回所有宏块坐标 [(mb_x, mb_y), ...]。"""
    validate_dimensions(width, height)
    return [
        (mb_x, mb_y)
        for mb_y in range(height // 16)
        for mb_x in range(width // 16)
    ]


def extract_block(plane: list, x0: int, y0: int, size: int) -> list:
    """从平面中提取 size x size 的块，越界抛 IndexError。"""
    ph = len(plane)
    pw = len(plane[0]) if ph else 0
    if x0 < 0 or y0 < 0 or x0 + size > pw or y0 + size > ph:
        raise IndexError(
            "块越界: x0=%d y0=%d size=%d 平面=%dx%d" % (x0, y0, size, pw, ph)
        )
    return [[plane[y0 + r][x0 + c] for c in range(size)] for r in range(size)]


def clone_frame(frame: YUV420PFrame) -> YUV420PFrame:
    """深拷贝一帧图像。"""
    return YUV420PFrame(
        frame.width,
        frame.height,
        [row[:] for row in frame.y],
        [row[:] for row in frame.u],
        [row[:] for row in frame.v],
    )
