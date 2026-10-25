"""编码器配置与配置寄存器映射。

本模块对齐技术文档 5.3 节的配置寄存器接口，定义配置寄存器地址以及配置
对象 EncoderConfig，支持寄存器 <-> 配置对象的双向映射。
"""

from dataclasses import dataclass

# 配置寄存器地址（与芯片配置寄存器接口一致）
REG_IMAGE_WIDTH = 0x00    # 图像宽度，单位像素，需 16 对齐
REG_IMAGE_HEIGHT = 0x01   # 图像高度，单位像素，需 16 对齐
REG_BASE_QP = 0x02        # 基准 Qp，0~51
REG_AQ_ENABLE = 0x03      # AQ 自适应量化使能
REG_PSYRD_ENABLE = 0x04   # Psy-RD 人眼加权使能
REG_DEBLOCK_ENABLE = 0x05  # 去块滤波使能
REG_PREDICT_MODE = 0x06   # 预测模式配置，固定为 1（16x16 + 4x4 两级预测）


@dataclass
class EncoderConfig:
    """编码器全局配置。"""

    width: int = 1920
    # 1080P 原始高度 1080 无法 16 对齐，取 16 对齐的 1088（多出的 8 行做边界填充）
    height: int = 1088
    base_qp: int = 28
    aq_enable: bool = True
    psy_rd_enable: bool = True
    deblock_enable: bool = True
    predict_mode: int = 1
    crf_enable: bool = True
    psy_rd_strength: float = 1.0

    def validate(self) -> None:
        """校验配置合法性，非法时抛出 ValueError。"""
        # 宽高必须为正整数且能被 16 整除（宏块对齐）
        for name, value in (("width", self.width), ("height", self.height)):
            if type(value) is not int or value <= 0:
                raise ValueError("%s 必须为正整数，当前为 %r" % (name, value))
            if value % 16 != 0:
                raise ValueError("%s 必须能被 16 整除，当前为 %d" % (name, value))
        # 基准 Qp 范围 0~51
        if not (0 <= self.base_qp <= 51):
            raise ValueError("base_qp 必须在 0..51 之间，当前为 %d" % self.base_qp)
        # 预测模式固定为 1
        if self.predict_mode != 1:
            raise ValueError("predict_mode 必须为 1，当前为 %d" % self.predict_mode)

    @classmethod
    def from_registers(cls, regs: dict) -> "EncoderConfig":
        """由寄存器字典构造配置对象（缺失字段取默认值）。"""
        d = cls()
        return cls(
            width=int(regs.get(REG_IMAGE_WIDTH, d.width)),
            height=int(regs.get(REG_IMAGE_HEIGHT, d.height)),
            base_qp=int(regs.get(REG_BASE_QP, d.base_qp)),
            aq_enable=bool(regs.get(REG_AQ_ENABLE, d.aq_enable)),
            psy_rd_enable=bool(regs.get(REG_PSYRD_ENABLE, d.psy_rd_enable)),
            deblock_enable=bool(regs.get(REG_DEBLOCK_ENABLE, d.deblock_enable)),
            predict_mode=int(regs.get(REG_PREDICT_MODE, d.predict_mode)),
        )

    def to_registers(self) -> dict:
        """导出为寄存器字典（地址 -> 值）。"""
        return {
            REG_IMAGE_WIDTH: int(self.width),
            REG_IMAGE_HEIGHT: int(self.height),
            REG_BASE_QP: int(self.base_qp),
            REG_AQ_ENABLE: 1 if self.aq_enable else 0,
            REG_PSYRD_ENABLE: 1 if self.psy_rd_enable else 0,
            REG_DEBLOCK_ENABLE: 1 if self.deblock_enable else 0,
            REG_PREDICT_MODE: int(self.predict_mode),
        }
