"""Address map and register offsets of npu_top (docs/programming_model.md)."""
import npu_ref as ref

REG_BASE, ABUF_BASE, WBUF_BASE = 0x00000, 0x10000, 0x20000
REGS_BYTES = 0x1000

ID, CTRL, STATUS, LAYER_CNT, RESULT = 0x000, 0x004, 0x008, 0x00C, 0x010
LOGITS = 0x020                      # 16 words
PERF_TOTAL, PERF_ARRAY_BUSY = 0x060, 0x064
PERF_LAYER = 0x080                  # 16 words
DMA_CTRL, DMA_SRC, DMA_DST, DMA_LEN, DMA_STATUS = 0x0C0, 0x0C4, 0x0C8, 0x0CC, 0x0D0
DESC = 0x100                        # 16 entries x 16 words

NPU_ID = 0x4E505501
NUM_DESC, DESC_WORDS, DESC_USED_WORDS = 16, 16, 10
DESC_WORD_MASK = {0: 0x0000_3FFF, 8: 0x0000_003F}   # other stored words: all 32 bits

OKAY, SLVERR = 0b00, 0b10


def window(addr):
    """The address window an access falls in, whatever its alignment."""
    if addr < REGS_BYTES:
        return "reg"
    if ABUF_BASE <= addr < ABUF_BASE + ref.ABUF_BYTES:
        return "abuf"
    if WBUF_BASE <= addr < WBUF_BASE + ref.WBUF_BYTES:
        return "wbuf"
    return "none"


def desc_addr(idx, word):
    return DESC + 64 * idx + 4 * word
