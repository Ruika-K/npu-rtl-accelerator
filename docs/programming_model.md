# NPU v1 编程模型

主机通过 AXI4-Lite 看到的一切：地址映射、寄存器、descriptor 格式、运行流程、错误码。
与 `docs/architecture.md` 配套。寄存器地址一旦被 RTL 和 Python 脚本共同引用，就只能追加，不能改动。

## 1. 地址映射

AXI4-Lite 地址只解码低 20 位，总窗口 1 MB。

| 范围 | 大小 | 内容 | 访问 |
| --- | --- | --- | --- |
| `0x00000 - 0x00FFF` | 4 KB | 寄存器与 descriptor 表 | 任何时候 |
| `0x10000 - 0x17FFF` | 32 KB | act_buffer（字节地址，32 位访问需 4 字节对齐） | 仅 NPU 空闲时 |
| `0x20000 - 0x3FFFF` | 128 KB | weight_buffer（字节地址，32 位访问需 4 字节对齐） | 仅 NPU 空闲时 |
| 其他 | - | 未映射 | 返回 SLVERR |

descriptor 中的 `in_base / out_base` 是 act_buffer 内部的字节偏移，`w_base / b_base` 是 weight_buffer 内部的字节偏移。
主机写入时加上窗口基址 `0x10000` 或 `0x20000`。

## 2. 寄存器

所有寄存器 32 位。类型：RO 只读，RW 读写，W1P 写 1 产生一拍脉冲（读回 0），W1C 写 1 清零。

| 偏移 | 名称 | 位 | 类型 | 复位值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `0x000` | ID | [31:0] | RO | `0x4E505501` | ASCII "NPU" + 版本 0x01 |
| `0x004` | CTRL | [0] START | W1P | 0 | 启动一次推理，busy 时写入被忽略 |
| | | [1] IRQ_EN | RW | 0 | 允许 `irq_o` |
| `0x008` | STATUS | [0] BUSY | RO | 0 | 正在运行 |
| | | [1] DONE | W1C | 0 | 一次推理完成 |
| | | [2] ERR | W1C | 0 | 出错停止，细节见 ERR_CODE 和 CUR_LAYER |
| | | [3] HOST_CONFLICT | W1C | 0 | busy 期间有主机访问片上存储被拒绝 |
| | | [11:8] CUR_LAYER | RO | 0 | 当前（或出错）的 descriptor 序号 |
| | | [23:16] ERR_CODE | RO | 0 | 见第 5 节 |
| `0x00C` | LAYER_CNT | [4:0] | RW | 0 | 要执行的 descriptor 个数，1 ~ 16 |
| `0x010` | RESULT | [3:0] ARGMAX | RO | 0 | 最近一次 ARGMAX 的结果 |
| | | [8] VALID | RO | 0 | START 时清零，ARGMAX 完成时置 1 |
| `0x020 - 0x05C` | LOGITS[0..15] | [31:0] | RO | 0 | ARGMAX 读到的 int8 值，符号扩展到 32 位 |
| `0x060` | PERF_TOTAL | [31:0] | RO | 0 | 从 START 到 DONE 的周期数 |
| `0x064` | PERF_ARRAY_BUSY | [31:0] | RO | 0 | 脉动阵列 `in_valid` 为 1 的周期数 |
| `0x080 - 0x0BC` | PERF_LAYER[0..15] | [31:0] | RO | 0 | 每个 descriptor 的周期数 |
| `0x100 - 0x4FF` | DESC[0..15] | 16 字 | RW | 0 | descriptor 表，每项 64 字节，见第 3 节 |

perf 寄存器在 START 时清零。未定义的偏移读返回 0，写被忽略（不报错，便于软件向前兼容）。

## 3. descriptor 格式

每个 descriptor 占 16 个 32 位字，偏移 `0x100 + 64 * idx`。

| 字 | 位 | 字段 | 说明 |
| --- | --- | --- | --- |
| W0 | [3:0] | OP | 0 NOP，1 CONV2D，2 MAXPOOL2X2，3 FC，4 ARGMAX |
| | [7:4] | K | 卷积核边长 1~7。FC 固定为 1 |
| | [11:8] | PAD | 四边相同的零 padding 0~7，填入值为 Z_IN |
| | [13:12] | STRIDE_M1 | stride 减 1，v1 必须为 0 |
| | [31:14] | 保留 | 写 0 |
| W1 | [15:0] | IN_H | 输入高度。FC 为 1 |
| | [31:16] | IN_W | 输入宽度。FC 为 1 |
| W2 | [15:0] | C_IN | 输入通道数。FC 为输入向量长度，ARGMAX 为 logits 个数 |
| | [31:16] | C_OUT | 输出通道数。FC 为输出向量长度 |
| W3 | [31:0] | IN_BASE | act_buffer 内字节偏移 |
| W4 | [31:0] | OUT_BASE | act_buffer 内字节偏移 |
| W5 | [31:0] | W_BASE | weight_buffer 内字节偏移，8 字节对齐 |
| W6 | [31:0] | B_BASE | weight_buffer 内字节偏移，4 字节对齐 |
| W7 | [31:0] | RQ_MULT | 正 int32，范围 `[2^30, 2^31)` |
| W8 | [5:0] | RQ_SHIFT | 1 ~ 62 |
| W9 | [7:0] | Z_IN | 输入 zero point，int8 |
| | [15:8] | Z_OUT | 输出 zero point，int8 |
| | [23:16] | ACT_MIN | 输出下限，int8。ReLU 时等于 Z_OUT |
| | [31:24] | ACT_MAX | 输出上限，int8 |
| W10 - W15 | | 保留 | 写 0 |

各 OP 使用的字段：

| OP | 使用 | 忽略 |
| --- | --- | --- |
| CONV2D | 全部 | STRIDE_M1 必须为 0 |
| FC | 同 CONV2D，硬件强制 K = 1, PAD = 0, IN_H = IN_W = 1 | |
| MAXPOOL2X2 | IN_H, IN_W, C_IN, IN_BASE, OUT_BASE | 其余 |
| ARGMAX | C_IN（<= 16）, IN_BASE | 其余 |

输出尺寸由硬件计算：`OUT = IN + 2*PAD - K + 1`（CONV2D），`OUT = IN / 2` 向下取整（MAXPOOL）。

## 4. 运行流程

1. 复位后读 ID，确认为 `0x4E505501`。
2. 把导出脚本生成的权重 blob 写入 `0x20000` 起的 weight_buffer。blob 内已经包含所有层的权重、bias 和对齐空隙，
   一次性顺序写入即可。
3. 写 DESC[0 .. L-1] 和 LAYER_CNT = L。
4. 把输入图像（int8，已减 128，CHW）写入 act_buffer 中 DESC[0].IN_BASE 指向的位置。
5. 可选：CTRL.IRQ_EN = 1。
6. CTRL.START = 1。
7. 等待 `irq_o` 或轮询 STATUS.DONE。读 RESULT 和 LOGITS，读 PERF_* 作为性能数据。
8. 写 1 清 STATUS.DONE。换下一张图只需重复第 4、6、7、8 步，权重与 descriptor 常驻。

## 5. 错误码

| ERR_CODE | 含义 |
| --- | --- |
| `0x00` | 无错误 |
| `0x01` | 非法 OP |
| `0x02` | 不支持的参数：K = 0 或 > 7，STRIDE_M1 != 0，C_IN 或 C_OUT = 0，K_total > 4096，ARGMAX C_IN > 16 |
| `0x03` | 地址越界或未对齐：输入、输出、权重、bias 区域超出 buffer，W_BASE 非 8 字节对齐，B_BASE 非 4 字节对齐 |
| `0x04` | LAYER_CNT = 0 或 > 16 |
| `0x05` | 输出尺寸 <= 0（K > IN + 2*PAD） |

出错时 sequencer 立即停止，BUSY 清零，ERR 置位，CUR_LAYER 指向出错的 descriptor。

## 6. 示例：LeNet-5 MNIST

act_buffer 布局（字节偏移）：

| 张量 | 偏移 | 大小 |
| --- | --- | --- |
| 输入 1x28x28 | `0x0000` | 784 |
| conv1 输出 6x28x28 | `0x0400` | 4704 |
| pool1 输出 6x14x14 | `0x1800` | 1176 |
| conv2 输出 16x10x10 | `0x1D00` | 1600 |
| pool2 输出 400 | `0x2400` | 400 |
| fc1 输出 120 | `0x2600` | 120 |
| fc2 输出 84 | `0x2700` | 84 |
| fc3 输出 10 | `0x2780` | 10 |

weight_buffer 布局（字节偏移，由导出脚本生成，这里只是示例）：

| 层 | 权重 `[K_total][N_pad8]` | 偏移 | bias | 偏移 |
| --- | --- | --- | --- | --- |
| conv1 | 25 x 8 = 200 B | `0x00000` | 8 x 4 B | `0x00100` |
| conv2 | 150 x 16 = 2400 B | `0x00200` | 16 x 4 B | `0x00C00` |
| fc1 | 400 x 120 = 48000 B | `0x01000` | 120 x 4 B | `0x0CC00` |
| fc2 | 120 x 88 = 10560 B | `0x0D000` | 88 x 4 B | `0x0FA00` |
| fc3 | 84 x 16 = 1344 B | `0x10000` | 16 x 4 B | `0x10600` |

descriptor 表（LAYER_CNT = 8）：

| idx | OP | K | PAD | IN_H x IN_W | C_IN -> C_OUT | IN_BASE -> OUT_BASE | W_BASE / B_BASE | ReLU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | CONV2D | 5 | 2 | 28 x 28 | 1 -> 6 | `0x0000 -> 0x0400` | `0x00000 / 0x00100` | 是 |
| 1 | MAXPOOL | - | - | 28 x 28 | 6 | `0x0400 -> 0x1800` | - | - |
| 2 | CONV2D | 5 | 0 | 14 x 14 | 6 -> 16 | `0x1800 -> 0x1D00` | `0x00200 / 0x00C00` | 是 |
| 3 | MAXPOOL | - | - | 10 x 10 | 16 | `0x1D00 -> 0x2400` | - | - |
| 4 | FC | 1 | 0 | 1 x 1 | 400 -> 120 | `0x2400 -> 0x2600` | `0x01000 / 0x0CC00` | 是 |
| 5 | FC | 1 | 0 | 1 x 1 | 120 -> 84 | `0x2600 -> 0x2700` | `0x0D000 / 0x0FA00` | 是 |
| 6 | FC | 1 | 0 | 1 x 1 | 84 -> 10 | `0x2700 -> 0x2780` | `0x10000 / 0x10600` | 否 |
| 7 | ARGMAX | - | - | - | 10 | `0x2780` | - | - |

CIFAR-10 版本只有两处不同：DESC[0] 变为 IN 32 x 32、C_IN 3、PAD 0，conv1 权重变为 75 x 8 字节；
其余 descriptor 的形状完全相同，只是 RQ_MULT / RQ_SHIFT / Z_* 和权重内容不同。
这就是"同一份 RTL 跑两个网络"的具体含义。
