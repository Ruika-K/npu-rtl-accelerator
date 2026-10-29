# NPU v1 架构说明

版本：v1 草案（2026-10-29）。本文档先于 RTL 存在：任何接口变更先改这里，再改代码。
与之配套的文档：`docs/programming_model.md`（寄存器、descriptor、主机流程）、
`docs/verification.md`（验证计划）、`docs/decisions.md`（每个关键决策的理由与被否掉的替代方案）。

## 0. 一句话描述

一个由 layer descriptor 驱动的 INT8 CNN 推理加速器：8x8 output-stationary 脉动阵列，
片上 activation / weight SRAM，AXI4-Lite 主机接口，整数算术与 Python golden model 逐 bit 一致。
同一份 RTL 通过换 descriptor 和权重即可运行 LeNet-5 的 MNIST 版本和 CIFAR-10 版本。

## 1. 范围

### 1.1 目标

| 项目 | v1 决定 |
| --- | --- |
| 目标网络 | LeNet-5 类小型 CNN。MNIST（28x28x1）和 CIFAR-10（32x32x3）各一套 descriptor 和权重 |
| 算子 | CONV2D（K 为 1~7，stride 1，pad 0~7，融合 bias / requant / clamp）、MAXPOOL 2x2 stride 2、FC（作为 CONV2D 的 1x1 特例）、ARGMAX |
| 数值 | int8 激活（非对称，带 zero point）、int8 权重（对称，per-tensor）、int32 累加、整数 requant |
| 计算核 | 8x8 output-stationary 脉动阵列，参数 `ARR_N` 可改为 4 或 16 |
| 片上存储 | activation buffer 32 KB，weight buffer 128 KB |
| 主机接口 | AXI4-Lite slave，32 位数据，覆盖寄存器和两块片上存储 |
| 验证 | numpy 整数 golden model 逐层 bit-exact；Verilator 本地与 GitHub Actions 可复现；VCS 作为额外验证 |

### 1.2 非目标（v1 明确不做）

- 训练、浮点运算、动态 shape、编译器、多核、DRAM 控制器。
- stride 2 卷积（降采样由 maxpool 完成）、per-channel 量化、depthwise 卷积、batch 大于 1。
- PLL、SPI、多时钟域、pad：留给 M6 的 `chip_top` 包装层。
- 外部存储 DMA（AXI4 master）：M6。v1 由主机通过 AXI4-Lite 把权重和图像写进片上存储。

以上每一项在第 10 节都有对应的扩展路线，v1 的接口为它们预留了位置。

## 2. 顶层结构

```text
                      AXI4-Lite slave (s_axil_*)                 irq_o
                               |                                   ^
  +----------------------------v-----------------------------------+-----------+
  | npu_top                                                                    |
  |                                                                            |
  |  axil_slave ---> npu_regs ------------------> npu_sequencer                |
  |  (协议转换)      ctrl/status/result            逐层取 descriptor，           |
  |                  descriptor 表 x16             启动对应 engine，等待 done    |
  |                  perf counters                        |                    |
  |                      |                 +--------------+--------------+     |
  |         host 访问片上存储               |              |              |     |
  |                      |            conv_engine    pool_engine   argmax_engine|
  |                      |            +----------+                             |
  |                      |            |conv_addr_gen                           |
  |                      |            |systolic_array 8x8                      |
  |                      |            |requant_unit x8                         |
  |                      |            |act_writer |                            |
  |                      v            +----------+                             |
  |     +----------- act_buffer (32 KB, 8 byte-bank, 1R1W, 非对齐 8 字节访问)   |
  |     +----------- weight_buffer (128 KB, 64 位字, 字节使能)                  |
  +----------------------------------------------------------------------------+
```

模块与文件对应关系（一个文件一个主模块，文件名等于模块名）：

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| `npu_top` | `rtl/npu_top.sv` | 顶层连线，对外只有 clk / rst / AXI4-Lite / irq |
| `npu_pkg` | `rtl/npu_pkg.sv` | 参数、opcode、descriptor 结构体、地址映射常量 |
| `axil_slave` | `rtl/host/axil_slave.sv` | AXI4-Lite 协议转成内部单周期 req/ack 总线 |
| `npu_regs` | `rtl/host/npu_regs.sv` | 寄存器、descriptor 表、perf counter、片上存储主机访问路由 |
| `npu_sequencer` | `rtl/ctrl/npu_sequencer.sv` | 层循环 FSM，错误检查，逐层计时 |
| `conv_engine` | `rtl/compute/conv_engine.sv` | CONV2D / FC 的 tile 循环、地址流、drain、写回 |
| `conv_addr_gen` | `rtl/compute/conv_addr_gen.sv` | 把 (n_tile, oh, ow0, k) 展开成读地址、lane mask、控制标志 |
| `systolic_array` | `rtl/compute/systolic_array.sv` | ARR_N x ARR_N PE 阵列，边缘 skew，shadow 累加器与列 drain |
| `pe` | `rtl/compute/pe.sv` | int8 x int8 乘加，int32 累加，shadow 寄存器 |
| `requant_unit` | `rtl/compute/requant_unit.sv` | bias、定点乘、舍入移位、加 zero point、clamp |
| `pool_engine` | `rtl/compute/pool_engine.sv` | 2x2 stride 2 maxpool |
| `argmax_engine` | `rtl/compute/argmax_engine.sv` | 读取 logits，输出最大值索引 |
| `act_buffer` | `rtl/mem/act_buffer.sv` | 8 个字节 bank 组成的 activation SRAM，支持任意字节地址起始的 8 字节读写 |
| `weight_buffer` | `rtl/mem/weight_buffer.sv` | 64 位宽权重 SRAM，带字节使能 |
| `sram_1r1w` | `rtl/mem/sram_1r1w.sv` | 行为级 1 读 1 写 SRAM，将来替换为工艺宏的唯一边界 |

## 3. 数值格式

### 3.1 量化方案

采用与 TFLite int8 推理相同结构的仿射量化：

```text
real = scale * (q - zero_point)
```

| 张量 | 类型 | zero point | scale 粒度 |
| --- | --- | --- | --- |
| 输入图像 | int8 | -128（像素 0~255 减 128） | per-tensor，scale = 1/255 |
| 各层激活 | int8 | 任意，由校准决定 | per-tensor |
| 权重 | int8 | 0（对称） | per-tensor |
| bias | int32 | 0 | scale = s_in * s_w |
| 累加器 | int32 | - | - |

累加器安全性：最长的点积是 FC1，长度 400，单项最大 128 * 127，总和约 6.5e6，远小于 2^31。
v1 对 K_total 做上限检查（见 `programming_model.md` 的错误码），超过 4096 报错。

### 3.2 zero point 折叠

卷积的真实值计算为 `sum_k (q_in[k] - z_in) * q_w[k]`。
为了让 PE 只做 int8 x int8，zero point 的修正由导出脚本折进 bias：

```text
bias_folded[n] = bias_q[n] - z_in * sum_k q_w[k, n]
acc[n]         = sum_k q_in[k] * q_w[k, n]          (硬件只做这一步)
```

成立的前提是 padding 位置的输入值必须是 `z_in` 而不是 0。`conv_addr_gen` 对越界 lane 填 `z_in`。

### 3.3 Requant

每个输出元素做一次：

```text
sum     = acc + bias_folded[n]                        // 33 位有符号
prod    = sum * rq_mult                               // rq_mult 为正 int32，范围 [2^30, 2^31)
rounded = (prod + (1 << (rq_shift - 1))) >>> rq_shift // 算术右移，round half up
q_out   = clamp(rounded + z_out, act_min, act_max)    // 结果为 int8
```

- `rq_mult / 2^rq_shift` 近似实数乘数 `s_in * s_w / s_out`，由导出脚本归一化得到。`rq_shift` 范围 1~62。
- ReLU 融合为 `act_min = z_out`（真实值 0 对应的量化值）。不做 ReLU 时 `act_min = -128, act_max = 127`。
- Python golden model 用 Python 大整数实现完全相同的四步，这是 bit-exact 的依据。

### 3.4 Maxpool 与 argmax

- 量化是单调映射，maxpool 直接比较 int8 值，输出沿用输入的 scale 和 zero point，不做 requant。
- argmax 对 FC3 的 10 个 int8 logits 做有符号比较，相等时取索引最小者。

## 4. 张量布局与片上存储

### 4.1 布局

- 所有 activation 为 CHW 布局（channel 优先，行其次，列最后），元素为 1 字节，地址 `base + c*H*W + h*W + w`。
- FC 的输入向量就是上一层 CHW 张量的自然展开，与 PyTorch 对 NCHW 做 `view(-1)` 的顺序一致，不需要重排。
- 权重按 GEMM 视角存放：`W[K_total][N_pad8]`，其中
  `K_total = C_in * K * K`，k 的展开顺序为 `(ci, kh, kw)`，kw 最内；
  `N_pad8` 是 C_out 向上取整到 8 的倍数，多出的列填 0。
  这样一行 `W[k][n0 .. n0+7]` 正好是一个对齐的 64 位字。
- bias 为 int32 数组 `bias[N_pad8]`，放在 weight buffer 中，由 descriptor 的 `b_base` 指向。

### 4.2 act_buffer

- 容量 32 KB，由 8 个 4 KB x 8 位的 1R1W SRAM bank 组成，bank 号为字节地址的低 3 位。
- 读端口：任意字节地址 A 起始的 8 个连续字节，一拍发出，固定延迟 2 拍返回。内部做法：
  bank b 读索引 `(A + ((b - A[2:0]) mod 8)) >> 3`，输出按 `A[2:0]` 旋转回自然顺序。
- 写端口：同样的非对齐 8 字节写，带 8 位字节使能。
- 两个端口来源：engine（conv / pool / argmax 共用一套读写口）和主机。NPU busy 时主机访问被拒绝并返回 SLVERR。
- 为什么是 8 个字节 bank：8 个连续地址一定落在 8 个不同 bank，所以 stride 1 的卷积永远没有 bank 冲突。这是选择 output-stationary 数据流的直接结果，见 `decisions.md`。

### 4.3 weight_buffer

- 容量 128 KB，64 位宽，16K 个字，单读端口加单写端口，带字节使能（主机 32 位写只用一半）。
- 读端口给 conv_engine 的权重流，每拍一个字（8 个 int8，对应一个 n_tile 的 8 列）。
- bias 也从这里读。为避免和权重流争抢读口，每个 n_tile 开始前预取 8 个 bias 到寄存器，占 4 拍。

### 4.4 容量核算（LeNet-5 MNIST，int8）

| 张量 | 大小 | 位置 |
| --- | --- | --- |
| 输入 28x28x1 | 784 B | act_buffer |
| conv1 输出 6x28x28 | 4704 B | act_buffer |
| pool1 输出 6x14x14 | 1176 B | act_buffer |
| conv2 输出 16x10x10 | 1600 B | act_buffer |
| pool2 / fc1 / fc2 / fc3 输出 | 400 / 120 / 84 / 10 B | act_buffer |
| 全部权重（N 对齐到 8） | 约 62.5 KB | weight_buffer |
| 全部 bias | 约 1 KB | weight_buffer |

activation 总计不到 9 KB，可以全部常驻、互不覆盖，省去 ping-pong 管理；权重加对齐空隙约 67 KB。

## 5. 计算核：8x8 output-stationary 脉动阵列

### 5.1 GEMM 视角

一层 CONV2D 等价于 `Out[M][N] = In[M][K_total] x W[K_total][N]`，其中
`M = H_out * W_out`（输出像素数），`N = C_out`，`K_total = C_in * K * K`。
`In[m][k]` 不在存储里实体化，由 `conv_addr_gen` 按需生成地址（im2col by address generation）。
FC 是 `H = W = 1, K = 1` 的特例，`K_total = C_in`。

### 5.2 PE 与阵列

- PE(i, j) 固定负责一个输出元素 `Out[m0 + i][n0 + j]`，累加器 int32 留在 PE 内部（output-stationary）。
- 每拍：activation 沿行从左向右流（`a` 经过每个 PE 延迟一拍），权重沿列从上向下流（`w` 同样延迟一拍）。
  边缘对 `a[i]` 做 i 拍 skew，对 `w[j]` 做 j 拍 skew，于是 `In[m0+i][k]` 与 `W[k][n0+j]` 在 PE(i, j) 相遇。
- 随 `a` 流动的控制标志：`valid`（本拍有数据）、`first`（k = 0，累加器清零后载入）、`last`（k = K_total - 1，把累加器拷入 shadow 寄存器）。
- 一个 tile 定义为固定 (n_tile, oh, ow0) 下 k 从 0 到 K_total - 1 的 K_total 拍。tile 之间背靠背，没有空拍。

### 5.3 Drain

- `last` 到达 PE(i, j) 的时刻是 `tile_start + K_total - 1 + i + j`。shadow 寄存器只在这一拍被写，之后保持 K_total 拍不变。
- 列 drain：当 `last` 到达 PE(ARR_N-1, 0) 后，从下一拍起每拍输出一整列 `j = 0, 1, ..., ARR_N-1`（8 个 int32，对应 8 个 m），用 8:1 mux 直接读 shadow，不做移位链。
- 不冲突条件：tile 周期 `P >= ARR_N`。所以 `P = max(K_total, ARR_N)`；K_total 小于 8 时由 `conv_addr_gen` 插空拍。
- 列 drain 的 8 个值属于同一个输出通道 `n = n0 + j`，在 CHW 布局里是 8 个连续字节，正好一次 `act_buffer` 非对齐写。
  这就是为什么 drain 按列而不是按行。

### 5.4 conv_engine 的循环

```text
for n_tile in 0 .. ceil(C_out / 8) - 1:
    prefetch bias[n_tile*8 .. +7] 到寄存器（4 拍）
    for oh in 0 .. H_out - 1:
        for ow0 in 0, 8, 16, ... < W_out:            # m_tile 不跨输出行
            for ci, kh, kw (即 k = 0 .. K_total - 1):
                ih    = oh - pad + kh
                A     = in_base + ci*H*W + ih*W + (ow0 - pad + kw)
                lane i 的输入 = (ih 越界 或 ow0+i-pad+kw 越界 或 ow0+i >= W_out) ? z_in : act[A + i]
                权重字  = w_base_word + k * (N_pad8 / 8) + n_tile
            drain 8 列 -> requant x8 -> 写 out_base + n*H_out*W_out + oh*W_out + ow0，lane mask 为 ow0+i < W_out
```

- 读地址与控制标志经过与 SRAM 读延迟匹配的流水，再进入阵列。
- 写回与下一 tile 的计算重叠，因为 drain 只占 8 拍，而 `P >= 8`。
- 列 `n >= C_out`（补齐列）的 drain 结果直接丢弃。

### 5.5 吞吐估算（LeNet-5 MNIST，单位：拍）

| 层 | n_tile | oh | ow_tile | K_total | 周期 | 有效 MAC |
| --- | --- | --- | --- | --- | --- | --- |
| conv1 5x5, 1->6, pad 2 | 1 | 28 | 4 | 25 | 2800 | 117.6K |
| pool1 | - | - | - | - | 约 700 | - |
| conv2 5x5, 6->16 | 2 | 10 | 2 | 150 | 6000 | 240K |
| pool2 | - | - | - | - | 约 350 | - |
| fc1 400->120 | 15 | 1 | 1 | 400 | 6000 | 48K |
| fc2 120->84 | 11 | 1 | 1 | 120 | 1320 | 10.1K |
| fc3 84->10 | 2 | 1 | 1 | 84 | 168 | 0.84K |
| 合计 | | | | | 约 17.5K | 416.5K |

峰值 64 MAC/拍，理想 6.5K 拍，预计利用率约 37%。三处损失都是结构性的，也是面试时值得讲清楚的：
C_out = 6 只用 8 列中的 6 列；W_out = 28 的最后一个 m_tile 只用 4 个 lane；
FC 只有一个输出像素，8 行只用 1 行，受权重带宽限制（每拍 8 字节权重最多支撑 8 个 MAC）。
perf counter 记录每层周期数和阵列有效拍数，用来在 README 里报告实际值。

## 6. pool_engine 与 argmax_engine

- `pool_engine`：对每个通道、每个输出行，读输入行 2r 和 2r+1 各 16 字节（两次 8 字节读），
  产生 8 个输出，一次 8 字节写。输出尺寸向下取整，与 PyTorch 默认一致。
- `argmax_engine`：从 `in_base` 读 `c_in` 个 int8（v1 限制 c_in <= 16），有符号比较，
  输出索引到 RESULT 寄存器，并把前 16 个 logits 存入 LOGITS 寄存器供调试。

## 7. 模块接口

命名约定：输入 `_i`，输出 `_o`，低有效复位 `rst_ni`。多比特向量使用 packed 数组。

### 7.1 npu_top

| 端口 | 方向 | 说明 |
| --- | --- | --- |
| `clk_i`, `rst_ni` | in | 单时钟域，异步低有效复位（顶层外部已同步） |
| `s_axil_aw*_i/o`, `s_axil_w*`, `s_axil_b*`, `s_axil_ar*`, `s_axil_r*` | - | AXI4-Lite slave，ADDR 32 位（内部使用低 20 位），DATA 32 位 |
| `irq_o` | out | 电平中断，`STATUS.DONE & CTRL.IRQ_EN` |

### 7.2 axil_slave 到 npu_regs 的内部总线

| 信号 | 说明 |
| --- | --- |
| `wr_req_o`, `wr_addr_o[19:0]`, `wr_data_o[31:0]`, `wr_strb_o[3:0]` | 单拍写请求 |
| `wr_ack_i`, `wr_err_i` | 下一拍起应答，err 映射为 SLVERR |
| `rd_req_o`, `rd_addr_o[19:0]` | 单拍读请求 |
| `rd_ack_i`, `rd_data_i[31:0]`, `rd_err_i` | 读应答 |

AXI4-Lite 侧一次只处理一个事务，不做 outstanding，先写后读的顺序由 AXI 自身保证。

### 7.3 npu_regs 到 npu_sequencer

| 信号 | 说明 |
| --- | --- |
| `start_i`（脉冲）、`layer_cnt_i[4:0]` | 由 CTRL.START 产生 |
| `desc_i[NUM_DESC]`（`desc_t` 数组） | descriptor 表的只读视图 |
| `busy_o`, `done_o`（脉冲）, `err_o`, `err_code_o[7:0]`, `cur_layer_o[3:0]` | 状态 |
| `layer_cycles_o[NUM_DESC][31:0]`, `total_cycles_o[31:0]`, `array_busy_cycles_o[31:0]` | perf |

### 7.4 npu_sequencer 到各 engine

每个 engine 一组：`start_i`（脉冲）、`desc_i`（当前层的 `desc_t`）、`busy_o`、`done_o`（脉冲）。
engine 内部参数非法时在 `start` 后一拍拉 `err_o` 并给出 `err_code_o`。

### 7.5 engine 到 act_buffer

| 信号 | 说明 |
| --- | --- |
| `rd_en_i`, `rd_addr_i[14:0]` | 字节地址，任意对齐 |
| `rd_data_o[63:0]`, `rd_valid_o` | 固定 2 拍延迟，lane i 在 `[8*i +: 8]` |
| `wr_en_i`, `wr_addr_i[14:0]`, `wr_data_i[63:0]`, `wr_be_i[7:0]` | 单拍写 |

### 7.6 engine 到 weight_buffer

| 信号 | 说明 |
| --- | --- |
| `rd_en_i`, `rd_word_addr_i[13:0]` | 64 位字地址 |
| `rd_data_o[63:0]`, `rd_valid_o` | 固定 1 拍延迟 |

### 7.7 systolic_array

| 端口 | 说明 |
| --- | --- |
| `in_valid_i`, `in_first_i`, `in_last_i` | 本拍输入向量的 k 位置标志 |
| `act_i[ARR_N][7:0]`, `wgt_i[ARR_N][7:0]` | 未 skew 的输入，skew 在模块内部完成 |
| `drain_valid_o`, `drain_col_idx_o[$clog2(ARR_N)-1:0]`, `drain_data_o[ARR_N][31:0]` | 一拍一列 |

### 7.8 requant_unit

输入 `acc_i[31:0]`、`bias_i[31:0]`、`mult_i[31:0]`、`shift_i[5:0]`、`z_out_i[7:0]`、`act_min_i[7:0]`、`act_max_i[7:0]`、`valid_i`；
输出 `q_o[7:0]`、`valid_o`。固定 3 拍流水，纯组合逻辑按 3.3 节公式切分。

## 8. 时钟、复位、中断、错误

- v1 单时钟域。`rst_ni` 在 `npu_top` 之外完成同步，模块内部全部用异步复位、同步释放的 `always_ff`。
- 中断为电平信号，软件写 1 清除 `STATUS.DONE` 后撤销。
- 错误是致命的：sequencer 停止，`STATUS.ERR` 置位，`ERR_CODE` 给出原因，`CUR_LAYER` 指出出错层；需要软件写 1 清除后重新 START。
- 主机在 busy 时访问片上存储：返回 SLVERR，操作不生效，同时置 `STATUS.HOST_CONFLICT`（非致命，只用于调试）。

## 9. 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `ARR_N` | 8 | 阵列边长，也是 act_buffer 的 bank 数，必须是 2 的幂 |
| `ABUF_BYTES` | 32768 | activation buffer 容量 |
| `WBUF_BYTES` | 131072 | weight buffer 容量 |
| `NUM_DESC` | 16 | descriptor 表项数 |
| `ABUF_RD_LAT` | 2 | act_buffer 读延迟 |
| `WBUF_RD_LAT` | 1 | weight_buffer 读延迟 |

## 10. 已知限制与扩展路线

| 限制 | 原因 | 扩展方式 |
| --- | --- | --- |
| stride 只能为 1 | 8 个 lane 跨 2 步会落到 4 个 bank，产生冲突 | 两拍读一次，或把 bank 数加倍 |
| per-tensor 量化 | 减少 descriptor 字段和 requant 查表 | 把 mult / shift 也按通道存进 weight buffer，与 bias 一起预取 |
| FC 利用率 1/8 | 单输出像素只占一行，受权重带宽限制 | batch 多张图，或把 K 拆到 8 行再做树形求和 |
| 无 DMA | v1 聚焦计算核 | M6 增加 AXI4 master，把权重和图像从外部存储搬入两块 buffer |
| 单时钟域 | 聚焦计算核 | M6 的 `chip_top` 加入像素流异步 FIFO、APB 或 SPI 配置桥、时钟门控 |
| 权重每个 m_tile 重读 | output-stationary 的固有代价 | 对本项目的片上 SRAM 带宽足够；如果换成外部存储，要加权重 cache 或改 weight-stationary |
