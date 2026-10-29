# 设计决策记录

每条记录回答四个问题：面对什么选择，选了什么，为什么，放弃了什么并付出什么代价。
这里记的是"为什么"，"是什么"在 `architecture.md` 和 `programming_model.md`。
新决策追加在末尾，推翻旧决策时在旧条目标注"被 ADR-xxx 取代"，不要删除。

## ADR-001 可编程 descriptor 驱动，而不是每层一套硬连线数据通路

- 选择：所有卷积层和全连接层共用一个 `conv_engine`，层的形状、地址、量化参数来自主机写入的 descriptor 表。
- 理由：加速器的核心价值是"同一硬件跑不同网络"。硬连线每层一套通路和一块 RAM，换网络就要重写 RTL，
  本质是把网络画成电路。descriptor 驱动是工业 NPU 的最小公分母（指令或命令队列的简化形式），
  也是面试时区分"做过加速器"和"做过一个固定电路"的分界线。
- 放弃：硬连线方案更容易把每一层调到极致，但不可复用；真正的指令集和编译器则超出个人项目的可控范围。
- 代价：需要通用地址生成器和参数检查逻辑，这部分是 v1 最复杂的控制逻辑。

## ADR-002 INT8 整数推理，而不是 INT8 存储加 FP16 计算

- 选择：int8 x int8 乘，int32 累加，整数 requant 回 int8。全程没有浮点。
- 理由：这是 TFLite、TensorRT INT8 和绝大多数边缘 NPU 的做法。int8 乘法器面积远小于 FP16 乘加，
  累加没有舍入误差，结果可以和软件逐 bit 对上。FP16 路径需要反量化单元和 FP16 MAC，
  既贵又让"bit-exact"几乎不可能。
- 放弃：FP16 精度更高、对量化质量不敏感。对 LeNet 这种网络 int8 精度损失可以忽略。
- 代价：需要自己写量化导出脚本，并把量化算术定义到每一位。

## ADR-003 zero point 折叠进 bias，PE 只做 int8 x int8

- 选择：导出脚本计算 `bias - z_in * sum(w)`，硬件 PE 不处理 zero point；padding 填 `z_in`。
- 理由：PE 是阵列里复制 64 份的单元，省掉一个 9 位减法器和一条 zero point 广播线，乘法器保持 8x8。
  这是 gemmlowp / TFLite 的标准技巧，正确性只依赖"padding 值等于 z_in"这一条约定。
- 放弃：在 PE 输入端做 `q - z_in`，概念更直白，但每个 PE 多一个减法和一位宽度。
- 代价：地址生成器必须把越界 lane 替换成 `z_in` 而不是 0，这个约定要在 TB 里专门检查。

## ADR-004 output-stationary 数据流，而不是 weight-stationary

这是 v1 最重要的架构决策。

- 选择：每个 PE 固定负责一个输出元素，activation 沿行流动，权重沿列流动，部分和留在 PE 内。
- 理由，按重要性排序：
  1. 存储访问简单。output-stationary 每拍需要同一个 k 下 8 个相邻输出像素的输入，在 CHW 布局下
     就是 8 个连续字节，用 8 个字节 bank 就能做到任意起始地址、零冲突。
     weight-stationary 每拍需要同一个输出像素在 8 个不同 k 下的输入，这 8 个地址分散在不同行和不同通道，
     要么 8 读端口，要么复杂的 bank 规划，要么先做 im2col 缓冲。
  2. 不需要部分和存储。weight-stationary 的 K 维要分 tile，部分和必须写进一块 int32 SRAM 再读回累加；
     output-stationary 的 K 循环在 PE 内部走完，部分和从不离开 PE。
  3. 适合小 C_out。LeNet 的 C_out 只有 6 和 16，weight-stationary 让权重"驻留"在阵列里的收益
     来自大 M，而 K 分 tile 的开销和 drain 换权重的开销在小网络上不划算。
- 放弃：weight-stationary 是 TPU 的选择，适合 K 和 N 都很大、输入流已经是矩阵形式的场景。
  把它作为 M6 以后的对比实验是合理的，但不适合作为第一版。
- 代价：权重在每个 m_tile 都要重新流过阵列，总读取量是权重大小乘以 m_tile 数。
  片上 SRAM 每拍 8 字节带宽足够；如果权重在片外，这个数据流就要换。
  另一个代价是 FC 利用率只有 1/8，但这是权重带宽的上限，任何数据流在 batch 为 1 时都一样。

## ADR-005 CHW 布局与 8 字节 bank 的非对齐访问

- 选择：activation 用 CHW，1 字节一个元素；act_buffer 由 8 个字节 bank 组成，支持任意字节地址起始的 8 字节读写。
- 理由：ADR-004 要求"8 个相邻输出像素对应的输入"是连续地址，CHW 下同一行内相邻像素正是连续的。
  同时 FC 的输入向量就是上一层张量的自然展开，不需要重排。字节 bank 加地址旋转是教科书做法，
  比双字读取加移位更规整，也不浪费带宽。
- 放弃：NHWC 让通道连续，适合通道数是 8 的倍数的网络；LeNet 的 C_in 是 1、3、6，NHWC 会浪费大部分 lane。
- 代价：stride 2 卷积会让 8 个 lane 落到 4 个 bank，因此 v1 不支持 stride 2，降采样由 maxpool 完成。

## ADR-006 drain 用 shadow 寄存器加列 mux，而不是移位链

- 选择：PE 在 `last` 时把累加器拷入 shadow；阵列在最后一个 PE 完成后，用 8:1 mux 每拍读出一列。
- 理由：因为 skew，各 PE 完成时刻呈对角线波前。移位链的方向要么和波前同向（数据被后到的 PE 覆盖），
  要么反向（每两拍才能出一个值，drain 需要 16 拍）。shadow 加 mux 没有这两个问题，
  只要求 tile 周期不小于 8 拍，这个条件通过在 K_total 小于 8 时插空拍来保证。
- 放弃：移位链的布线更局部，在大阵列上更友好；8x8 规模下 mux 完全可以接受。
- 代价：每个 PE 多 32 个触发器，以及 8 行 x 8:1 x 32 位的 mux。

## ADR-007 主机接口用 AXI4-Lite，DMA 用 AXI4 master 并推迟到 M6

- 选择：寄存器和片上存储都通过一个 AXI4-Lite slave 访问；外部存储搬运留给 M6 的 AXI4 master。
- 理由：加速器作为 SoC 里的一个 IP，几乎总是挂在 AXI 互连上；AXI4-Lite 是配置口的事实标准，
  简历读者一眼就能定位。APB 用在这里也能工作，但它更多出现在低速外设，
  而且本项目的作者在工作中更常用 AXI。把 DMA 推迟是因为 v1 的片上存储放得下整个 LeNet，
  先用 AXI4-Lite 装载能让计算核更早被端到端验证。
- 放弃：APB 实现更小。如果将来需要，在 M6 的 `chip_top` 里加一个 APB 或 SPI 到 AXI4-Lite 的桥即可，
  计算核不受影响。
- 代价：AXI4-Lite 的 5 个通道比 APB 的状态机多一点代码；32 位逐字装载 67 KB 权重在仿真里需要约 1.7 万次写，
  TB 可以用后门加载缩短时间。

## ADR-008 descriptor 表放在寄存器文件里，而不是放在存储里由硬件取

- 选择：16 项 x 10 个有效字的触发器阵列，主机通过 AXI4-Lite 直接读写，sequencer 直接索引。
- 理由：v1 的网络最多 8 层，16 项足够；放寄存器省掉"从存储取 descriptor"的状态机和一个读端口的仲裁。
  16 x 10 x 32 约 5K 触发器，对这个规模可以接受。
- 放弃：descriptor 放在存储里、寄存器只给一个指针，是更可扩展的方式，适合几十层以上的网络。
  M6 加 DMA 之后如果要跑更深的网络，再改成"指针加预取"。
- 代价：层数上限写死在参数里。

## ADR-009 v1 单时钟域，SoC 胶水全部放到 chip_top

- 选择：`npu_top` 只有一个时钟、一个复位、一个 AXI4-Lite、一个中断。像素流 CDC、PLL、时钟门控、
  SPI 或 APB 配置桥、pad，都属于 M6 的 `chip_top`。
- 理由：计算核的正确性和数据流是这个项目的主体，也是作者需要补的部分；CDC、异步 FIFO、时钟控制
  是作者已有的技能，用一个薄包装层展示即可，不应占用前五个里程碑的时间。
- 放弃：一开始就做多时钟域能更早暴露集成问题，但会让每个单元 TB 都变复杂。
- 代价：M6 时要为 `npu_top` 的 AXI4-Lite 和中断做一次跨时钟域处理。

## ADR-010 Verilator 为主仿真器，VCS 为辅

- 选择：所有 testbench 限制在 Verilator 5 支持的子集；GitHub Actions 跑 lint、单元和集成仿真；
  VCS 结果作为附加记录写进 `STATUS.md`。
- 理由：仓库的目标读者是任何想复现的人，他们没有 VCS。CI 全绿本身就是作品的一部分。
  作者的 VCS 环境只在受限服务器上，不能依赖。
- 放弃：UVM 和 class-based testbench。它们在 Verilator 上支持不完整，而且对这个规模的项目收益有限。
- 代价：testbench 风格偏过程式；覆盖率工具链较弱。

## ADR-011 MNIST 和 CIFAR-10 都做，MNIST 先做

- 选择：M5 以 MNIST 为完成标准，CIFAR-10 作为同一份 RTL 的第二组 descriptor 和权重。
- 理由：两者在 LeNet-5 下只有 conv1 的 C_in 和 pad 不同，CIFAR-10 几乎零硬件成本，
  却能直接证明"可编程"不是口号。先做 MNIST 是因为它更小、更容易调试、量化误差更可控。
- 放弃：只做 MNIST 会让 RGB 三通道路径（C_in = 3）没有被真实网络覆盖。
- 代价：多一套训练和量化数据，以及 CIFAR-10 上 LeNet 本身只有六成左右的准确率，README 里要说明这是网络能力而非硬件问题。

## ADR-012 不支持 stride 2 和 per-channel 量化

- 选择：v1 的 descriptor 保留了 STRIDE_M1 字段但要求为 0；量化参数 per-tensor。
- 理由：LeNet 不需要 stride 2；per-channel 对 LeNet 的精度提升可以忽略，却要把 mult / shift 变成每通道查表。
  两者都在 `architecture.md` 第 10 节有明确的扩展路径，字段和地址已经预留。
- 放弃：更通用的算子支持。
- 代价：换一个使用 stride 2 卷积的网络之前必须先做扩展。

## ADR-013 分支与提交策略

- 选择：每个里程碑一个分支（`docs/architecture-v1`、`feat/m1-requant-pe` 等），通过 PR 合入 `main`；
  `main` 始终可 lint、可仿真。`docs/STATUS.md` 在每个 PR 里更新。
- 理由：PR 历史本身就是展示工程过程的材料；小 PR 便于自己回顾，也便于别人阅读。
- 放弃：长期开发分支一次性合并。
- 代价：需要维护分支命名和 PR 描述的纪律。
