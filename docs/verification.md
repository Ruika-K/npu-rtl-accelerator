# NPU v1 验证计划

目标只有一句：RTL 的每一层输出与 Python 整数 golden model 逐 bit 一致，并且任何人 clone 仓库后一条命令就能复现。

## 1. 参考模型与数据

目录 `model/`（M1 时创建）：

| 文件 | 作用 | 依赖 |
| --- | --- | --- |
| `train_lenet.py` | PyTorch 训练 fp32 LeNet-5，MNIST 和 CIFAR-10 各一个 checkpoint | torch, torchvision。可选，不运行也能跑仿真 |
| `quantize.py` | 后训练量化：权重 per-tensor 对称，激活用校准集的 min/max 确定 scale 和 zero point；计算 RQ_MULT / RQ_SHIFT；折叠 zero point 进 bias；输出权重 blob、descriptor 表、量化参数 JSON | numpy，读取 checkpoint |
| `npu_ref.py` | numpy 整数 golden model。只用整数运算实现 `architecture.md` 第 3 节的全部公式，可逐层 dump | numpy |
| `gen_vectors.py` | 选取 N 张测试图，生成输入 hex、每层期望输出 hex、期望 argmax | numpy |
| `data/<net>/weights.hex`, `desc.json`, `vectors/` | 生成结果，提交进仓库，使仿真不依赖 PyTorch | - |

规则：

- `npu_ref.py` 是唯一的真值来源。PyTorch 只负责产生权重，之后所有期望值都来自整数模型。
- 提交的数据文件由固定随机种子生成，脚本可以重新生成并 diff 验证。
- 数据文件总量控制在 1 MB 以内。

## 2. 分层验证

### 2.1 单元级（每个模块一个自检 testbench，`tb/unit/<module>_tb.sv`）

| 模块 | 激励 | 检查 |
| --- | --- | --- |
| `requant_unit` | 随机 acc / bias / mult / shift / zero point，加上饱和、舍入恰好在 0.5、shift 极值等定向用例 | 与 testbench 内置的 SV 整数参考函数逐个比对；参考函数再用 Python 交叉验证一次 |
| `pe` | 随机 int8 序列，first / last 标志 | 累加值、shadow 时序 |
| `systolic_array` | 随机 `In[M][K]`、`W[K][8]`，K 从 1 到 512，M 为 8 的倍数和非倍数 | drain 列数据等于 numpy 矩阵乘；K < 8 时空拍正确 |
| `act_buffer` | 随机非对齐 8 字节读写，字节使能，读写同拍不同地址 | 与 SV 中的字节数组模型比对 |
| `conv_addr_gen` | 随机 H / W / C_in / C_out / K / pad | 地址流、lane mask、first / last 与 Python 生成的序列逐拍一致 |
| `axil_slave` + `npu_regs` | AXI4-Lite 读写、W1P / W1C 语义、busy 期间访问存储 | 响应码、寄存器值、HOST_CONFLICT |
| `npu_sequencer` | 伪 engine（立即 done 或延迟 done）、非法 descriptor | 层顺序、perf 计数、错误码和 CUR_LAYER |
| `pool_engine`, `argmax_engine` | 随机张量 | 与 numpy 结果一致 |

### 2.2 集成级（`tb/integration/`）

1. 单层 CONV2D：随机形状和权重，通过 AXI4-Lite 装载，运行，回读输出张量与 golden model 比对。
2. 单层 FC、MAXPOOL、ARGMAX 同上。
3. LeNet-5 MNIST 完整 8 层：逐层回读，逐层比对；再跑 100 张图统计准确率，与 `npu_ref.py` 的准确率必须完全相等。
4. LeNet-5 CIFAR-10：同上，只换数据文件。
5. 错误注入：每个错误码至少一个用例。

### 2.3 断言（SVA，放在 `tb/common/` 并 bind 到 RTL）

- AXI4-Lite：valid 拉高后在 ready 之前不得撤销或改数据；bresp / rresp 只能是 OKAY 或 SLVERR。
- 存储：engine 写与主机写不得同拍发生；读地址不得越界。
- 阵列：`first` 与 `last` 之间的 valid 拍数等于 K_total；drain 期间 shadow 不被覆盖（tile 周期 >= ARR_N）。
- sequencer：busy 为 0 时不得有 engine start；done 与 err 互斥。

## 3. 工具与运行方式

| 工具 | 用途 | 运行位置 |
| --- | --- | --- |
| Verilator 5（`--timing`） | lint、单元与集成仿真 | 本地、GitHub Actions |
| VCS | 同一套 filelist 的额外验证，记录在 `docs/STATUS.md` | 培训服务器 |
| Python 3 + numpy | golden model、向量生成 | 本地、GitHub Actions |
| PyTorch | 只用于训练，不进 CI | 本地可选 |

约束：

- testbench 使用 Verilator 支持的 SystemVerilog 子集：过程式 TB、`$readmemh`、`$fopen` / `$fscanf`、简单 class 不使用，不使用 UVM。
- 每个 testbench 以 `PASS` 或 `FAIL` 行结束并用 `$finish` / `$fatal` 退出，Makefile 以退出码判定。
- `make lint`、`make unit`、`make integ` 三个目标，CI 全部执行。
- 覆盖率在 M5 用 Verilator `--coverage` 做一次功能点核对，不追求数字。

## 4. 里程碑的完成定义

| 里程碑 | 完成定义 |
| --- | --- |
| M1 | `requant_unit`、`pe` 单元 TB 通过；`npu_ref.py` 的 requant 函数与 SV 参考函数对 10 万随机样本一致 |
| M2 | `systolic_array` 对 200 组随机 GEMM bit-exact，包括 K < 8 和 K > 256 |
| M3 | `axil_slave`、`npu_regs`、`npu_sequencer` 单元 TB 通过；错误码全部覆盖 |
| M4 | 单层 CONV2D / FC / MAXPOOL / ARGMAX 集成 TB bit-exact |
| M5 | LeNet-5 MNIST 和 CIFAR-10 端到端 bit-exact；准确率与 golden model 相等；CI 全绿；README 有性能表 |
| M6 | 可选项各自定义：DMA 集成 TB、chip_top 的 CDC TB、FPGA 综合报告 |
