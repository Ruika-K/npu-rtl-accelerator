# Project Status

此文件是多个对话之间共享状态的入口。每完成一个有意义的改动，就更新这里；不要依赖聊天记录保存项目事实。

## 当前里程碑

M0 完成（工程初始化与首次发布）。v1 架构文档完成，等待 review 后进入 M1。

## 已完成

- 建立 RTL、testbench、仿真文件清单和文档目录。
- 建立 `npu_top` 占位模块与 reset smoke test。
- 建立 VCS 的统一 `make compile` / `make run` 入口。
- 加入 Git 忽略规则和项目级约定。
- 发布到 `https://github.com/Ruika-K/npu-rtl-accelerator`。
- 2026-10-29：完成 v1 架构定义，见 `docs/architecture.md`、`docs/programming_model.md`、
  `docs/verification.md`、`docs/decisions.md`。要点：descriptor 驱动、INT8 整数推理、
  8x8 output-stationary 脉动阵列、AXI4-Lite 主机接口、Python 整数 golden model bit-exact、Verilator CI。

## 里程碑路线

| 里程碑 | 内容 | 状态 |
| --- | --- | --- |
| M1 | `npu_pkg`、`requant_unit`、`pe`，Python `npu_ref.py` 的 requant 部分，单元 TB | 未开始 |
| M2 | `systolic_array` GEMM 与 numpy 对比 | 未开始 |
| M3 | `axil_slave`、`npu_regs`、`npu_sequencer`，错误码 | 未开始 |
| M4 | `act_buffer`、`weight_buffer`、`conv_addr_gen`、`conv_engine`、`pool_engine`、`argmax_engine`，单层 bit-exact | 未开始 |
| M5 | LeNet-5 MNIST 与 CIFAR-10 端到端，训练与量化脚本，CI 全绿，README 性能表 | 未开始 |
| M6 | 可选：AXI4 master DMA、`chip_top`（CDC / 配置桥 / 时钟门控）、FPGA 综合数据 | 未开始 |

## 下一步

1. review 四份 v1 文档，确认接口后合入 `main`。
2. 开 `feat/m1-requant-pe` 分支：写 `rtl/npu_pkg.sv`、`rtl/compute/requant_unit.sv`、`rtl/compute/pe.sv`，
   `model/npu_ref.py` 的 requant 函数，单元 TB，Makefile 增加 Verilator 目标，CI 增加仿真。
3. 在培训服务器用 VCS 跑同一套 filelist，记录到下方表格。

## 已知限制

- 本地没有培训班的 VCS 环境，主仿真器为 Verilator。
- 受限服务器只允许代码进入，不允许从服务器导出。
- 因此本仓库始终以本地原创源码为唯一 source of truth。
- 培训课程的规格文档只用于理解基线，其寄存器表、端口名、地址表都不进入本仓库。

## 决策记录

详细理由见 `docs/decisions.md`。这里只保留索引。

| 日期 | 决策 | 原因 |
| --- | --- | --- |
| 2026-07-13 | 以 `NPU/` 为仓库根目录 | 避免无意义的嵌套目录 |
| 2026-07-13 | 使用 SystemVerilog 的 `.sv` 扩展名 | 明确语言并便于仿真器识别 |
| 2026-07-13 | RTL 与 testbench 分开管理 | 综合边界清晰，也方便分别组织验证代码 |
| 2026-07-13 | 本地仓库是唯一 source of truth | 符合受限服务器的数据流和课程材料边界 |
| 2026-10-29 | descriptor 驱动的可编程计算核（ADR-001） | 同一 RTL 跑不同网络 |
| 2026-10-29 | INT8 整数推理，TFLite 风格 requant（ADR-002/003） | 面积小、可 bit-exact |
| 2026-10-29 | 8x8 output-stationary 脉动阵列（ADR-004/005/006） | 存储访问简单、无部分和 SRAM |
| 2026-10-29 | AXI4-Lite 主机接口，DMA 推迟到 M6（ADR-007） | SoC 集成标准、先验证计算核 |
| 2026-10-29 | Verilator 为主仿真器（ADR-010） | 任何人可复现，CI 可跑 |
| 2026-10-29 | MNIST 与 CIFAR-10 都支持（ADR-011） | 证明可编程性，RGB 路径被覆盖 |
| 2026-10-29 | 每个里程碑一个分支加 PR（ADR-013） | PR 历史即工程过程 |

## VCS 运行记录

| 日期 | Commit | VCS 版本 | Test | 结果 | 备注 |
| --- | --- | --- | --- | --- | --- |
| — | — | — | — | 尚未运行 | — |
