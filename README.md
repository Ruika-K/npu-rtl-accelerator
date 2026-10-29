# NPU RTL Accelerator

这是一个从零理解并独立实现 NPU 的个人 SystemVerilog 项目。目标不只是让代码跑起来，也要留下清楚的架构决策、验证过程和可复现的里程碑，最终作为作品集项目展示。

> 本仓库只包含本人独立编写的代码、公开资料和个人笔记，不包含培训机构源码、受限服务器内容或其他未经许可的材料。

## 仓库放在哪里

`/Users/ruikakuraba/NPU` 就是仓库根目录，不需要再创建一层同名项目文件夹。顶层 RTL 文件位于 `rtl/npu_top.sv`，而不是根目录下一个没有扩展名的 `npu_top` 文件。

```text
NPU/
├── rtl/                  # 可综合 SystemVerilog RTL
│   └── npu_top.sv        # 随里程碑增加 rtl/host, rtl/ctrl, rtl/compute, rtl/mem
├── tb/                   # testbench、断言（随里程碑增加 tb/unit, tb/integration, tb/common）
│   └── npu_top_tb.sv
├── model/                # M1 起：Python 整数 golden model、量化导出、测试向量
├── sim/filelists/        # Verilator / VCS 共用的文件清单
├── docs/
│   ├── architecture.md       # 数值格式、数据流、模块划分、接口
│   ├── programming_model.md  # 地址映射、寄存器、descriptor、运行流程
│   ├── verification.md       # 验证计划与里程碑完成定义
│   ├── decisions.md          # 每个关键决策的理由
│   └── STATUS.md             # 共享进度
├── .github/workflows/    # GitHub 自动检查
├── AGENTS.md             # 项目约定
└── Makefile              # 统一的仿真入口
```

项目变大以后，再按功能增加 `rtl/compute/`、`rtl/memory/`、`tb/unit/` 等目录；当前不要过早建立大量空目录。

## 在 VCS 服务器运行

把整个仓库复制到允许使用 VCS 的服务器，在仓库根目录执行：

```sh
make compile
make run
```

也可以用 `VCS_FLAGS`、`TOP` 或 `FILELIST` 覆盖默认参数。仿真生成物统一进入 `build/`，不会提交到 GitHub。

## 推荐开发节奏

1. 先在 `docs/architecture.md` 写清一个小功能的接口和行为。
2. 在 `rtl/` 独立实现它。
3. 在 `tb/` 编写自检 testbench，至少覆盖正常、边界和错误场景。
4. 将仓库复制到培训服务器，用 VCS 编译和运行。
5. 只把自己电脑上的原创修复提交到 GitHub；不要从受限环境提取任何代码。
6. 在 `docs/STATUS.md` 记录结果和下一步。

## 设计概览（v1）

- 由 layer descriptor 驱动的 INT8 CNN 推理加速器：同一份 RTL 通过更换 descriptor 和权重运行 LeNet-5 的 MNIST 与 CIFAR-10 版本。
- 计算核为 8x8 output-stationary 脉动阵列，int8 x int8 乘、int32 累加、整数 requant，全程无浮点。
- 片上 32 KB activation buffer 与 128 KB weight buffer，主机通过 AXI4-Lite 配置与装载。
- 验证以 numpy 整数 golden model 为唯一真值，RTL 逐层 bit-exact；Verilator 本地与 CI 可复现，VCS 作为附加验证。

详细说明见 `docs/architecture.md`，理由见 `docs/decisions.md`。

## 当前状态

M0 完成，v1 架构文档完成，M1（requant 单元与 PE）尚未开始。进度与下一步见 `docs/STATUS.md`。
