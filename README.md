# NPU RTL Accelerator

这是一个从零理解并独立实现 NPU 的个人 SystemVerilog 项目。目标不只是让代码跑起来，也要留下清楚的架构决策、验证过程和可复现的里程碑，最终作为作品集项目展示。

> 本仓库只包含本人独立编写的代码、公开资料和个人笔记，不包含培训机构源码、受限服务器内容或其他未经许可的材料。

## 仓库放在哪里

`/Users/ruikakuraba/NPU` 就是仓库根目录，不需要再创建一层同名项目文件夹。顶层 RTL 文件位于 `rtl/npu_top.sv`，而不是根目录下一个没有扩展名的 `npu_top` 文件。

```text
NPU/
├── rtl/                  # 可综合 SystemVerilog RTL
│   └── npu_top.sv
├── tb/                   # testbench、断言、参考模型
│   └── npu_top_tb.sv
├── sim/filelists/        # VCS 等仿真器使用的文件清单
├── docs/                 # 架构、验证计划、项目进度
├── .github/workflows/    # GitHub 自动检查
├── AGENTS.md             # 所有 Codex 对话共享的项目约定
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

## 当前状态

当前处于 M0（工程初始化）。下一步是确定第一个可以独立验证的小模块及其规格，详见 `docs/STATUS.md`。
