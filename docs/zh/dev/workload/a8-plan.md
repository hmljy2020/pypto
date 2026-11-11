# pypto4A8 workload 方案

设计笔记 · 2026-11-11 · 待实现方案

目标是把 PyPTO 算子转换成 WSE + superscalarNPU 性能模型的工作负载（workload）。
首个里程碑采用单线程、单 PE，以整条 tile 操作为粒度进行乱序调度和执行。

## 确定的路线

```text
PyPTO operator
  -> early Tile IR after ConvertTensorToTileOps
  -> target adaptation passes
  -> CPU interpretation with concrete inputs
  -> PTO operation trace + object descriptions
  -> superscalarNPU timing replay
```

从 `ConvertTensorToTileOps` 刚完成后的 PyPTO 内部 IR 对象出发。
这一层已经明确表达 tile 计算、load/store 和控制流；Python 格式的 pass dump
是这些对象的可读表示。

这一层也已经带有 `Mat/Acc` 等存储标注。我们通过目标适配 pass，
为这些信息赋予目标 PE 上的明确含义。

## 各阶段负责什么

目标适配包含三个步骤：

1. **操作支持**：明确合法操作、dtype、shape 和属性，按需要定义分解规则。
2. **存储与搬运**：映射存储类别，保留视图和别名，加入目标需要的数据搬运与布局转换。
3. **执行与同步**：将执行层次映射到线程/PE，保留顺序和同步关系。初版统一映射到单线程单 PE。

解释器执行循环、分支和标量表达式，计算 tile 操作，并记录具体操作实例。
执行过程中维护输入输出、访问范围和依赖；数据相关分支使用实际计算结果。
标量准备负责提供具体数值，模型耗时来自 tile 操作和所建模的同步。

回放模型将 trace 放入有限窗口，根据依赖和可用资源选择就绪操作，
计算执行与完成时间，并释放存储。是否需要一次搬运由适配阶段决定；
这次搬运何时执行、如何与计算重叠，由回放模型决定。

## Workload 包含什么

保存的 trace 描述给定输入下的一次执行。每条操作携带 opcode、具体操作数和属性、
launch/线程身份，以及生产者依赖。对象描述携带 shape、dtype、stride、layout、
逻辑地址范围、别名和代际身份。

产物暂定为 `trace.jsonl`、`objects.json` 和 `manifest.json` 三部分。
Manifest 记录输入身份、编译器版本、适配 pass 配方、目标配置和格式版本。
同一 workload 可以据此在多组计时参数下回放。

## 用矩阵乘打通第一个例子

采用 FP16 的 A `[128, 512]` 和 B `[512, 128]`，输出 FP32 的 C `[128, 128]`。
K 维分成四块，每块 128：

```text
load A0, B0 -> matmul                  -> acc0
load A1, B1 -> matmul_acc(acc0, A1, B1) -> acc1
load A2, B2 -> matmul_acc(acc1, A2, B2) -> acc2
load A3, B3 -> matmul_acc(acc2, A3, B3) -> acc3
store acc3 -> C
```

已生成的早期 Tile IR 在循环展开后共 13 条操作：8 次 load、4 次矩阵操作、1 次 store。
目标适配再加入需要的搬运，因此最终操作数量由选定的存储配置明确决定。

首个闭环要得到正确数值和完整 trace：保留 accumulator 依赖链，
同时暴露独立 load 的重叠机会。回放结果应能体现依赖、带宽和有限 tile 存储的影响。

## 开发顺序与位置

先定义 trace/对象格式和最小 PE 存储配置，再实现 load、store、matmul、matmul_acc
及其周围标量/控制流的适配与解释执行。将矩阵乘 trace 对接计时回放后，
扩展视图、尾块和 vector 操作，再推进多 PE 与 WSE Calendar 通信。

开发使用 fork 的 `pypto4A8` 分支。生成端代码计划放在 `python/pypto/workload/`，
测试放在 `tests/ut/workload/` 和 `tests/st/workload/`；模型侧负责调度和计时。
临时示例与生成产物放在外层工作区的 `workspace/matmul-tile-ir/`。

具体需要解决的问题见[适配问题笔记](a8-gaps.md)。
