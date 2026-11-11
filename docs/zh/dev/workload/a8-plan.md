# pypto4A8 workload 方案

状态：建模方向已确定，实现尚待开发。记录日期：2026-11-11。
下文的具体目录和 trace 字段是提案，不是已经存在的 API。
剩余工作和参考来源见[适配缺口](a8-gaps.md)。

## 目标与已确定范围

利用 PyPTO 为采用 superscalarNPU 核的 WSE 系统生成工作负载（workload）。
在 PyPTO fork 的 `pypto4A8` 分支开发，短期不以向主线提 PR 为目标。
PyPTO 负责生成 workload，性能模型负责调度与计时。

初版模型输入是已经保存、参数确定的 PTO tile 操作序列。
一条可执行的 tile 操作就是一个调度和执行单位。先实现单线程、单 PE，
反映 tile 级乱序执行。输入不需要经过 BlockISA 解码，也不要求标量流水线
和内部微指令拆分。多 PE 和 WSE Calendar 通信放在后续里程碑。

忽略的是标量**耗时**。决定循环次数、分支、地址、shape、mask 和操作属性的
标量值仍然必须算出来。忽略标量耗时不等于忽略这些语义。

## 生成与回放流程

```text
PyPTO operator + concrete inputs / shapes / scalar arguments
  -> PyPTO Tile IR immediately after ConvertTensorToTileOps
  -> target adaptation passes (legality, storage/movement, execution/sync)
  -> CPU Tile IR interpretation + recording
  -> normalize and validate PTO operation trace
  -> workload package
  -> tile-level superscalar replay
  -> WSE integration with a separate communication plan
```

主生成器是目标适配后的早期 Tile IR 解释器。CPU 插桩保留为可选生成器与验证路径：

| 生成器 | 做法 | 取舍与当前决定 |
| ------ | ---- | -------------- |
| Tile IR 解释器 | 先将早期 Tile IR 适配到我们的目标，再求值控制流、标量表达式和 tile 操作 | 主要开发路线，记录具体操作数、对象身份和依赖 |
| PTO CPU 插桩 | 生成 PTO，由 PTOAS 生成 C++，使用 `pto-isa` CPU 实现执行并记录 trace | 可选验证或替代生成器；需规范化目标专用 staging、布局和存储复用 |

CPU 插桩是已确定的“执行—记录—回放”路线中的实现选项，不改变模型接口。
目前尚未用我们生成的矩阵乘完整验证 PTO → C++ → CPU 插桩链路；
通过 `skip_ptoas=True` 生成 `.pto` 不能证明这条链路已经打通。
这条插桩链路不是解释器里程碑的前置条件。比较两种生成器的 trace 前，
必须保证它们保留相同的已声明目标语义。

这里的“静态 trace”是保存下来的、参数确定的**动态执行路径**，
不代表程序所有可能的执行。数据相关分支需要具体数据或功能参考求值器。
应记录输入身份及执行假设。依赖时序的竞争、轮询和自适应行为不能从一次
记录推广到所有运行，不属于初版支持范围。

记录阶段由 CPU 完成功能执行。回放阶段由模型把操作放入有界窗口，
等待依赖和资源、发射就绪操作并计算完成时间。CPU 时间戳及 CPU 完成顺序
既不是模型计时，也不是所有操作之间的依赖。顺序记录仍须保留独立操作的并行性。

## 选定 IR 边界与目标适配 pass

从 `ConvertTensorToTileOps` 刚完成后切入，位于后续目标存储/布局特化之前。
直接遍历 PyPTO 内部 IR 对象：`passes_dump` 中的 Python 文件是便于阅读的 IR 表示，
不是由 MLIR 转换的 Python，也不是解释器的输入格式。主路线不要求生成 PTO MLIR。

这一层并非完全架构无关。已查看的矩阵乘含有 `Mat/Acc`、编译器矩阵桥接属性，
以及 `InCore/CHIP_DIE/SubWorker` 标注。必须显式映射或检查这些信息，
不能把它们直接当作 WSE 节点、PE 放置或目标存储容量。

实现三个逻辑适配步骤，不一定拆成三个独立 pass：

| 步骤 | 职责 | 初版范围 |
| ---- | ---- | -------- |
| 操作合法化与支持检查 | 检查操作、dtype、shape 和属性；仅在目标需要时分解操作 | Load、store、matmul、matmul_acc；不支持的组合报错 |
| 目标存储与搬运适配 | 将源存储要求映射到目标 profile，保留别名/视图，插入必要搬运与布局转换 | 处理 `Mat/Acc`，不自动继承 Ascend 的 `Mat -> Left/Right` staging 路径 |
| 执行身份与同步适配 | 映射执行层次，保留显式顺序和同步语义 | 单逻辑线程、单 PE；后续扩展放置与 Calendar 语义 |

各部分的交接边界如下：

| 部件 | 决定什么 | 不提前决定什么 |
| ---- | -------- | -------------- |
| 适配 pass | 合法操作、存储类别、必要传输、执行范围和同步含义 | 发射时间或无限制的物理 tile 分配 |
| 解释器 | 具体路径、标量值、逻辑地址/视图、动态对象代际、操作实例和依赖 | 目标性能或跨线程全局时序 |
| 回放模型 | 有限资源分配、准入、发射/完成、竞争、存储释放和反压 | 缺失的操作语义或未记录的分支结果 |

例如，是否必须 staging copy 由目标适配决定；它何时执行、能否与矩阵运算重叠，
由回放模型决定。不增加 staging 的初始 profile 下，矩阵乘例子有 13 条操作；
若目标需要额外传输，则必须显式加入，并修改预期数量。

## Trace 完整性的边界

输出是**支持子集内、给定输入和确定执行路径下完整的逻辑 PTO 操作流**，
附带对象和访存描述，不是已经完成物理分配与编码的机器指令流。
`tile.matmul_acc` 可以直接成为规范化 `TMATMUL_ACC` 记录，不必经过 PTO MLIR。

输入绑定提供具体 tensor 描述和逻辑地址范围，物理放置由选定的目标策略负责。
不产生执行操作的视图仍需更新别名元数据。数据相关分支需要实际 tile 功能结果或
参考求值器，只记录操作名称不够。未知语义必须报错，不能生成看似完整的 trace。
Calendar 计划、计时和资源配置仍是模型的额外输入。

## 提议的 workload 契约

```text
workload/
  manifest.json       # schema, toolchain revisions, inputs, target profile
  objects.json        # allocations, generations, views, aliases, memory spaces
  trace.jsonl         # operation instances, operands, attributes, dependencies
  communication.json  # optional logical collectives and node placement
  calendar.json       # optional routes, CalReg plan, ABI/version identity
```

文件名和序列化格式暂未定稿。即使初版只有一次 launch、一个线程，身份中也要
保留 launch 和逻辑线程。需要的信息包括：

- 唯一操作实例、线程内序号，以及来源操作身份。
- opcode 和完整语义属性，例如转置、类型转换、舍入、累加、valid shape 和 mask。
- 逻辑对象、代际、视图偏移、shape、stride、layout 和 dtype。
  CPU 指针值本身不是可持久化的对象身份。
- 读写角色、生产者依赖、重叠内存访问、生命周期，以及存在时的显式同步。
- 已求值的标量操作数；初版不计标量执行延迟。
- 目标和工具链版本、pass 配方、输入或 seed 身份，以及不支持的功能。
  遇到不支持的语义应明确报错。

下面是一个规范化记录的提议示例，用于说明契约；它不是当前 `pto-isa` 的 JSON 格式：

```json
{
  "id": "launch0.thread0.op8",
  "thread": 0,
  "sequence": 8,
  "opcode": "TMATMUL_ACC",
  "reads": ["a1.v0", "b1.v0", "acc.v0"],
  "writes": ["acc.v1"],
  "depends_on": ["launch0.thread0.op6", "launch0.thread0.op7", "launch0.thread0.op2"],
  "attrs": {"m": 128, "n": 128, "k": 128, "acc_dtype": "fp32"}
}
```

对象表负责把这些名称绑定到存储和视图。逻辑版本表达数据流，
不意味着物理 tile 存储无限。回放仍需处理分配、别名、复用和容量反压。

## 首个矩阵乘 workload

采用 A `[128, 512]` FP16、B `[512, 128]` FP16、C `[128, 128]` FP32，
K 维拆成四块，每块 128：

```text
load A0, B0 -> matmul     -> acc0
load A1, B1 -> matmul_acc(acc0) -> acc1
load A2, B2 -> matmul_acc(acc1) -> acc2
load A3, B3 -> matmul_acc(acc2) -> acc3
store acc3 -> C
```

在早期 Tile IR 边界，动态执行共包含 8 次 load、1 次 matmul、
3 次 matmul_acc 和 1 次 store，即 13 条可执行 tile 操作。
已查看的内存空间 lowering 会额外加入 8 次 staging move，使后续边界变成 21 条。
数量依赖 pass 配方，这两种 trace 不能互相混用。
应显式选择并记录截取边界，不依赖带编号的 dump 文件名。
观察到的 21 条操作来自 Ascend lowering，不是 superscalarNPU 必须执行的序列。

首个回放应保留 accumulator 依赖链，同时在有限资源内允许独立 load 重叠。
数值校验需要声明精度策略；采用 FP16 乘积的 CPU 参考不一定与全 FP32
参考逐位一致。

## 仓库职责与提议目录

PyPTO 与 `pto-isa` 保持独立仓库，协同固定版本。
个人 fork 为 [hmljy2020/pypto4A8](https://github.com/hmljy2020/pypto4A8)，
本地和远端都使用 `pypto4A8` 分支；`origin` 保留源仓库，`fork` 指向个人仓库。
若修改插桩，需要自己的 `pto-isa` fork/开发分支；这是后续任务，本文不表示已经完成。

```text
python/pypto/workload/
  export.py                 # generation entry point
  pipeline.py               # early Tile IR boundary and target pass recipe
  transforms/               # target legality, storage/movement, execution/sync
  interpreter.py            # primary functional execution and trace producer
  cpu_trace.py              # optional instrumentation adapter / validation
  objects.py                # object identity, aliases, lifetimes
  trace.py                  # normalized trace contract
  validate.py               # supported subset and consistency checks
  targets/a8.py             # target mapping and workload profile
tests/ut/workload/           # semantic and schema checks
tests/st/workload/           # CPU generation/replay integration
docs/en/dev/workload/        # authoritative design and interface documents
docs/zh/dev/workload/        # synchronized Chinese documentation
```

新增 workload 专用工具链配置，记录仓库 URL、commit，以及显式开发路径。
不要修改默认解析器管理的缓存，也不要假设设置 `PTO_ISA_ROOT` 就能选中修改后的源码。
保留默认 runtime 的依赖契约，通过明确的 CPU workload 入口接入。
生成的 manifest 应包含实际版本和工作区改动身份。
不必仅为导出 trace 就新增编译器 backend。

`pto-isa` 负责指令插桩和 CPU 功能语义；PyPTO 负责规范化 workload 格式和生成文档；
模型仓库负责资源配置、调度与经过校准的计时。

临时示例代码、编译产物和查阅材料放在外层工作区的 `workspace/matmul-tile-ir/`，
不放进 PyPTO checkout。这是会被清理的临时目录，不能作为可复现测试 fixture
或受版本管理文档的依赖。正式实现、测试和设计文档放在上面的仓库目录中。

## 开发顺序与验收

1. 定义支持的操作子集、trace schema、对象模型和单线程 PE 配置，显式列出未决参数。
2. 在早期 Tile IR 边界加入目标适配步骤，再实现支持的标量/控制流与 tile 子集的 CPU 解释执行。
   生成 trace 前，检查必要传输和存储映射。
3. 导出并校验矩阵乘 workload，检查所选 pass 边界和目标 profile 下的操作数量、地址、累加依赖和数值结果。
   后续可以通过 CPU 插桩提供独立交叉验证。
4. 对接有界 tile 级回放，验证依赖停顿、独立操作重叠、资源竞争、tile 容量和内存反压。
5. 扩展视图、尾块、vector 操作和存储复用；不支持的操作必须报错，不能静默丢弃。
6. 增加多逻辑线程/PE、显式同步和共享资源计量，之后再加入 Calendar 通信与 NoC 计时。

验证应覆盖原地累加、重叠视图、同一地址被新代际复用、带 stride/尾块的传输，
以及在实现两种生成器时检查两者语义等价。
Calendar 还需覆盖重复/迟到 segment、消费者发射前的可见性、准入失败和收发进展。
数值正确不能替代性能计时验证。
