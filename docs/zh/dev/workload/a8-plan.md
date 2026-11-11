# pypto4A8 workload 方案

设计笔记 · 2026-11-11 · CPU 解释器已实现，时间回放待实现

目标是把 PyPTO 算子转换成 WSE + superscalarNPU 性能模型的工作负载（workload）。
首个里程碑采用单线程、单 PE，以整条 tile 操作为粒度进行乱序调度和执行。

## 确定的路线

```text
PyPTO operator
  -> early Tile IR after ConvertTensorToTileOps
  -> target adaptation passes
  -> CPU interpretation with input descriptions (optional payloads)
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

解释器执行循环、分支和标量表达式，通过描述 tile 操作记录具体操作实例。
输入输出、访问范围和依赖的维护不要求 tile 数值载荷；可选的功能执行负责数值计算。
依赖载荷的分支需要功能模式，纯 trace 模式在值未知时报告来源操作。
模型耗时来自 tile 操作和所建模的同步。

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

开发使用 fork 的 `pypto4A8` 分支。CPU 解释器代码位于 `python/pypto/workload/`，
单元测试位于 `tests/ut/workload/`，系统测试规划在 `tests/st/workload/`；模型侧负责调度和计时。
临时示例与生成产物放在外层工作区的 `workspace/matmul-tile-ir/`。

具体需要解决的问题见[适配问题笔记](a8-gaps.md)。

## 已实现的 CPU 解释器

`pypto.workload.TileInterpreter` 直接执行内部 `ir.Program`。当前仍依赖 CPU PyTorch，
不启动设备。Trace 捕获与数值执行独立：

```python
from pypto import DataType
from pypto.workload import TensorSpec, TileInterpreter

inputs = {
    "a": TensorSpec(shape=(128, 512), dtype=DataType.FP16),
    "b": TensorSpec(shape=(512, 128), dtype=DataType.FP16),
    "c": TensorSpec(shape=(128, 128), dtype=DataType.FP32),
}
interpreter = TileInterpreter(max_steps=1_000_000)
result = interpreter.run(
    program, entry="matmul_k_blocked", inputs=inputs,
    mode="trace_only", capture_trace=True,
)
outputs = result.outputs  # tuple of detached object descriptions / scalar values
result.write_trace("workspace/matmul-tile-ir/trace.jsonl")
```

默认 `trace_only` 执行标量、控制流和 tile 语义，更新对象版本与依赖，
不调用数值 handler，不分配 tile 数据载荷。输入可以是 `TensorSpec`、CPU Tensor 或二者混合。
Tensor 仅提供元数据，包括 store 在内的执行过程不改变其数据。`TensorSpec` 接受二维非负
shape、PyPTO FP16/FP32/BF16、BOOL 或 8/16/32/64 位有符号/无符号整数 dtype 和可选的非负元素 stride，默认连续布局。
重复传入同一实例保留共享身份；值相同但实例不同的 spec 代表独立输入。
不同 Tensor 视图共享存储仍会被拒绝。

`mode="functional"` 要求真实 CPU Tensor，先执行相同的语义与效果处理，再调用数值 handler。
对象返回值保持 Tensor 形式，store 原地更新目的输入。`result.mode` 保存模式。
两种模式的 `capture_trace=False` 都仅关闭记录，不影响执行、操作 ID、版本或生产者。
每次运行重置对象、步数和 trace，保留实例注册表。输入键匹配入口参数名，或去掉编译器
`__ssa_vN` 后缀后无歧义的名称。

解释器支持标量常量、算术、比较、布尔短路、cast、tuple、赋值、独立变量栈帧的函数调用、
普通/in-core/hierarchy 作用域、`if`、顺序 `for/while`、`yield` 和 `return`。
栈帧使用 `Var.unique_id`；步数包含表达式、语句和空循环迭代。

内置 `tile.load/store/move/matmul/matmul_acc` 使用二维描述，功能模式载荷仍仅支持 FP16/FP32。
语义实现校验参数，描述访问、输出形状、有效矩形、dtype 和逻辑存储。
功能模式下，load 按实际 PyTorch stride 复制有效矩形，tile 填充区置零；move 创建独立副本；
store 仅更新源 tile 的有效矩形。matmul 将输入转为 FP32 后计算并累加，这是功能参考精度，
不模拟硬件逐位精度或 TF32。lhs 有效 K 决定计算范围，rhs 有效 K 必须覆盖它。
`matmul_acc` 原地更新累加器并递增版本；`init_cond=True` 仅覆盖实际计算矩形，读取列表不含
累加器，但覆盖写依赖仍保留其上一写入者。整数测试输入精确比较，随机输入允许 FP32 归约顺序差异。

纯 trace 注册表还支持 `tile.create`、`add/sub/mul/maximum`、`muls`、`exp`、`cast`、
`row_max/row_sum` 和 `row_expand_sub/mul/div`。声明式操作表提供参数数量、支持的属性、
操作类别和 PTO 映射；通用 handler 从 `call.type` 读取输出元数据并记录效果。
归约 scratch 记为写入，复用时保留上一写入者依赖。`tile.create` 记录逻辑分配，
不映射到 PTO 指令。这些新增操作尚无数值 compute handler，功能模式会明确报错。

还支持 `div/adds/abs/sqrt/log`、`row_min/col_sum`、`col_expand_mul`、
`full/cmps/sel/mgather` 和 `slice/reshape/transpose`。二元操作接受单维广播。
比较结果保留 IR 的 packed UINT8 几何形状，不伪装成稠密 BOOL tile；`sel` 记录掩码、
数值输入的读取及 UINT32 scratch 写入。Vec `mgather` 读取源 Tensor 和 INT32 索引 tile；
缺少索引载荷时，访问范围显式标记为 `data_dependent`，使用整个对象的生产者依赖，
不会提供计时回放所需的具体 gather 地址。Mat gather 尚未支持。

slice 和逻辑 reshape 创建描述视图，不创建新存储。快照增加 `storage_id`、
`storage_offset`（逻辑未打包元素单位）、`view_of` 和 `view_producer`。
视图保留独立对象 ID、shape 和 stride，共享存储版本及写入者。创建视图不产生 PTO 指令，
也不写存储（`memory_access=False`）；消费者同时依赖视图创建和最新存储写入者。
一次操作写多个别名时，共享版本仅递增一次。当前对视图的写入保守地使整个分配失效，
尚无字节范围冲突跟踪。非连续 reshape 使用 `stride=null` 和显式 `index_map`，保存源
shape、stride、offset 及此前的映射，组合逻辑行优先 reshape，不伪造仿射 stride，也不分配或
复制 tile 载荷。对此视图继续 slice 会组合新映射；无法用仿射偏移表示时，
`storage_offset=null`。降维 slice 和独立传入的重叠 Tensor 输入仍会报错。
transpose 是实际 `pto.ttrans` 操作，具有独立输出存储及可选 scratch 写入。
`workspace/tile-trace-audit-more/audit_more.py` 的 14 个用例在两个截点覆盖上述路径及 RMSNorm。
这些新增 handler 仅支持 trace；功能模式仍明确报告缺少数值实现。

下一组描述补充 `relu/minimums/rsqrt/row_prod/col_max/row_argmax`、
`assemble/concat/cmp/transpose_view` 及纯 `row_expand/col_expand`。
col_sum 接受可选 scratch，cast 接受 saturation mode 和可选 scratch。
索引归约输出 INT32 并写 scratch。二元操作检查物理广播，输出有效范围直接取 IR 描述，
保留输入各自的尾块矩形，不再把有效范围当成需要相互广播的维度。
assemble 记录对目标存储的读取与插入写入，现有别名同步看到新版本；concat 创建独立输出。
transpose_view 共享存储，transpose 仍为实际复制。纯广播以 `shape_only_inputs` 标记形状模板，
避免将其误解为需要读取载荷的操作数。

Tensor→Tile 转换先为 `[1,N] op [M,1]` 的两个向量显式广播，再执行二元操作，
保留参数顺序，不再误选 row-expand 算术。第三轮工作区测试编译 20 个用例并捕获两个截点，
与前两轮的 21 个用例一起验证。C++ 回归需要重编译扩展。本工作区的
`workspace/tile-trace-audit-round3/incremental-build/` 辅助脚本使用当前 checkout 的 Ninja
命令重编译修改对象，与同 checkout 的其他已有对象重新链接；所有新产物及 Python 包覆盖目录
都放在 workspace 中。运行审计或测试时，将其 `python/` 目录放到 `PYTHONPATH` 首位；
默认源码目录下的扩展仍需正常重建后才能包含 C++ 修改。

move 接受与结果类型一致的显式 `blayout/slayout`，包括 Cube→Vector 布局；
布局仅作为元数据，CPU 存储不打包。早期 matmul 可以接收 Vec 对象，随后存储推导
才插入 Cube 搬运；解释器描述当前 IR，不强制最终设备放置约束。
工作区 `workspace/tile-trace-audit/audit.py` 编译七个用例，包含 vector 融合、softmax
和两个 KV 块的在线 softmax FlashAttention，再用纯 TensorSpec 捕获早期和存储推导后的两个截点。

`RuntimeObject` 独立保存 shape、PyPTO dtype、元素 stride、有效范围、存储类别、布局、
逻辑身份、版本和生产者，`data` 载荷可为空。`describe()` 返回独立的 JSON 兼容元数据快照。
`Mat/Left/Right/Acc/Vec` 为逻辑存储类别，记录隐式或显式布局，CPU 载荷不打包。
显式 tile stride/offset、物理 MemRef、打包 Tensor 布局、不支持的属性和并行循环会被拒绝。
早期 Mat bridge 标记和调用方向元数据不改变这些语义；`mx_tensor_views_blocked` 对普通 ND
输入无需运行时动作。cache 策略、分阶段/原子 store 和 FIXPIPE epilogue 留待后续 handler 支持。

每条事件包含从零开始的 ID、IR/PTO 名称、函数/源码位置、具体 args/kwargs、输入输出快照
和直接生产者依赖；load/store 还包含 Tensor ID 与偏移。初始输入版本为零、无生产者；
每次操作写入（包括新 tile 的首次写入）递增版本。这是单线程依赖，尚非完整的调度/别名模型。
执行引擎负责 ID、版本和生产者，记录器仅保存快照。在相同控制流路径上，相同受支持 IR 与
输入描述在两种模式中产生相同 trace。当前导出 JSONL；manifest/objects 独立文件和计时回放待实现。

扩展注册两个独立入口：

```python
interpreter.register_op("tile.example", semantics=describe_example, compute=compute_example)
```

语义入口接收 `(interpreter, call, evaluated_args)`，返回
`OpResult(value, reads, writes, pto_op, parameters, views=())`。它校验属性，通过
`interpreter.new_tile(call.type)` 创建描述，并报告对象读写效果，不访问载荷。
`interpreter.call_attributes(call)` 提供仅求值一次的 kwargs。
数值入口接收 `(interpreter, call, evaluated_args, result)`，填充或更新载荷；
对于依赖数据的标量，可替换 `result.value`。`allocate_payload(obj)` 分配新的零填充 tile，
`payload(obj)` 获取必需的已有数据。两个入口都不得修改身份、版本和生产者。
compute 可省略：只有语义实现的操作可以捕获 trace，在功能模式下会报错。
自定义操作名无需注册为内置 IR 操作。

依赖缺失数据的标量由 `interpreter.unknown(call)` 显式表示为 `UnknownValue`，
保存来源操作 ID、名称、函数和位置；算术、比较和 cast 传播未知值。对象 shape、stride、
dtype 仍已知。分支条件、循环边界、地址、shape、标量 mask 参数和指令参数/属性需要具体值，
遇到未知值会报告来源与消费位置。布尔表达式仅求值必需的操作数；未知标量返回值仍显式保留。
缺少语义实现或属性不受支持时，两种模式均报错；缺少 compute 或必需载荷时，
功能模式报告操作/对象上下文。

失败运行抛出 `InterpreterError`，`interpreter.recorder.records` 保留成功记录的前缀用于诊断，
不返回 `ExecutionResult`。仅从成功结果导出完整 trace。此前成功结果及其对象返回描述
与后续运行保持独立。

在外层工作区运行可重复示例：

```bash
PYTHONPATH=pypto/python .venv-ir/bin/python workspace/matmul-tile-ir/run_interpreter.py
```

示例从默认 pass 序列取得 `ConvertTensorToTileOps` 和 `InferTileMemorySpace` 截点，
用纯 TensorSpec 捕获 13/21 条事件，核对功能模式 trace 完全相同、整数输入数值结果精确一致，
向 `workspace/matmul-tile-ir/interpreter-output/` 导出 JSONL。
`tests/ut/workload/test_interpreter.py` 还覆盖随机输入精度、纯 trace 模式不分配 tile 载荷、
Tensor 输入不变、未知值诊断、短路、尾块/非连续访问、快照及注册表隔离。
