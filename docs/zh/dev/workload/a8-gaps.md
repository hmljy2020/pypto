# pypto4A8 适配缺口

本文记录[workload 方案](a8-plan.md)尚需完成的工作，日期为 2026-11-11。
缺口集中在 trace 语义、目标映射和性能建模。
重建 BlockISA 解码器或标量执行流水线不属于已确定的初版范围。

## Trace 生成与规范化

| 缺口 | 现有基础 | 需要解决的问题 |
| ---- | -------- | -------------- |
| 可执行生成器 | PyPTO 提供早期 Tile IR；`pto-isa` 可提供可选 CPU 参考 | 实现目标适配，再解释执行支持的 Tile IR 子集；CPU 插桩单独评估 |
| 标量与控制流求值 | tile 操作周围仍有循环、分支、偏移和 shape 表达式 | 记录时算出具体值；忽略耗时，不忽略效果 |
| Pass 边界 | `ConvertTensorToTileOps` 已产生 `Mat/Acc` 和执行角色标注，后续 pass 继续增加特化 | 在该 pass 后直接获取 IR 对象，检查前置条件，固定目标适配配方 |
| 操作数完整性 | CPU 记录已有 opcode、worker 身份、序号、tile 地址/shape/layout/dtype 和标量操作数 | 补齐 stride、backing 身份、内存空间、分配范围与有效范围、模板属性 |
| 依赖 | 输入输出角色和 IR effect 提供基础 | 保留读改写、别名、内存冲突、视图、对象代际和真实同步 |
| Trace 身份 | 已有 CPU 地址和 worker 内序号 | 规范化为 launch/thread/op/object 身份，处理重复 launch 和地址复用 |
| 可复现性 | PyPTO 及其 runtime 固定外部 `pto-isa` 版本 | 记录生成器、pass 配方、版本、输入身份、profile/schema 版本；采用插桩时显式选择 CPU 库 |

本地 `pto-isa` 已有 `PTO_CPU_SIM_TRACE_MODE`、JSONL 导出，以及
`ResetInstructionTrace()` / `DumpInstructionTraceJson()` 接口。
直接调用 kernel 时，需要在执行线程上显式管理采集和导出，不会自动进入 launch wrapper。
插桩是可复用的生成端基础，但还不是完整 workload 契约；尤其不能假定模板参数
和全部同步效果已经出现在现有记录中。

PyPTO 源码树不直接包含 `pto-isa`。默认 runtime 解析器使用 `runtime/pto_isa.pin`，
要求受管理 checkout 干净且匹配固定版本；它导出 `PTO_ISA_ROOT`，但不读取它作为覆盖配置。
查阅时 pin 为 `f83d4b7e4be0a4ca9805944ae2d9ce8c84e581f7`，
被检查的并列 checkout 为 `49adb4a6`。后者存在插桩不代表前者具有相同能力。
实现见 PyPTO 源码中的 `python/pypto/runtime/pto_isa.py`。

## 解释执行前的目标适配

以下缺口属于主生成路线，不能拖到临时 trace 清洗中处理，也不能让计时模型静默猜测。

| 缺口 | 完成条件 |
| ---- | -------- |
| 合法操作子集 | 检查 dtype、shape、有效范围和语义属性，在记录前明确支持的分解规则和报错条件 |
| 早期架构绑定 | 将 `Mat/Acc`、矩阵桥接元数据和执行角色映射到目标 profile，保留用户可见语义 |
| 必要数据搬运 | 显式决定目标存储转换/布局转换，区分元数据视图与 copy；不能盲目保留或删除 Ascend staging |
| 功能完整性 | 求值循环携带变量、分支和标量表达式，执行数据相关控制流需要的 tile 数值，拒绝不支持的行为 |
| 动态依赖身份 | 解释时建立逐实例生产者/读写关系、逻辑地址范围、别名和代际；仅修改元数据的操作仍需更新这些状态 |
| 物理资源职责 | Workload 表达存储类别和生命周期要求；有限资源的分配时刻、容量停顿及发射调度留给回放模型 |
| 完整性声明 | 校验一次受支持执行的逻辑操作流，不声称得到完整编码/分配后的目标程序；另需资源和通信 profile |

对于矩阵乘，当选定目标 profile 不需要额外传输时保留 13 条逻辑操作。
若适配加入 staging，则按该 profile 推导预期数量。Ascend 的 21 条操作 trace
不是目标必须执行的序列。使用 CPU 插桩交叉验证时，应先统一这些语义差别，
再比较操作数量或性能。

## superscalarNPU 适配

一条可执行 PTO tile 操作就是一个调度单位。
剩余 gap 是这些单位的语义和代价，并不是缺少 block 编码。

| 缺口 | 决策或需要实现的行为 |
| ---- | -------------------- |
| 操作覆盖 | 定义 load/store、矩阵、vector、reduction、cast、move 的支持范围及 dtype/layout/shape 约束；不支持的组合报错 |
| Tile 存储映射 | 将 PyPTO/PTO 存储描述映射到选定 PE 配置；旧 Ascend L0/L1/UB 名称不等于 superscalarNPU 存储 |
| 依赖与复用 | 跟踪 RAW 和存储冲突；定义重命名/复用策略，保留必要的 WAR/WAW 顺序并计入分配生命周期 |
| 准入与调度 | 指定有限窗口/队列、准入宽度、发射仲裁、资源占用、完成与释放，不要求细粒度指令解码器 |
| 整条操作计时 | 按 shape、dtype、layout 定义 latency、initiation interval、吞吐和端口占用；不能把单个微操作延迟直接当成整条 tile 延迟 |
| 内存计时 | 在声明的抽象层级上建模传输字节、stride、尾块、带宽/延迟、在途请求和竞争 |
| 元数据与实际工作 | 视图和绑定可能只改对象元数据，需要与真实 copy、layout 转换和搬运区分 |
| 架构配置 | 先采用单线程单 PE，显式分配计算/存储预算，不自动继承全部四 PE 假设 |
| 多 PE 扩展 | 增加逐线程 trace、PE 映射、共享资源和 event/barrier 语义；跨线程记录顺序不能强加全局串行关系 |

四 PE 参考组织包含逐 PE 的 Cube/Vector 执行资源及共享结构。
这不意味着 WSE 文档中的整核吞吐可以再乘四。
从一个 PE 扩展到多个 PE，需要声明整核计算和内存预算如何分配。

忽略标量耗时后，模型适合研究这一假设下的 tile 调度，
不能据此声称已经覆盖整核 IPC 或受标量限制的 kernel 性能。
固定 trace 也不能预测因并发时序改变而产生的另一条执行路径。
初版支持确定性、具有同步约束的 workload，明确拒绝不支持的轮询和竞争相关行为。

## WSE Calendar 通信 ABI

普通 PTO load/store 能表达数据搬运，但不能单独确定集合参与者、物理节点位置、
路由、Calendar 时隙服务和接收完成协议。
这些信息必须来自显式通信语义及配套计划，不能只从 load/store opcode 推断。

下表中的名称表达模型事件，不是新增机器指令编码。

| 缺口 | 需要的信息或行为 |
| ---- | ---------------- |
| 通信意图 | 集合类型、组、参与者、root/source、归约操作、逻辑传输身份和每节点通信 owner |
| 放置与路由 | 逻辑线程/PE → WSE 节点映射、拓扑、逐 source 的 `routeBits`、转发/落核语义和路由校验 |
| Calendar 计划绑定 | 将逻辑 key、`keyId`、`opcode`、路由表和 `CalReg` 镜像绑定到同一版本化计划；opcode 本身不能恢复镜像 |
| 几何与寻址 | 对称 arena/backing、source 的 `selfOff`、行数/行字节/stride、目标范围、容量、对齐和分段 |
| 轮次身份 | 匹配 launch、opcode 域、epoch 和 segment/source 身份；定义复用/排空规则并拒绝迟到旧数据 |
| 接收计量 | `expVal` 统计唯一且成功提交的远端有效载荷字节；不计头部、padding、重复包和本地自拷贝 |
| 准入与对齐 | 集合对齐前保留源/目标存储及接收/完成资源；已对齐节点必须具备有界收发服务 |
| 收发进展 | 接收登记完成与数据完成分离；最终接收等待不能阻塞发送发射/推进；区分源读出完成、发送排空和接收完成 |
| 可见性与生命周期 | 保留源和目的代际；仅在范围覆盖、写可见、本地 copy 义务和接收计量均满足后发布目的就绪 |
| 本地自拷贝与归约 | 明确 fabric self-delivery 策略、本地 copy 工作、归约语义和资源代价；本地字节不能错误满足远端 `expVal` |
| 装载与恢复 | 校验拓扑/ABI/计划版本兼容性，在停机窗口安装 `CalReg`；epoch/backing 复用或重配置前排空在途责任 |
| NoC 计时 | 实现 Calendar 放行、链路/端口竞争、转发/落核服务、缓冲、反压和时钟域换算；路由流量统计不足以得到性能 |

新 WSE 白皮书要求**先登记接收并完成资源准入，再对齐**。
旧 Calendar 文档则是在对齐后建立 MTE4 接收。
集成应采用新契约，同时保留路由、epoch 和字节计量思想，不能直接照搬旧 MTE/PIPE 绑定。

概念事件序列如下：

```text
producer ready + source retained
  -> post_receive + destination/resources reserved
  -> bounded send service ready
  -> group align
  -> send + optional local self-copy
  -> send drain and receive writes/accounting complete
  -> publish destination generation -> dependent compute
```

实际重叠由依赖表达，不强制全局串行时间线。
路由和 epoch 的标量准备可以离线求值；资源准入、对齐、数据移动和可见性仍需建模。

## 现有 WSE 模型与目标设计的差别

- `src/wse_model/noc/network.py` 明确只建模路由和字节流量，不提供网络计时。
  `CalReg` 内容与服务时序仍需模型实现。
- 决策 0007 将白皮书的核切换为 superscalarNPU，但未迁移 Python/ACIR 实现。
  现有核和内存行为不能作为新目标已经实现的证据。
- 白皮书规定 local SRAM；旧模型的 `LocalDram` 和 2 KiB 页规则不能静默成为新目标约束。
  还需明确本地地址/缓存语义与 SSM 统一 GM/一致性假设之间的差别。
- 40 节点 Calendar 基线使用 80-bit 路由位图；40/48 节点拓扑仍有未决问题。
  拓扑和路由位宽需要一起版本化。
- WSE 吞吐和带宽数值是整核目标预算，不是已经校准的 SSM 参数。
  未决目标选择应在 profile 中显式保留。

## 查阅来源基线

以下是本地查阅基线，不保证每个工作区干净，也不表示所有规范已经实现。
生成可复现 workload 时还需冻结准确版本和相关工作区改动。

| 仓库 | 查阅 HEAD | 相关文件 |
| ---- | --------- | -------- |
| PyPTO | `705cb86f` | `python/pypto/runtime/pto_isa.py`, `src/ir/op/tile_ops/matmul.cpp`, `src/ir/op/tile_ops/transform.cpp` |
| PTO Tile Library (`pto-isa`) | `49adb4a6` | `include/pto/cpu/trace.hpp`, `include/pto/common/pto_instr.hpp`, `docs/coding/cpu_sim.md` |
| SuperScalarModel | `5341e55dd` | `modelSpec/superscalar_model_top_framework.md`, `TimingSim/pe/vec/VecValuInfo.h` |
| wse-model | `e9ecb1a` | `docs/design/wse-system-architecture-whitepaper.md`, `docs/design/wse-calendar-scheme.md`, decision 0007, `src/wse_model/noc/network.py` |

这里的 `pto-isa` 指支持 CPU 执行的 Tile Library。
另一个 `pto-spec` 仓库以及 SuperScalarModel 的 block 指令定义是不同产物，
共享 PTO 名称不意味着接口可以直接互换。
