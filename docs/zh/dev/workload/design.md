# SSNPU 手写 Tile IR 与静态捕获

状态：负载侧首版已实现 · 2026-11-12

`pypto.workload.ssnpu.Builder` 直接构造带 SSNPU 属性的标准 PyPTO Function、Call、
TileType、循环、分支和 yield 节点。通过 `TileInterpreter.run()` 捕获，使用
`ExecutionResult.write_capture()` 导出。下面两个例子都是可执行 builder，不需要
新前端语法、自动分片、目标 lowering 或计时模型。

已有解释器与导入历史见 [plan.md](plan.md)，剩余集成问题见 [gaps.md](gaps.md)。

## 构造接口

| API | 语义 |
| --- | --- |
| `with ss.launch(name, blocks=2, threads=4) as fn` | 构造公共线程入口；通过 `fn.param`、`fn.return_type` 声明签名 |
| `ss.block_id()`、`ss.thread_id()` | 独立 block/thread 身份，原有 `subblock_id` 保持为零 |
| launch 的 `cores=[...]`、`pe_mapping=[...]` | 独立放置信息；默认一 block 一 Core，每 block 为 PE0～PE3 |
| `ss.shared_tile(shape, dtype)` | 声明归属 block 的共享句柄 |
| `ss.load_into(handle, tensor, offsets, pes=[0])` | 单生产者加载完整 tile；经过该定义的线程绑定同一个不可变版本 |
| `ss.load_local(tensor, offsets, shape)` | 将作者指定的区域加载到线程私有 tile |
| `ss.op(name, *operands, pes=None)` | 每个参与线程一条操作，可显式指定结果 shape/dtype |
| `reuse(value)` | 仅修饰该共享输入，不产生指令 |
| 普通共享输入 | 对该具体版本声明 last-use |
| `ss.barrier()` | 全 block 汇合并等待此前阶段的全部操作完成 |
| `ss.for_loop`、`loop.iter_arg`、`loop.return_var`、`ss.yield_` | 原有显式循环 builder、循环携带值与 yield |
| `ss.if_stmt`、`ss.function`、普通 GlobalVar 调用 | 原有分支和辅助函数 |

`pes=None`（模块常量 `ssnpu.ALL`）表示当前 block 的全部逻辑线程/PE 编号。
PE 映射将这些编号转换为物理放置。参与集合不会自动切分操作数或结果。
只由部分线程执行的分支必须显式指定相应操作的 `pes`；`full`、`load_local`、`op`、
`store` 都支持该参数。声明的参与者必须经过匹配的动态站点；不同分支不会默默拼接消费或同步。

TileType 的 memory space 保持未指定；调用的命名空间属性明确标注
`LocalTileReg`/`SharedTileReg`，此上下文拒绝用 Ascend Mat/Vec 枚举代替。
共享版本及其转置视图归属 block，私有 tile 不得跨 task/block/thread 引用。
目标层消费属性后将调用交给通用算术 handler，继续校验形状、有效区域与 dtype。

捕获专用的 `ssnpu.*` 使用普通 Op/Call 节点，不注册到 C++ lowering。
`str(program)` 可读输出保留属性；打印的 `pl.ssnpu.*` 是查看格式，不是新的 Python DSL
解析接口。属性序列使用已有 MakeTuple 表达式，因此本轮无需改 C++ 类型或绑定。

## 共享版本、消费与同步

句柄身份由 launch（即 `task_id`）、block、声明站点和动态调用/循环路径构成。
每次动态 `load_into` 创建独立的不可变版本。重用 Python/IR 变量名或再次写入同一
句柄不会改变已有值引用；不同版本拥有独立 storage identity。

消费者可以先于生产线程捕获，最终统一连接实际生产者。
`TileInterpreter(thread_order=(3, 2, 1, 0))` 可以验证这一点：对象和操作的分配编号
允许变化，但规范化后的定义、消费与依赖关系一致。动态 ID 是特定 IR 的不透明稳定字符串；
修改 IR 结构可能改变 ID。

每个共享操作数记录 `version_id`、`use_id`、`operand`、`participants` 和
`retention`（`reuse`/`last-use`）。同一次消费的各 PE 共用 use_id，多输入由 operand
区分。转置视图只是描述符操作，保留来源版本，不产生生命周期消费；QK 真正消费 Kᵀ 的版本。

最终校验覆盖生产者唯一性、定义一致性、私有归属、参与者/版本/保留标记匹配、同步站点/轮次
及 last-use 后读取。生命周期顺序来自线程内顺序和完整阶段 barrier，不使用全局捕获顺序。
不同参与子集的旧读与最终消费之间需要 barrier 建立顺序。只有 reuse 的版本可以保持 live-out，
捕获不会编造 last-use。生产者与同步顺序冲突造成的依赖环也会明确报错。

## 可执行 matmul

两个 block 各处理 256 行，每 PE 拥有两个 32 行分片。每个 K 迭代包括一次共享 B 加载、
八次私有 A 加载及八次 matmul_acc；第一次消费保留 B，第二次是 last-use。

```python
from pypto import DataType, ir
from pypto.workload import TensorSpec, TileInterpreter
from pypto.workload.ssnpu import Builder, reuse


def build_matmul():
    ss = Builder()
    with ss.launch("matmul", blocks=2) as fn:
        a = fn.param("A", ir.TensorType([512, 512], DataType.FP16))
        b = fn.param("B", ir.TensorType([512, 128], DataType.FP16))
        c = fn.param("C", ir.TensorType([512, 128], DataType.FP32))
        fn.return_type(c.type)
        bid, tid = ss.block_id(), ss.thread_id()
        row = ir.Add(
            ir.Mul(bid, ss.expr(256), DataType.INDEX, ir.Span.unknown()),
            ir.Mul(tid, ss.expr(32), DataType.INDEX, ir.Span.unknown()),
            DataType.INDEX,
            ir.Span.unknown(),
        )
        row1 = ir.Add(row, ss.expr(128), DataType.INDEX, ir.Span.unknown())
        shared = ss.shared_tile([128, 128], DataType.FP16)
        c0 = ss.full([32, 128], 0)
        c1 = ss.full([32, 128], 0)
        k = ss.var("k", ir.ScalarType(DataType.INDEX))
        with ss.for_loop(k, 0, 4, 1) as loop:
            acc0, acc1 = loop.iter_arg("acc0", c0), loop.iter_arg("acc1", c1)
            final0, final1 = loop.return_var("final0"), loop.return_var("final1")
            offset = ir.Mul(k, ss.expr(128), DataType.INDEX, ir.Span.unknown())
            bk = ss.load_into(shared, b, [offset, 0])
            a0 = ss.load_local(a, [row, offset], [32, 128])
            a1 = ss.load_local(a, [row1, offset], [32, 128])
            ss.barrier()
            out0 = ss.op("matmul_acc", acc0, a0, reuse(bk))
            out1 = ss.op("matmul_acc", acc1, a1, bk)
            ss.barrier()
            ss.yield_(out0, out1)
        ss.store(final0, c, [row, 0])
        ss.store(final1, c, [row1, 0])
        ss.return_stmt(c)
    return ir.Program([fn.get_result()], "shared_b", ir.Span.unknown()), {
        "A": TensorSpec((512, 512), DataType.FP16),
        "B": TensorSpec((512, 128), DataType.FP16),
        "C": TensorSpec((512, 128), DataType.FP32),
    }
```

每 block 预期 **36 次 load、32 次 matmul_acc、8 次 store 和 8 次初始化**。
每 block 有 8 轮 barrier，每轮四条线程到达事件。两个 block 共 168 条操作与 8 个 B 版本。

## 可执行 FA 前向

固定单头 dense attention：Q/K/V 为 `[256,64]` FP16，O 为 `[256,64]` FP32。
每线程持有 32 行 Q 与私有在线 softmax 状态。PE0 加载两个 KV 块；显式转置视图提供 Kᵀ。
QK、归约与在线累加使用 FP32，PV 前将概率转为 FP16。行归约按照现有 Tile IR handler
契约显式提供 scratch tile。使用上面的 imports：

```python
def build_fa():
    ss = Builder()
    with ss.launch("fa", blocks=2) as fn:
        q, k, v = [
            fn.param(name, ir.TensorType([256, 64], DataType.FP16))
            for name in ("Q", "K", "V")
        ]
        output = fn.param("O", ir.TensorType([256, 64], DataType.FP32))
        fn.return_type(output.type)
        bid, tid = ss.block_id(), ss.thread_id()
        span = ir.Span.unknown()
        row = ir.Add(
            ir.Mul(bid, ss.expr(128), DataType.INDEX, span),
            ir.Mul(tid, ss.expr(32), DataType.INDEX, span),
            DataType.INDEX,
            span,
        )
        ks = ss.shared_tile([128, 64], DataType.FP16)
        vs = ss.shared_tile([128, 64], DataType.FP16)
        qi = ss.load_local(q, [row, 0], [32, 64])
        m = ss.full([32, 1], -float("inf"))
        normalizer = ss.full([32, 1], 0)
        o = ss.full([32, 64], 0)
        scratch = ss.full([32, 128], 0)
        j = ss.var("j", ir.ScalarType(DataType.INDEX))
        with ss.for_loop(j, 0, 2, 1) as loop:
            mi, li, oi = [
                loop.iter_arg(name, value)
                for name, value in zip(("m", "l", "o"), (m, normalizer, o))
            ]
            mf, lf, of = [loop.return_var(name) for name in ("mf", "lf", "of")]
            offset = ir.Mul(j, ss.expr(128), DataType.INDEX, span)
            kj = ss.load_into(ks, k, [offset, 0])
            vj = ss.load_into(vs, v, [offset, 0])
            ss.barrier()
            kt = ss.op("transpose_view", kj)
            scores = ss.op("muls", ss.op("matmul", qi, kt), 0.125)
            mn = ss.op("maximum", mi, ss.op("row_max", scores, scratch))
            alpha = ss.op("exp", ss.op("sub", mi, mn))
            p = ss.op("exp", ss.op("row_expand_sub", scores, mn))
            ln = ss.op("add", ss.op("mul", alpha, li), ss.op("row_sum", p, scratch))
            pv = ss.op("matmul", ss.op("cast", p, dtype=DataType.FP16), vj)
            on = ss.op("add", ss.op("row_expand_mul", oi, alpha), pv)
            ss.barrier()
            ss.yield_(mn, ln, on)
        out = ss.op("row_expand_div", of, lf)
        ss.store(out, output, [row, 0])
        ss.return_stmt(output)
    return ir.Program([fn.get_result()], "shared_kv", span), {
        **{name: TensorSpec((256, 64), DataType.FP16) for name in ("Q", "K", "V")},
        "O": TensorSpec((256, 64), DataType.FP32),
    }
```

预期 **320 条操作**，包括 32 次 QK/PV matmul、16 个转置视图、8 次最终归一化和 8 次 store。
QK 输出 `[32,128]`，PV 和最终输出为 `[32,64]`，归一化分母为 `[32,1]`。

## 复现与导出

在外层 `hw-native-sys` 根目录运行，使用当前 checkout 的构建及安装好依赖的 Python 环境。
示例和生成物位于临时目录 `workspace/ssnpu-workload-design/`；稳定回归位于
`pypto/tests/ut/workload/test_ssnpu.py`。

```bash
export PYTHONPATH="$PWD/pypto/python"
export PYTHONPYCACHEPREFIX="$PWD/workspace/ssnpu-workload-design/pycache"
python workspace/ssnpu-workload-design/matmul_ir_design.py
python workspace/ssnpu-workload-design/fa_ir_design.py
```

产物位于该 workspace 下 `output/matmul/`、`output/fa/`，包括 `ir.txt`、分线程 JSONL、
`objects.json`、`tasks.json`、`events.jsonl`、`capture_config.json`、`manifest.json`。
也可以直接调用：

```python
program, inputs = build_matmul()
result = TileInterpreter().run(program, "matmul", inputs)
result.write_capture("workspace/ssnpu-workload-design/output/matmul")
```

测试时从 PyPTO checkout 加载 `.claude/skills/testing/load-env.sh`，并显式传入
`$PYPTO_TEST_JOBS`。pytest 的缓存和临时目录使用 workspace 下的绝对路径。

## 捕获格式与回放责任

SSNPU 使用 `format_version=2`、`ssnpu_spec_version=1`，普通解释器捕获保持格式 1 及
原有文件命名和行为。完成 manifest 仍最后写入。`capture_trace=False` 抑制 trace/event
输出，但不会跳过语义、参与者、生命周期或 barrier 校验。

| 字段 | 契约 |
| --- | --- |
| `task_id / block_id / subblock_id / thread_id` | 独立实例身份；task 即 launch |
| `tasks[].instances[].mapping` | 独立物理 `core_id`、`pe_id` |
| `ssnpu.dynamic_id / site / phase` | 动态操作分组与同步阶段 |
| `ssnpu.shared_inputs[]` | 每个操作数的具体版本和 reuse/last-use 契约 |
| 对象 `shared_handle / shared_version` | 声明与定义身份；不可变共享 storage 的普通 version 为 1 |
| `dependencies` | 生产者、视图和重叠访存依赖；等待前驱完成 |
| `order_dependencies` | 本线程前一条操作，表示指令登记/发射顺序 |
| `sync_dependencies` | 此前 block 阶段的全部操作，必须完成后才能开始当前操作 |
| `memory_accesses[]` | storage/object ID、读写、元素偏移、有效矩形、stride、storage offset；null 范围保守处理 |
| barrier 事件 `parameters` | 动态 ID、站点、轮次、全部参与者与完成契约 |
| barrier `completion_dependencies` | 汇合释放前必须完成的操作 |

访存分析识别单射行主序二维 Tensor 描述符的不相交矩形，不会用整个输出 Tensor 的
最近 writer 将独立输出分片串行化。至少一方为写的重叠访问保留 RAW/WAR/WAW 顺序。
launch 内没有其他顺序的冲突采用规范的 block/phase/thread 顺序，不取决于捕获访问顺序。
未知布局或别名回退到保守重叠。Tensor 快照 version 仍是捕获记账；地址内存回放应使用
`memory_accesses` 与显式依赖。

后端见到生产者指令时建立版本状态并申请容量，完整传输完成后发布 ready。
reuse 保留版本；last-use 本身完成还不足以回收，必须等待所有最终参与者及相关旧读完成，
包括尚未进入乱序窗口的旧读。旧读尚未登记完时，不能仅凭当前引用数为零释放。
barrier 本身不释放共享 tile。

本格式声明 `shared_storage_policy=independent_versions`。原地固定槽位后端必须另外
建立“旧读完成 → 新写入”的约束，不能把重复句柄当作允许覆盖活跃版本的许可。
当前不提供固定槽位属性或自动容量分配。

首版验收路径是纯 trace。已有数值 handler 继续保留；缺失数值实现及依赖未知载荷的控制流
明确报错。FA 端到端数值验证、尾块、mask、自动映射/lowering、周期估计和 WSE 通信属于后续工作。

## 对齐参考

设计基线：PTO-spec `b0165320`、SuperScalarModel `637bde0ec`。
`asl/block/operands/B.IOS.asl` 描述 reuse/last-use mask，
`asl/block/model/state/source-lifetime.asl` 跟踪代际和消费完成。
计时模型的 `TimingSim/group/sharetreg_rat.h`、`TimingSim/frontend/bctrl/BIssue.cpp`、
`BROB.cpp` 使用后继定义生产者提交时回收的策略；适配器需要与上面的生命周期契约对齐。
