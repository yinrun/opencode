# QNN Op-Replace Agent

自主 AI agent 系统，端到端替换 Transformer 模型中 QNN HTP 内置算子为自定义 HVX kernel。

**能力**：给定一个算子的数学规格（spec.md），agent 自动完成 IDL 定义 → HVX C++ kernel → 三平台编译 → 设备部署 → 数值验收。

**当前状态**：21/21 算子全部通过验收（cosine ≥ 0.999, max_abs ≤ 1e-3），0 永久失败。

## 前置条件

| 依赖 | 版本/路径 |
|------|-----------|
| OpenCode | 源码编译（Go）`~/workspace/opencode/` |
| QNN SDK | 2.42 `/home/yinrun/software/qualcomm/qairt/2.42.0.251225` |
| Hexagon SDK | 6.4 |
| NDK | r29 `/home/yinrun/software/android-ndk-r29` |
| Python | 3.10+，安装 `fastmcp` 和 `numpy` |
| 设备 | SM8850 通过 USB 连接（serial: 204cbd30） |
| LLM API | kiro-api endpoint `http://10.189.155.29:8000/v1` |

## 快速开始

### 1. 跑完整个算子队列（无人值守）

```bash
./run_loop.sh
```

脚本循环执行直到 `state.json` 中所有算子 completed。每轮处理 1 个算子。

### 2. 跑单个算子（调试/手动）

```bash
./run.sh builder <op_name>
# 例如：./run.sh builder softmax
```

### 3. 检查当前状态

```bash
python3 -c "
import json
with open('state.json') as f:
    s = json.load(f)
for op, info in s['ops'].items():
    icon = {'completed':'✅','in_progress':'��','pending':'⬚'}.get(info['status'],'?')
    print(f'  {icon} {op}: {info[\"status\"]}')
print(f'Total: {s[\"global\"][\"completed_count\"]} completed')
"
```

## 添加一个新算子

### Step 1：写 spec.md

在 `ops/<op_name>/` 下创建 `spec.md`，格式：

```markdown
# <OpName> Op Spec

## 数学定义
y = <公式>

## 输入输出
| 名称 | Shape | DType |
|------|-------|-------|
| input | [1, 1, 128, 896] | float16 |
| output | [1, 1, 128, 896] | float16 |

## 用途
<在哪些模型/层中使用>

## 数值陷阱
- <可能的精度问题>

## 验收标准
- cosine ≥ 0.999, max_abs ≤ 1e-3
```

参考已有的 `ops/tanh/spec.md` 或 `ops/matmul/spec.md`。

### Step 2：注册到 state.json

在 `state.json` 的 `queue` 数组末尾加入 op 名，并在 `ops` 字段添加：

```json
"<op_name>": {
  "status": "pending",
  "iterations": 0,
  "metrics": null,
  "report_path": null,
  "session_id": null,
  "started_at": null,
  "completed_at": null
}
```

### Step 3：运行

```bash
./run.sh builder <op_name>
```

或加入 `run_loop.sh` 自动队列。

### Step 4：验收

Agent 完成后检查：
- `ops/<op>/outputs/device/Result_0/output_native.raw` — 设备执行输出
- `state.json` 中该 op 的 `metrics` — cosine 和 max_abs_err
- `ops/<op>/patches/` — 如果有迭代，记录了每次假设和补丁

## 恢复中断的任务

Agent 中途退出（context 耗尽、网络中断等）时：

```bash
# run_loop.sh 会自动 resume，不需要手动操作
./run_loop.sh

# 或手动恢复单个 op
./run.sh builder <中断的 op>
```

Agent 会检查已有产物（.so、output_native.raw），不会重复已完成的步骤。

## 常见问题

### "Device Creation failure"
→ `ADSP_LIBRARY_PATH` 缺 `libQnnHtpV81Skel.so`。确认设备 `/data/local/tmp/qnn_op_test/htp/` 下有该文件。

### cosine 极低（如 0.03）但数值有规律
→ I/O dtype 不匹配。确认 qnn-net-run 命令包含 `--use_native_input_files --use_native_output_files`。

### "unable to find graphName"
→ 设备上 `htp_config.json` 残留上一个 op 的 graph_name。重新 push 当前 op 的 config。

### "Register Op Packages failure"
→ InterfaceProvider 名写错。用 `nm -D <arm>.so | grep InterfaceProvider` 查真实导出符号。

### Agent 输出为空 / 不执行任何操作
→ OpenCode subagent 偶尔空返回。重新运行即可（run_loop.sh 自动重试）。

### 多设备环境 adb 报错
→ MCP server 不支持多设备。确保 `export ANDROID_SERIAL=204cbd30` 或只连一台设备。

## 目录结构

```
.
├── opencode.json              ← Agent + MCP 配置
├── AGENTS.md                  ← Rules + 12 条 Lessons
├── state.json                 ← 队列状态（21 ops）
├── run_loop.sh                ← 无人值守循环脚本
├── run.sh                     ← 单次运行脚本
├── .opencode/agents/
│   ├── op-orchestrator.md     ← Primary agent 定义
│   └── op-builder.md          ← Builder subagent 定义（核心）
├── mcp_servers/               ← 4 个 Python MCP server
│   ├── adb_server.py          ← 设备交互
│   ├── qnn_build_server.py    ← IDL + NDK 编译
│   ├── qnn_runner_server.py   ← 测试输入 + 执行
│   └── tensor_compare_server.py ← 数值对比
├── ops/<op_name>/             ← 每个算子的完整产物
│   ├── spec.md                ← 输入：算子规格
│   ├── build/                 ← 源码 + 编译产物
│   ├── patches/               ← 迭代记录
│   └── report.md              ← 最终报告
└── baseline/
    ├── metadata.json
    └── test_inputs/<op>/      ← 确定性测试输入（seed=42）
```

## 验收标准

| 指标 | 阈值 |
|------|------|
| cosine similarity | ≥ 0.999 |
| max absolute error | ≤ 1e-3 |

对比对象：自定义 HVX kernel 输出 vs QNN 内置参考算子输出，使用相同 seed=42 的测试输入。

## 技术栈

- **Agent Runtime**: OpenCode (Go) + Claude Opus 4.7
- **MCP**: Python FastMCP (stdio)
- **编译**: QNN SDK 2.42 + Hexagon SDK 6.4 + NDK r29
- **设备**: SM8850 HTP DSP (hexagon-v81)
- **对比**: numpy (cosine similarity + max abs error)
