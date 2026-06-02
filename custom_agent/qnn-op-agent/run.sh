#!/bin/bash
# QNN Op-Replace Agent 启动脚本
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# 环境变量
export QNN_SDK_242="${QNN_SDK_242:-/home/yinrun/software/qualcomm/qairt/2.42.0.251225}"
export ANDROID_SERIAL="${ANDROID_SERIAL:-204cbd30}"
export PATH="/home/yinrun/software/android-ndk-r29:$PATH"

# 确保设备上没有占用 HTP 的进程
echo "[$(date)] Clearing HTP..."
adb -s "$ANDROID_SERIAL" shell "pkill -9 -f qnn_llama_runner" 2>/dev/null || true
sleep 2

MODE="${1:-fresh}"

case "$MODE" in
  fresh)
    echo "[$(date)] Starting fresh agent run..."
    opencode run --agent op-orchestrator --dangerously-skip-permissions \
      "读取 state.json，找到第一个 pending 算子，委派给 op-builder 执行完整 8 阶段工作流。完成后继续下一个。"
    ;;
  resume)
    echo "[$(date)] Resuming from last session..."
    opencode run --agent op-orchestrator --dangerously-skip-permissions --continue \
      "从 state.json 恢复执行，继续处理 pending/in_progress 算子。"
    ;;
  builder)
    # 直接跑 builder（调试用）
    OP="${2:-softmax}"
    echo "[$(date)] Running op-builder directly for: $OP"
    opencode run --agent op-builder --dangerously-skip-permissions \
      "实现自定义 HTP 算子: $OP。读取 ops/$OP/spec.md，按 8 阶段工作流执行。参考 AGENTS.md 中的 lessons。"
    ;;
  test)
    echo "[$(date)] Quick status check..."
    opencode run --agent op-orchestrator --dangerously-skip-permissions \
      "读取 state.json 并报告当前队列状态。不要执行任何算子。"
    ;;
  *)
    echo "Usage: $0 {fresh|resume|builder [op_name]|test}"
    exit 1
    ;;
esac

EXIT_CODE=$?
echo "[$(date)] Agent exited with code: $EXIT_CODE"
exit $EXIT_CODE
