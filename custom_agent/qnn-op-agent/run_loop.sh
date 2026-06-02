#!/bin/bash
# 自动循环 resume 直到所有算子完成
# 解决 opencode agent 中途退出的问题

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

export QNN_SDK_242="${QNN_SDK_242:-/home/yinrun/software/qualcomm/qairt/2.42.0.251225}"
export ANDROID_SERIAL="${ANDROID_SERIAL:-204cbd30}"
export PATH="/home/yinrun/software/android-ndk-r29:$PATH"
export OPENCODE_EXPERIMENTAL_BACKGROUND_SUBAGENTS=true

MAX_ROUNDS=30
ROUND=0

while [ $ROUND -lt $MAX_ROUNDS ]; do
    ROUND=$((ROUND + 1))
    
    # 检查是否还有 pending/in_progress
    REMAINING=$(python3 -c "
import json
with open('state.json') as f:
    s = json.load(f)
remaining = sum(1 for o in s['ops'].values() if o['status'] in ('pending', 'in_progress'))
print(remaining)
" 2>/dev/null)
    
    if [ "$REMAINING" = "0" ]; then
        echo "[Round $ROUND] All ops completed!"
        break
    fi
    
    echo "[Round $ROUND] $REMAINING ops remaining. Starting agent..."
    
    # 清理 HTP
    adb -s "$ANDROID_SERIAL" shell "pkill -9 -f qnn_llama_runner" 2>/dev/null || true
    sleep 2
    
    # 跑 agent
    opencode run --agent op-builder --dangerously-skip-permissions \
        "读取 state.json，找到第一个 pending 或 in_progress 算子，实现它。参考 ops/silu/build/src/ 的代码结构和 AGENTS.md 的 lessons。验收标准: cosine≥0.999, max_abs≤1e-3。必须加 --use_native_input_files --use_native_output_files。完成后更新 state.json。"
    
    echo "[Round $ROUND] Agent exited. Sleeping 5s..."
    sleep 5
done

echo "=== Final Status ==="
python3 -c "
import json
with open('state.json') as f:
    s = json.load(f)
for op, info in s['ops'].items():
    icon = {'completed':'✅','in_progress':'🔄','pending':'⬚'}.get(info['status'],'?')
    print(f'  {icon} {op}: {info[\"status\"]}')
print(f'Total: {s[\"global\"][\"completed_count\"]} completed')
"
