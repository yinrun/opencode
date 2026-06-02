# MatMul Op Spec

## 数学定义

```
C = A @ B
```

通用矩阵乘法。在 Transformer 中用于 Linear 层（QKV projection、output projection、FFN）。

## 模型参数（Qwen3 0.6B）

| 用途 | A shape | B shape | C shape |
|------|---------|---------|---------|
| QKV proj (decode) | [1, 896] | [896, 1024] | [1, 1024] |
| Out proj (decode) | [1, 896] | [896, 896] | [1, 896] |
| Gate/Up proj (decode) | [1, 896] | [896, 4864] | [1, 4864] |
| Down proj (decode) | [1, 4864] | [4864, 896] | [1, 896] |

注意：B 是权重矩阵（静态），A 是激活（动态）。

## 输入输出（V1 测试用 Gate proj）

| 名称 | Shape (decode) | Shape (prefill S=128) | DType |
|------|---------------|----------------------|-------|
| A (activation) | [1, 896] | [128, 896] | float16 |
| B (weight) | [896, 4864] | [896, 4864] | float16 |
| C (output) | [1, 4864] | [128, 4864] | float16 |

## HVX 实现要点

1. **Tiling**：M×K×N 分块，HVX 一次处理 64 个 fp16 元素
2. **V1 策略（简单）**：外层 loop over M 和 N tiles，内层 dot product over K
3. **dot product**：`Q6_Wsf_vmpyacc_WsfVhfVhf`（fp16×fp16 → fp32 累加）
4. **不上 HMX**：V1 只用 HVX，性能让步（HMX 留 V2）
5. **Layout**：PlainFloat16Tensor，B 可能需要转置存储（row-major vs col-major）

## 数值陷阱

- K=896 的 dot product 需要 fp32 累加（fp16 累加 896 次必溢出）
- 大矩阵 tiling 的边界处理（K 不是 64 的倍数时需要 mask）
- 896 = 14 × 64，4864 = 76 × 64，都是 64 对齐 ✓
- 权重 B 的内存布局对性能影响巨大（列优先 vs 行优先）

## 验收标准

- cosine ≥ 0.999 vs QNN 内置 MatMul/FullyConnected
- max_abs_error ≤ 1e-3
- e2e ppl delta ≤ 1%
- latency ≤ 2× QNN（V1 软指标，不阻断）

## 参考

- QNN SDK 有内置 MatMul 和 FullyConnected op
- 难度 hard：tiling 设计复杂，性能优化空间大
- V1 目标是功能正确，不追求性能持平 QNN
- V2 再上 HMX 矩阵单元
