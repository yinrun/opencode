# SwiGLU Op Spec

## 数学定义

```
y = silu(gate) * up
silu(x) = x * sigmoid(x)
```

Qwen3 的 MLP 结构：gate_proj 和 up_proj 分别投影后做 SwiGLU 激活。

## 模型参数（Qwen3 0.6B）

| 参数 | 值 |
|------|-----|
| hidden_dim | 896 |
| intermediate_dim | 4864 |

## 输入输出

| 名称 | Shape (decode) | Shape (prefill S=128) | DType |
|------|---------------|----------------------|-------|
| gate | [1, 1, 4864] | [1, 128, 4864] | float16 |
| up | [1, 1, 4864] | [1, 128, 4864] | float16 |
| output y | [1, 1, 4864] | [1, 128, 4864] | float16 |

## HVX 实现要点

1. **Element-wise**：chunk-walk pattern（参考 sigmoid v7 模板）
2. **sigmoid 部分**：Schraudolph + polynomial-corrected exp + 2 Newton recip
3. **两路对齐**：gate 和 up 必须逐元素对齐
4. **融合**：silu(gate) * up 可以在同一个 HVX loop 内完成（减少一次内存读写）
5. **Layout**：PlainFloat16Tensor + F16CroutonTensor 双 variant

## 数值陷阱

- sigmoid 在 |x| > 10 时 fp16 精度下降 → polynomial 校正覆盖 [-16, 16]
- silu(x) = x * sigmoid(x)，当 x 很大时 silu ≈ x，数值稳定
- 两路乘法的中间结果可能超 fp16 range → 用 qf32 中间精度

## 验收标准

- cosine ≥ 0.999 vs QNN 分解形式（sigmoid + mul）
- max_abs_error ≤ 1e-3
- e2e ppl delta ≤ 1%

## 参考

- 已有经验：SiLU 项目（30 分钟完成，sigmoid v7 模板直接复用）
- Qwen3 源码：`Qwen3MLP.forward()` 中 `self.act_fn(self.gate_proj(x)) * self.up_proj(x)`
