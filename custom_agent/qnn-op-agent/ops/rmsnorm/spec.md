# RMSNorm Op Spec

## 数学定义

```
y = x * w / sqrt(mean(x², dim=-1, keepdim=True) + eps)
```

其中 `eps = 1e-6`（Qwen3 默认）。

## 模型参数（Qwen3 0.6B）

| 参数 | 值 |
|------|-----|
| hidden_dim | 896 |
| eps | 1e-6 |
| weight shape | [896] |

## 输入输出

| 名称 | Shape (decode) | Shape (prefill S=128) | DType |
|------|---------------|----------------------|-------|
| input x | [1, 1, 896] | [1, 128, 896] | float16 |
| weight w | [896] | [896] | float16 |
| output y | [1, 1, 896] | [1, 128, 896] | float16 |

## HVX 实现要点

1. **Row-wise reduce**：对 D=896 维做 sum-of-squares，需要 fp32 累加器避免 fp16 溢出
2. **rsqrt**：Quake fast rsqrt + 1 Newton iteration（参考 L2Norm 项目经验）
3. **逐元素乘法**：x * w * rsqrt_result，三路乘法
4. **Layout**：PlainFloat16Tensor（row-wise 依赖 D 维连续，不能用 Crouton）

## 数值陷阱

- fp16 直接累加 mean(x²) 在 D=896 时可能溢出 → 必须 fp32 累加
- eps 必须在 sqrt 内部：`rsqrt(mean + eps)` 而非 `1/(sqrt(mean) + eps)`
- weight 乘法顺序：先算 `x * rsqrt_result` 再乘 w，减少中间值范围

## 验收标准

- cosine ≥ 0.999 vs QNN 内置 RMSNorm
- max_abs_error ≤ 1e-3
- e2e ppl delta ≤ 1%

## 参考

- Qwen3 源码：`modeling_qwen3.py` 中 `Qwen3RMSNorm`
- 已有经验：L2Norm 项目的 row-wise + tree reduce + Quake rsqrt 模式
