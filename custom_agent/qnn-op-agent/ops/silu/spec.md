# SiLU Op Spec

## 数学定义

```
y = x * sigmoid(x) = x / (1 + exp(-x))
```

逐元素激活，无参数、无归约维度。

## 模型参数（Qwen3 0.6B FFN）

| 参数 | 值 |
|------|-----|
| intermediate_dim | 4864 |
| activation | SiLU（applied to gate_proj output） |

注：SiLU 在 Qwen3 中作为 SwiGLU 的激活分支使用 (`silu(gate_proj(x)) * up_proj(x)`)，
本算子只覆盖 element-wise SiLU 部分，不含 gate × up 乘法。

## 输入输出

| 名称 | Shape (decode) | Shape (prefill S=128) | DType |
|------|---------------|----------------------|-------|
| input x | [1, 1, 4864] | [1, 128, 4864] | float16 |
| output y | [1, 1, 4864] | [1, 128, 4864] | float16 |

算子是 element-wise，对任意 shape 都成立，shape 仅作 HVX tile 切分参考。

## HVX 实现要点

1. **Element-wise 并行**：无 reduce、无 broadcast，按 64-element fp16 vector 切分即可
2. **Layout**：PlainFloat16Tensor（element-wise 与 Crouton 都行，先用 Plain）
3. **sigmoid 近似**：
   - 方案 A：`1 / (1 + exp(-x))` 直接算，需要 exp 近似
   - 方案 B：分段多项式 / minimax 近似（参考 QNN 内置 sigmoid）
   - 方案 C：查表 + 线性插值（fp16 范围有限，2^16 LUT 可行）
4. **fp32 中间**：sigmoid 在 |x|≈10 处接近饱和，fp16 直接算 exp 易溢出，建议 fp32 中转

## 数值陷阱

- `exp(-x)` 当 x 极负时溢出 → 用稳定形式：
  - `x ≥ 0`: `y = x / (1 + exp(-x))`
  - `x < 0`: `y = x * exp(x) / (1 + exp(x))`
- fp16 表示范围 [~-65504, 65504]，但 sigmoid 输入 |x| > 16 已饱和，可提前 clamp
- x = 0 时 y = 0（精确），不需要 eps
- 大负值时 y → 0（不是 NaN），保持单调性

## 验收标准

- cosine ≥ 0.999 vs numpy 参考（`x * (1/(1+exp(-x)))`，fp32 计算后 cast fp16）
- max_abs_error ≤ 1e-3
- e2e ppl delta ≤ 1%

## 参考

- PyTorch: `torch.nn.functional.silu` / `nn.SiLU`
- Qwen3 源码：`modeling_qwen3.py` 中 `Qwen3MLP.act_fn`
- QNN 内置 op：`Sigmoid` + `ElementWiseMultiply`（基线对照）
- 已有经验：rmsnorm 项目的 fp32 中间 + PlainFloat16Tensor 模式
