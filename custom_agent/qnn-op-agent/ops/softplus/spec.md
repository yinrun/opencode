# SOFTPLUS — 算子规格

## 数学定义

y = ln(1 + exp(x))

## QNN 对应算子

`ElementWiseSoftplus`

## 输入定义域

全实数域

## 输入输出

- **Input**: `in0` — fp16, shape [1, 1, 128, 4864]
- **Output**: `out` — fp16, shape [1, 1, 128, 4864] (same as input)

## HVX 实现要点

- 模板 A（全域 unary）
- Kernel: `float xv = (float)in(n,h,w,d); out(n,h,w,d) = <func>(xv);`
- 数值稳定实现：x>20 时直接返回 x（exp(x) overflow）。用 log1pf(expf(x)) 或 stable form max(x,0)+log(1+exp(-|x|))。

## 验收标准

- cosine ≥ 0.999
- max_abs ≤ 1e-3
