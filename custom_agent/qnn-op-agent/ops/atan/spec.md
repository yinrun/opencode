# ATAN — 算子规格

## 数学定义

y = arctan(x)

## QNN 对应算子

`ElementWiseAtan`

## 输入定义域

全实数域

## 输入输出

- **Input**: `in0` — fp16, shape [1, 1, 128, 896]
- **Output**: `out` — fp16, shape [1, 1, 128, 896] (same as input)

## HVX 实现要点

- 模板 A（全域 unary）
- Kernel: `float xv = (float)in(n,h,w,d); out(n,h,w,d) = <func>(xv);`
- 全域 op，atanf() 输出 ∈ (-π/2, π/2)。

## 验收标准

- cosine ≥ 0.999
- max_abs ≤ 1e-3
