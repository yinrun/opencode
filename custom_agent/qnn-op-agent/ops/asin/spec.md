# ASIN — 算子规格

## 数学定义

y = arcsin(x)

## QNN 对应算子

`ElementWiseAsin`

## 输入定义域

x ∈ [-1, 1]，超出范围输出 NaN

## 输入输出

- **Input**: `in0` — fp16, shape [1, 1, 128, 896]
- **Output**: `out` — fp16, shape [1, 1, 128, 896] (same as input)

## HVX 实现要点

- 模板 B（受限域）
- Kernel: `float xv = (float)in(n,h,w,d); out(n,h,w,d) = <func>(xv);`
- 受限域 op，输入必须在 [-1, 1] 范围内。用 asinf() 实现。

## 验收标准

- cosine ≥ 0.999
- max_abs ≤ 1e-3
