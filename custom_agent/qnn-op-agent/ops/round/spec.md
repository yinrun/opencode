# ROUND — 算子规格

## 数学定义

y = round(x) (四舍五入到最近整数)

## QNN 对应算子

`ElementWiseRound`

## 输入定义域

全实数域

## 输入输出

- **Input**: `in0` — fp16, shape [1, 1, 128, 896]
- **Output**: `out` — fp16, shape [1, 1, 128, 896] (same as input)

## HVX 实现要点

- 模板 A（全域 unary）
- Kernel: `float xv = (float)in(n,h,w,d); out(n,h,w,d) = <func>(xv);`
- roundf() 实现（banker's rounding: 0.5 取偶数）。

## 验收标准

- cosine ≥ 0.999
- max_abs ≤ 1e-3
