# LOG — 算子规格

## 数学定义

y = ln(x) (自然对数)

## QNN 对应算子

`ElementWiseLog`

## 输入定义域

x > 0，x ≤ 0 输出 -inf 或 NaN

## 输入输出

- **Input**: `in0` — fp16, shape [1, 1, 128, 896]
- **Output**: `out` — fp16, shape [1, 1, 128, 896] (same as input)

## HVX 实现要点

- 模板 B（受限域）
- Kernel: `float xv = (float)in(n,h,w,d); out(n,h,w,d) = <func>(xv);`
- 受限域 op，输入必须 > 0。用 logf() 实现。测试输入用 uniform(0.01, 10.0)。

## 验收标准

- cosine ≥ 0.999
- max_abs ≤ 1e-3
