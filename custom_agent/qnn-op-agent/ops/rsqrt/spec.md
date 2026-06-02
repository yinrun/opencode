# Rsqrt Op Spec

## 数学定义
y = 1 / sqrt(x) (element-wise)

## 输入输出
| 名称 | Shape | DType |
|------|-------|-------|
| input | [1, 1, 128, 1] | float16 |
| output | [1, 1, 128, 1] | float16 |

## 用途
RMSNorm 中 rsqrt(mean(x²) + eps)

## 数值陷阱
- x=0 → inf
- 负数输入 → NaN
- Quake fast rsqrt + Newton iteration 可提高精度

## 验收标准
- cosine ≥ 0.999, max_abs ≤ 1e-3
