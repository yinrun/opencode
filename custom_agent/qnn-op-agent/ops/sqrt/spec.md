# Sqrt Op Spec

## 数学定义
y = sqrt(x) (element-wise)

## 输入输出
| 名称 | Shape | DType |
|------|-------|-------|
| input | [1, 1, 128, 1] | float16 |
| output | [1, 1, 128, 1] | float16 |

## 用途
Normalization 中间步骤

## 数值陷阱
- x<0 → NaN
- x=0 → 0

## 验收标准
- cosine ≥ 0.999, max_abs ≤ 1e-3
