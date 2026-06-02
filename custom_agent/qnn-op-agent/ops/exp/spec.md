# Exp Op Spec

## 数学定义
y = exp(x) (element-wise)

## 输入输出
| 名称 | Shape | DType |
|------|-------|-------|
| input | [1, 1, 128, 128] | float16 |
| output | [1, 1, 128, 128] | float16 |

## 用途
Softmax 中 exp(x - max)

## 数值陷阱
- fp16 exp 溢出: x > 11.09 → inf
- 需要先减 max 再 exp
- 可用 Schraudolph 近似或 polynomial

## 验收标准
- cosine ≥ 0.999, max_abs ≤ 1e-3
