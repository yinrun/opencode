# Tanh Op Spec

## 数学定义
y = tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)

或: y = 2*sigmoid(2x) - 1

## 输入输出
| 名称 | Shape | DType |
|------|-------|-------|
| input | [1, 1, 128, 896] | float16 |
| output | [1, 1, 128, 896] | float16 |

## 用途
GELU 精确版; LSTM gate

## 数值陷阱
- |x| > 5 时 tanh ≈ ±1 (fp16 精度足够)
- 可复用 sigmoid: tanh(x) = 2*sigmoid(2x) - 1

## 验收标准
- cosine ≥ 0.999, max_abs ≤ 1e-3
