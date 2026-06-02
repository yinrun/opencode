# ElementWiseDiv Op Spec

## 数学定义
y = a / b (element-wise, with broadcasting)

## 输入输出
| 名称 | Shape | DType |
|------|-------|-------|
| a | [1, 1, 128, 896] | float16 |
| b | [1, 1, 1, 1] or [1, 1, 128, 896] | float16 |
| output | [1, 1, 128, 896] | float16 |

## 用途
Attention score / sqrt(d_k); Softmax exp(x) / sum

## 数值陷阱
- 除零保护: b=0 时输出 0 或 inf
- fp16 小数除法精度

## 验收标准
- cosine ≥ 0.999, max_abs ≤ 1e-3
