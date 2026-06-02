# GELU Op Spec

## 数学定义
y = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

或近似: y = x * sigmoid(1.702 * x)

## 输入输出
| 名称 | Shape | DType |
|------|-------|-------|
| input | [1, 1, 128, 4864] | float16 |
| output | [1, 1, 128, 4864] | float16 |

## 用途
GPT/BERT 系列 MLP 激活函数

## 数值陷阱
- tanh 版本需要高精度 tanh 实现
- sigmoid 近似版本更适合 HVX (复用 SiLU 的 sigmoid)
- x³ 在 fp16 下大值会溢出

## 验收标准
- cosine ≥ 0.999, max_abs ≤ 1e-3
