# ElementWiseMul Op Spec

## 数学定义
y = a * b (element-wise, with broadcasting)

## 输入输出
| 名称 | Shape | DType |
|------|-------|-------|
| a | [1, 1, 128, 896] | float16 |
| b | [1, 1, 128, 896] | float16 |
| output | [1, 1, 128, 896] | float16 |

## 用途
RoPE 中 x*cos, x*sin; attention scale

## 验收标准
- cosine ≥ 0.999, max_abs ≤ 1e-3
