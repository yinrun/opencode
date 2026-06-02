# Sigmoid Op Spec

## 数学定义
y = 1 / (1 + exp(-x))

## 输入输出
| 名称 | Shape | DType |
|------|-------|-------|
| input | [1, 1, 128, 4864] | float16 |
| output | [1, 1, 128, 4864] | float16 |

## 用途
SiLU 的组成部分; Gate 机制

## 数值陷阱
- x > 11: sigmoid ≈ 1.0
- x < -11: sigmoid ≈ 0.0
- 中间区域需要 polynomial 近似

## 参考
- 已有 SiLU 实现中的 sigmoid v7 模板可直接复用

## 验收标准
- cosine ≥ 0.999, max_abs ≤ 1e-3
