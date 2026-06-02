# Neg Op Spec

## 数学定义
y = -x (element-wise)

## 输入输出
| 名称 | Shape | DType |
|------|-------|-------|
| input | [1, 1, 128, 896] | float16 |
| output | [1, 1, 128, 896] | float16 |

## 用途
Causal mask 中的 -inf 生成

## 验收标准
- bit-exact (翻转符号位即可)
