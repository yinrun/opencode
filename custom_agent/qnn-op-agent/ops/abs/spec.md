# Abs Op Spec

## 数学定义
y = |x| (element-wise)

## 输入输出
| 名称 | Shape | DType |
|------|-------|-------|
| input | [1, 1, 128, 896] | float16 |
| output | [1, 1, 128, 896] | float16 |

## 用途
误差计算、梯度裁剪

## 验收标准
- bit-exact (清除符号位即可)
