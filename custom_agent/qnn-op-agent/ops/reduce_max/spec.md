# ReduceMax Op Spec

## 数学定义
y = max(x, axis=-1, keepdim=True)

## 输入输出
| 名称 | Shape | DType |
|------|-------|-------|
| input | [1, 1, 128, 128] | float16 |
| output | [1, 1, 128, 1] | float16 |

## 用途
Softmax 数值稳定: max subtraction

## HVX 实现
- 用 Q6_Vhf_vmax_VhfVhf 做向量内 max
- 然后 tree reduce 跨 vector

## 验收标准
- bit-exact
