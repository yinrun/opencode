# ReduceMean Op Spec

## 数学定义
y = mean(x, axis=-1, keepdim=True) = sum(x) / N

## 输入输出
| 名称 | Shape | DType |
|------|-------|-------|
| input | [1, 1, 128, 896] | float16 |
| output | [1, 1, 128, 1] | float16 |

## 用途
LayerNorm mean; 统计计算

## 数值陷阱
- 同 ReduceSum: fp32 累加器
- 除以 N 用 fp32 再转回 fp16

## 验收标准
- cosine ≥ 0.999, max_abs ≤ 1e-3
