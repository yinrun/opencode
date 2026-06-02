# ReduceSum Op Spec

## 数学定义
y = sum(x, axis=-1, keepdim=True)

## 输入输出
| 名称 | Shape | DType |
|------|-------|-------|
| input | [1, 1, 128, 896] | float16 |
| output | [1, 1, 128, 1] | float16 |

## 用途
RMSNorm sum-of-squares; Softmax denominator

## 数值陷阱
- fp16 累加 896 个值可能溢出 → 需要 fp32 累加器
- Tree reduce 比顺序累加更精确

## 验收标准
- cosine ≥ 0.999, max_abs ≤ 1e-3
