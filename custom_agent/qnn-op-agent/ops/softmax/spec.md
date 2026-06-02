# Softmax Op Spec

## 数学定义

```
y = exp(x - max(x, dim=-1, keepdim=True)) / sum(exp(x - max(x, dim=-1, keepdim=True)), dim=-1, keepdim=True)
```

数值稳定形式（max-subtract）。

## 模型参数（Qwen3 0.6B）

| 参数 | 值 |
|------|-----|
| num_heads | 14 |
| head_dim | 64 |
| max_seq_len | 32768 |

## 输入输出

| 名称 | Shape (decode, kv_len=K) | Shape (prefill S=128) | DType |
|------|--------------------------|----------------------|-------|
| input (attn scores) | [1, 14, 1, K] | [1, 14, 128, 128] | float16 |
| output | [1, 14, 1, K] | [1, 14, 128, 128] | float16 |

注意：K（kv cache 长度）在 decode 时逐步增长。V1 测试用固定 K=128。

## HVX 实现要点

1. **Row-wise reduce**：对最后一维做 max 和 sum，类似 RMSNorm 的 reduce 模式
2. **三步流程**：
   - Step 1: row-wise max（fp32 累加）
   - Step 2: exp(x - max)（Schraudolph + polynomial，参考 sigmoid 项目）
   - Step 3: row-wise sum + reciprocal + 逐元素乘
3. **数值稳定性**：max-subtract 是必须的，否则 exp 溢出
4. **Layout**：PlainFloat16Tensor（reduce 维连续）

## 数值陷阱

- exp 在 fp16 下 x > 11 就溢出 → max-subtract 保证 exp 参数 ≤ 0
- sum 累加需要 fp32（D=128 时 fp16 可能还行，但 K 可能很大）
- reciprocal 用 Newton iteration（参考 sigmoid 项目的 recip）
- 当 K 很大时（如 8192），reduce 的 fp32 累加器精度仍然足够

## 验收标准

- cosine ≥ 0.999 vs QNN 内置 Softmax
- max_abs_error ≤ 1e-3
- e2e ppl delta ≤ 1%

## 参考

- QNN SDK 有内置 Softmax op（OpSoftmax）
- 已有经验：sigmoid 项目的 exp 实现 + L2Norm 项目的 row-wise reduce
- 难度 medium-hard：需要 2 次 reduce（max + sum）+ exp + recip
