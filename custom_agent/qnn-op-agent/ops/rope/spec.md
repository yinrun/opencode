# RoPE (Rotary Position Embedding) Op Spec

## 数学定义

```
q_rot[..., 0::2] = q[..., 0::2] * cos - q[..., 1::2] * sin
q_rot[..., 1::2] = q[..., 1::2] * cos + q[..., 0::2] * sin
```

同样应用于 k。频率计算：
```
theta_i = base^(-2i/d), i = 0, 1, ..., d/2-1
freqs = position * theta_i
cos = cos(freqs), sin = sin(freqs)
```

其中 `base = 1000000`（Qwen3 默认 rope_theta）。

## 模型参数（Qwen3 0.6B）

| 参数 | 值 |
|------|-----|
| num_heads | 14 |
| num_kv_heads | 2 |
| head_dim | 64 |
| rope_theta | 1000000 |
| max_position | 32768 |

## 输入输出

| 名称 | Shape (decode, pos=P) | Shape (prefill S=128) | DType |
|------|----------------------|----------------------|-------|
| q | [1, 14, 1, 64] | [1, 14, 128, 64] | float16 |
| k | [1, 2, 1, 64] | [1, 2, 128, 64] | float16 |
| cos_cache | [1, 1, 1, 64] | [1, 1, 128, 64] | float16 |
| sin_cache | [1, 1, 1, 64] | [1, 1, 128, 64] | float16 |
| q_out | [1, 14, 1, 64] | [1, 14, 128, 64] | float16 |
| k_out | [1, 2, 1, 64] | [1, 2, 128, 64] | float16 |

## HVX 实现要点

1. **偶/奇 lane 拆分**：head_dim=64 → 32 对偶奇元素，每对做旋转
2. **cos/sin 预计算**：作为输入传入（不在 kernel 内计算 theta）
3. **向量化**：64 个 fp16 元素 = 1 个 HVX vector，天然对齐
4. **多头并行**：外层循环 over heads，内层 HVX 处理 head_dim
5. **Layout**：PlainFloat16Tensor（head_dim 维连续）

## 数值陷阱

- cos/sin 值域 [-1, 1]，fp16 精度足够
- 乘法 q*cos 和 q*sin 的中间结果在 fp16 range 内（输入已归一化）
- 偶奇拆分的 shuffle 操作：用 `Q6_Vh_vshuff_Vh` 或手动 interleave

## 验收标准

- cosine ≥ 0.999 vs QNN 内置 RoPE（如有）或 PyTorch 参考
- max_abs_error ≤ 1e-3
- e2e ppl delta ≤ 1%

## 参考

- Qwen3 源码：`Qwen3RotaryEmbedding` + `apply_rotary_pos_emb()`
- 注意：Qwen3 用 half-rotary（前半 rotate，后半不动）还是 full-rotary 需确认
