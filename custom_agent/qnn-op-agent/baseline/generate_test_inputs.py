#!/usr/bin/env python3
"""
Generate deterministic test inputs for 5 QNN ops (Qwen3 0.6B dimensions).

Uses numpy with seed=42 to produce reproducible fp16 .raw files.
Shapes are based on prefill mode (seq_len=128) from each op's spec.md.

Model: Qwen3 0.6B
  hidden_dim = 896
  intermediate_dim = 4864
  num_heads = 14
  num_kv_heads = 2
  head_dim = 64
  seq_len = 128
  rope_theta = 1000000
  eps = 1e-6
"""

import json
import os
from datetime import datetime, timezone

import numpy as np

# === Configuration ===
SEED = 42
DTYPE = np.float16
OUTPUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_inputs")

# Qwen3 0.6B model dimensions
HIDDEN_DIM = 896
INTERMEDIATE_DIM = 4864
NUM_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64
SEQ_LEN = 128


def save_raw(tensor: np.ndarray, path: str):
    """Save tensor as little-endian .raw binary file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor.astype(DTYPE).tofile(path)
    print(f"  Saved: {path} | shape={tensor.shape} | bytes={tensor.nbytes}")


def generate_rmsnorm(rng: np.random.Generator, output_dir: str) -> dict:
    """Generate RMSNorm inputs: x [1, 128, 896], weight [896]."""
    x = rng.standard_normal((1, SEQ_LEN, HIDDEN_DIM)).astype(DTYPE)
    # Weight is typically close to 1.0 (initialized as ones, then trained)
    w = rng.uniform(0.5, 1.5, size=(HIDDEN_DIM,)).astype(DTYPE)

    save_raw(x, os.path.join(output_dir, "input_x.raw"))
    save_raw(w, os.path.join(output_dir, "weight.raw"))

    return {
        "input_x": [1, SEQ_LEN, HIDDEN_DIM],
        "weight": [HIDDEN_DIM],
    }


def generate_swiglu(rng: np.random.Generator, output_dir: str) -> dict:
    """Generate SwiGLU inputs: gate [1, 128, 4864], up [1, 128, 4864]."""
    gate = rng.standard_normal((1, SEQ_LEN, INTERMEDIATE_DIM)).astype(DTYPE)
    up = rng.standard_normal((1, SEQ_LEN, INTERMEDIATE_DIM)).astype(DTYPE)

    save_raw(gate, os.path.join(output_dir, "gate.raw"))
    save_raw(up, os.path.join(output_dir, "up.raw"))

    return {
        "gate": [1, SEQ_LEN, INTERMEDIATE_DIM],
        "up": [1, SEQ_LEN, INTERMEDIATE_DIM],
    }


def generate_rope(rng: np.random.Generator, output_dir: str) -> dict:
    """Generate RoPE inputs: q [1,14,128,64], k [1,2,128,64], cos/sin [1,1,128,64]."""
    q = rng.standard_normal((1, NUM_HEADS, SEQ_LEN, HEAD_DIM)).astype(DTYPE)
    k = rng.standard_normal((1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)).astype(DTYPE)

    # Pre-computed cos/sin cache for positions 0..127
    # theta_i = base^(-2i/d), freqs = pos * theta_i
    positions = np.arange(SEQ_LEN, dtype=np.float32)
    dim_indices = np.arange(0, HEAD_DIM, 2, dtype=np.float32)
    theta = 1000000.0 ** (-dim_indices / HEAD_DIM)
    # freqs shape: [SEQ_LEN, HEAD_DIM/2]
    freqs = np.outer(positions, theta)
    # Expand to full head_dim by repeating (cos/sin apply to pairs)
    cos_cache = np.cos(freqs).astype(np.float32)
    sin_cache = np.sin(freqs).astype(np.float32)
    # Repeat each value for the pair: [SEQ_LEN, HEAD_DIM/2] -> [SEQ_LEN, HEAD_DIM]
    cos_full = np.repeat(cos_cache, 2, axis=-1).astype(DTYPE)
    sin_full = np.repeat(sin_cache, 2, axis=-1).astype(DTYPE)
    # Reshape to [1, 1, SEQ_LEN, HEAD_DIM]
    cos_full = cos_full.reshape(1, 1, SEQ_LEN, HEAD_DIM)
    sin_full = sin_full.reshape(1, 1, SEQ_LEN, HEAD_DIM)

    save_raw(q, os.path.join(output_dir, "q.raw"))
    save_raw(k, os.path.join(output_dir, "k.raw"))
    save_raw(cos_full, os.path.join(output_dir, "cos_cache.raw"))
    save_raw(sin_full, os.path.join(output_dir, "sin_cache.raw"))

    return {
        "q": [1, NUM_HEADS, SEQ_LEN, HEAD_DIM],
        "k": [1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM],
        "cos_cache": [1, 1, SEQ_LEN, HEAD_DIM],
        "sin_cache": [1, 1, SEQ_LEN, HEAD_DIM],
    }


def generate_softmax(rng: np.random.Generator, output_dir: str) -> dict:
    """Generate Softmax inputs: attn_scores [1, 14, 128, 128]."""
    # Attention scores are typically in range [-5, 5] before softmax
    attn_scores = rng.standard_normal((1, NUM_HEADS, SEQ_LEN, SEQ_LEN)).astype(DTYPE)

    save_raw(attn_scores, os.path.join(output_dir, "input.raw"))

    return {
        "input": [1, NUM_HEADS, SEQ_LEN, SEQ_LEN],
    }


def generate_matmul(rng: np.random.Generator, output_dir: str) -> dict:
    """Generate MatMul inputs: A [128, 896], B [896, 4864] (Gate proj, prefill)."""
    # Activation matrix (dynamic)
    a = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(DTYPE)
    # Weight matrix (static, typically smaller magnitude)
    b = (rng.standard_normal((HIDDEN_DIM, INTERMEDIATE_DIM)) * 0.02).astype(DTYPE)

    save_raw(a, os.path.join(output_dir, "a.raw"))
    save_raw(b, os.path.join(output_dir, "b.raw"))

    return {
        "a": [SEQ_LEN, HIDDEN_DIM],
        "b": [HIDDEN_DIM, INTERMEDIATE_DIM],
    }


def main():
    print(f"Generating deterministic test inputs (seed={SEED})...")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Model: Qwen3 0.6B")
    print(f"  hidden_dim={HIDDEN_DIM}, intermediate_dim={INTERMEDIATE_DIM}")
    print(f"  num_heads={NUM_HEADS}, num_kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
    print(f"  seq_len={SEQ_LEN}")
    print()

    rng = np.random.default_rng(SEED)

    ops_shapes = {}

    # Generate inputs for each op
    generators = {
        "rmsnorm": generate_rmsnorm,
        "swiglu": generate_swiglu,
        "rope": generate_rope,
        "softmax": generate_softmax,
        "matmul": generate_matmul,
    }

    for op_name, gen_fn in generators.items():
        print(f"[{op_name}]")
        op_dir = os.path.join(OUTPUT_BASE, op_name)
        shapes = gen_fn(rng, op_dir)
        ops_shapes[op_name] = {"shapes": shapes, "dtype": "float16"}
        print()

    # Write metadata.json
    metadata = {
        "model": "Qwen3-0.6B",
        "model_params": {
            "hidden_dim": HIDDEN_DIM,
            "intermediate_dim": INTERMEDIATE_DIM,
            "num_heads": NUM_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "seq_len": SEQ_LEN,
            "rope_theta": 1000000,
            "eps": 1e-6,
        },
        "seed": SEED,
        "dtype": "float16",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ops_tested": list(generators.keys()),
        "test_inputs": ops_shapes,
    }

    metadata_path = os.path.join(os.path.dirname(OUTPUT_BASE), "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata written to: {metadata_path}")
    print("Done!")


if __name__ == "__main__":
    main()
