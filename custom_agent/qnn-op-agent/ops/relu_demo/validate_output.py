#!/usr/bin/env python3
"""Validate Relu FP16 output from device against reference.

Usage: python validate_output.py <device_output.raw>
"""
import numpy as np
import sys
import os

SHAPE = (1, 8, 1, 64)
DTYPE = np.float16

def load_raw(path):
    data = np.fromfile(path, dtype=DTYPE)
    assert data.size == np.prod(SHAPE), f"Size mismatch: got {data.size}, expected {np.prod(SHAPE)}"
    return data.reshape(SHAPE)

def main():
    test_dir = os.path.join(os.path.dirname(__file__), "test_data")
    ref_path = os.path.join(test_dir, "reference_output.raw")
    input_path = os.path.join(test_dir, "input.raw")

    if len(sys.argv) < 2:
        print("Usage: python validate_output.py <device_output.raw>")
        sys.exit(1)

    device_path = sys.argv[1]

    # Load data
    x = load_raw(input_path)
    ref = load_raw(ref_path)
    out = load_raw(device_path)

    # Metrics
    diff = (out.astype(np.float32) - ref.astype(np.float32))
    max_abs_err = np.abs(diff).max()
    mean_abs_err = np.abs(diff).mean()

    # Cosine similarity
    ref_flat = ref.flatten().astype(np.float32)
    out_flat = out.flatten().astype(np.float32)
    cos_sim = np.dot(ref_flat, out_flat) / (np.linalg.norm(ref_flat) * np.linalg.norm(out_flat) + 1e-10)

    # Exact match count
    exact_match = (out == ref).sum()
    total = ref.size

    print("=" * 60)
    print("Relu FP16 Validation Results")
    print("=" * 60)
    print(f"Input shape:     {SHAPE}")
    print(f"Input range:     [{x.min():.4f}, {x.max():.4f}]")
    print(f"Output range:    [{out.min():.4f}, {out.max():.4f}]")
    print(f"Reference range: [{ref.min():.4f}, {ref.max():.4f}]")
    print("-" * 60)
    print(f"Max abs error:   {max_abs_err:.6f}")
    print(f"Mean abs error:  {mean_abs_err:.6f}")
    print(f"Cosine sim:      {cos_sim:.8f}")
    print(f"Exact match:     {exact_match}/{total} ({100*exact_match/total:.1f}%)")
    print("-" * 60)

    # Check correctness: relu should zero out negatives
    neg_mask = (x < 0)
    neg_outputs = out[neg_mask]
    pos_mask = (x >= 0)
    pos_outputs = out[pos_mask]
    pos_inputs = x[pos_mask]

    neg_all_zero = (neg_outputs == 0).all()
    pos_passthrough = (pos_outputs == pos_inputs).all()

    print(f"Negatives → 0:   {'PASS' if neg_all_zero else 'FAIL'} ({neg_mask.sum()} elements)")
    print(f"Positives pass:  {'PASS' if pos_passthrough else 'FAIL'} ({pos_mask.sum()} elements)")
    print("=" * 60)

    if max_abs_err == 0 and neg_all_zero and pos_passthrough:
        print("✅ PERFECT - bit-exact match!")
        return 0
    elif cos_sim >= 0.999 and max_abs_err <= 1e-3:
        print("✅ PASS - within tolerance")
        return 0
    else:
        print("❌ FAIL - output does not match reference")
        return 1

if __name__ == "__main__":
    sys.exit(main())
