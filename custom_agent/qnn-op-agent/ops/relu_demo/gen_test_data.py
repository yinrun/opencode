#!/usr/bin/env python3
"""Generate test input and reference output for Relu FP16 demo.

Input shape: [1, 8, 1, 64] fp16
Contains mix of positive and negative values to verify relu behavior.
"""
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "test_data")
os.makedirs(OUT_DIR, exist_ok=True)

# Generate input with known pattern: mix of positive and negative
np.random.seed(42)
x = np.random.randn(1, 8, 1, 64).astype(np.float16)

# Save as raw binary (fp16)
input_path = os.path.join(OUT_DIR, "input.raw")
x.tofile(input_path)
print(f"Input saved: {input_path} ({x.nbytes} bytes, shape={x.shape})")
print(f"  range: [{x.min():.4f}, {x.max():.4f}]")
print(f"  negative count: {(x < 0).sum()} / {x.size}")

# Compute reference: relu = max(x, 0)
y_ref = np.maximum(x, np.float16(0.0))
ref_path = os.path.join(OUT_DIR, "reference_output.raw")
y_ref.tofile(ref_path)
print(f"Reference saved: {ref_path} ({y_ref.nbytes} bytes)")
print(f"  range: [{y_ref.min():.4f}, {y_ref.max():.4f}]")
print(f"  zero count: {(y_ref == 0).sum()} / {y_ref.size}")

# Create input_list.txt for qnn-net-run
input_list_path = os.path.join(OUT_DIR, "input_list.txt")
with open(input_list_path, "w") as f:
    f.write("input.raw\n")
print(f"Input list: {input_list_path}")

print("\nDone! Test data ready.")
