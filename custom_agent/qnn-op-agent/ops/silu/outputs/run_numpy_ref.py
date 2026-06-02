"""SiLU test input + numpy reference, fp16 (matches HTP PlainFloat16Tensor)."""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
in_path  = ROOT / "baseline/test_inputs/silu/input.raw"
out_path = ROOT / "ops/silu/outputs/numpy_ref.raw"

shape = (1, 1, 128, 4864)
rng = np.random.default_rng(42)
x = rng.standard_normal(shape).astype(np.float16)
in_path.parent.mkdir(parents=True, exist_ok=True)
x.tofile(in_path)

xf = x.astype(np.float32)
y = (xf / (1.0 + np.exp(-xf))).astype(np.float16)
out_path.parent.mkdir(parents=True, exist_ok=True)
y.tofile(out_path)

print(f"input  : shape={x.shape} dtype={x.dtype} bytes={in_path.stat().st_size}")
print(f"output : shape={y.shape} dtype={y.dtype} bytes={out_path.stat().st_size}")
