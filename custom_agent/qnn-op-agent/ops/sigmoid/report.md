# HESigmoid Report

## Result
- **Status**: PASS
- **Iterations**: 1 (zero patches)
- **Shape**: [1, 1, 128, 4864] fp16
- **cosine** = 0.9999999999998367
- **max_abs_err** = 2.44e-4 (≈ 1 fp16 ULP near 0.5 saturation region)
- **relative_error** = 7.86e-10

## Approach
Cloned tanh's template (per L10). Three changes only:
1. Source files renamed: `HETanh*` → `HESigmoid*`, package name `HETanhPackage` → `HESigmoidPackage`
2. Kernel one-liner: `tanhf(xv)` → `1.0f / (1.0f + expf(-xv))`
3. model dims: 896 → 4864 (sigmoid spec uses larger tensor)

Manual aarch64 link for model.so (per L7) to avoid `build_single_op_model` source overwrite.

## Why not bit-exact (vs tanh which was bit-exact)
Sigmoid involves an extra fp32 division `1/(1+...)` whose intermediate rounding diverges from numpy `1/(1+np.exp(-x))` at scattered points. Both numpy and the kernel run fp32 internally, but expf vendor implementations + division order produce ~1 ULP drift at fp16 cast time. Still well inside acceptance.

## Files
- `src/HESigmoid.cpp`, `src/HESigmoidInterface.cpp`
- `model_src/model_hesigmoid.cpp`
- `Makefile`, `device_configs/{htp,net_run}_config.json`
- Device: `/data/local/tmp/qnn_op_test/{libQnnHESigmoidModel.so,libQnnHtpSigmoid.so,htp/libQnnHtpSigmoid.so}`
