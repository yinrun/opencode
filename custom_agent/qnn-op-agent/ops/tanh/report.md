# HETanh Op Report

## Spec
- Op: HETanh — element-wise unary tanh
- Math: y = tanh(x)
- Shape: input [1,1,128,896] fp16 → output [1,1,128,896] fp16

## Result
| Metric | Value |
|--------|-------|
| cosine_similarity | 1.0 |
| max_abs_error | 0.0 |
| relative_error | 0.0 |
| bit_exact | yes |
| iterations | 1 |

## Implementation
- Reused gelu pattern verbatim — element-wise unary, full real-line domain (no input clamping needed)
- Kernel: fp16 → fp32 promote → `tanhf` → fp16 round
- Variants registered: `Tensor` (quant fallback) + `PlainFloatTensor` + `PlainFloat16Tensor`
- DEF_TENSOR_PROPERTIES: `Op("HETanh", "in0"), Flat("*"), MainMemory("*")`

## Build
- hexagon-v81: 63 KB
- aarch64-android: 1.25 MB
- x86_64: built (unused, model.so linked from aarch64 path)
- model.so manually linked with QnnModel.cpp + QnnWrapperUtils.cpp + QnnModelPal.cpp (skipping `qnn-runner_build_single_op_model` to keep `getGraphInfoFromModels()` + `DO_GRAPH_NODE_VALIDATIONS 0` patch — L7)

## Patches
None — 1-shot pass.

## Numerical Path
```
device kernel:  fp16 in -> fp32 promote -> tanhf -> fp16 store
numpy reference: fp16 in -> fp32 cast -> np.tanh -> fp16 cast
```
Identical paths → bit-exact.

## Lesson
See L10 in AGENTS.md.
