# HEReduceMax — report

## Spec
- input  [1,1,128,128] fp16
- output [1,1,128,1]   fp16
- y = max(x, axis=-1, keepdim=True)

## Result (Phase 5)
- cosine_similarity = 0.9999999999999988
- max_abs_error     = 0.0
- relative_error    = 0.0
- shapes_match      = true
- bit-exact match against numpy fp32 max + fp16 cast

## Iterations
- 1 (no patches needed)

## Implementation
- src/HEReduceMax.cpp — single-input op, 3 variants (Tensor / PlainFloatTensor / PlainFloat16Tensor)
- src/HEReduceMaxInterface.cpp — standard `HEReduceMaxPackageInterfaceProvider` export
- model_src/model_hereducemax.cpp — single-op graph, output dims hardcoded to {1,1,128,1}, `getGraphInfoFromModels()` (no `model.finalize()`), `DO_GRAPH_NODE_VALIDATIONS=0`
- Makefile — hexagon-v81 + aarch64-android + x86_64-linux-clang

## Run command
```
qnn-net-run \
  --backend libQnnHtp.so \
  --model libQnnHEReduceMaxModel.so \
  --input_list input_list.txt \
  --output_dir out \
  --op_packages <arm>.so:HEReduceMaxPackageInterfaceProvider:CPU,<hex>.so:HEReduceMaxPackageInterfaceProvider:HTP \
  --config_file net_run_config.json \
  --use_native_input_files --use_native_output_files
```

Output: `out/Result_0/out_native.raw`.

## Lesson
Reduce-family unary ops (sum, max, mean, min) are isomorphic — same DEF_TENSOR_PROPERTIES,
same model.cpp output-shape change, only the inner-loop reduction operator and identity
element differ. Confirmed L8 generalises to max with 0 patches.
