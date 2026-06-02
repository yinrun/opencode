# HENeg Op Report

## Result
- **Op**: HENeg (package `HENegPackage`)
- **Shape**: fp16 [1, 1, 128, 896]
- **cosine** = 1.0000000000000002
- **max_abs_err** = 0.0
- **mean_abs_err / relative_err** = 0.0
- **PASS** (bit-exact, matches numpy `(-x).astype(f16)`)
- **Iterations**: 1 (after re-applying L7 model_*.cpp patch)

## Kernel
```cpp
template <typename T_Ttype>
int heNegImpl(T_Ttype &out, const T_Ttype &in) {
    out.set_dims(in);
    for (Idx n = 0; n < in.dim(0); n++)
      for (Idx h = 0; h < in.dim(1); h++)
        for (Idx w = 0; w < in.dim(2); w++)
          for (Idx d = 0; d < in.dim(3); d++) {
            float xv = in(n, h, w, d);
            out(n, h, w, d) = -xv;
          }
    return GraphStatus::Success;
}
```
Standard scalar loop. fp16 → fp32 promote → negate → fp16 round-on-store.
Bit-exact because `-x` cannot lose precision (same magnitude, flipped sign bit).

## Key Patch
Source `model_heneg.cpp` had been left in template-default state (`model.finalize()` + `DO_GRAPH_NODE_VALIDATIONS 1`) from a previous `build_single_op_model` call. First run failed with the L1/L7 double-finalize segfault:
```
QnnModel::finalize() finalizing graph failed.
Graph Prepare failure / Segmentation fault
```
Patched to `getGraphInfoFromModels(&model, 1, graphsInfo)` + `DO_GRAPH_NODE_VALIDATIONS 0`, manually re-linked aarch64 model.so with `clang++ --target=aarch64-none-linux-android21`, re-pushed → composeGraph passed first try.

## Lesson Confirmation
L7 reproduced exactly: any time you see a `model_*.cpp` containing `model.finalize(graphsInfo, ...)` you can be 100% sure it will sig11 on device. Always grep before deploying.
