# reduce_mean — Report

## Result
- **Status**: PASS
- **Iterations**: 1
- **Cosine similarity**: 0.9999999999936685
- **Max abs error**: 0.0
- **Relative error**: 0.0
- **Bit-exact**: YES (matches numpy fp32 sum/N → fp16 cast path exactly)

## Shapes
- Input  : [1, 1, 128, 896] float16
- Output : [1, 1, 128, 1]   float16

## Approach
Direct clone of L8 reduce_sum / L9 reduce_max template:
- Single-input op, output_shape != input_shape, K = 896 (last axis).
- fp32 accumulator over K, then `acc / Kf` (fp32 division), single fp16 round on store.

## Key code
```cpp
template <typename T_Ttype>
int heReduceMeanImpl(T_Ttype &out, const T_Ttype &in) {
    const Idx B = in.dim(0), H = in.dim(1), W = in.dim(2), K = in.dim(3);
    const float Kf = (float)K;
    for (Idx bi=0; bi<B; bi++)
      for (Idx hi=0; hi<H; hi++)
        for (Idx wi=0; wi<W; wi++) {
          float acc = 0.0f;
          for (Idx ki=0; ki<K; ki++) acc += (float)in(bi,hi,wi,ki);
          out(bi,hi,wi,0) = acc / Kf;
        }
    return GraphStatus::Success;
}
```

## Why fp32 division (not multiply by 1/K)
1/896 is not representable exactly in fp32. `acc / Kf` matches numpy's
`sum(f32) / N(f32)` path bit-exactly. Multiplying by a precomputed 1/K
would introduce one extra fp32 round and break bit-exactness (still well
inside acceptance bounds, but no reason to deviate from numpy reference).

## Patches
0 patches. Template (silu single-input + reduce_sum output_shape != input_shape)
applied directly with only:
- op rename (HEReduceSum → HEReduceMean)
- kernel inner change: `out = acc` → `out = acc / Kf`

## Files
- src/HEReduceMean.cpp
- src/HEReduceMeanInterface.cpp
- model_src/model_hereducemean.cpp (input dims 896, output dims 1, graph_name HEReduceMean_graph)
- HEReduceMean.xml
- Makefile (PACKAGE_NAME=HEReduceMeanPackage, LIBRARY_NAME=libQnnHtpReduceMean.so)
- device_configs/{htp_config,net_run_config}.json
- input_list.txt
