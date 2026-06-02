// HEReduceMax - reduce-max along last axis with keepdim
// Input  shape [B, H, W, K]   (e.g. [1,1,128,128])
// Output shape [B, H, W, 1]
//
// Numerical: fp32 max accumulator (lossless promote, single fp16 round on
// store). For max, fp32 vs fp16 accumulator is mathematically identical, but
// the fp32 cast keeps the pattern aligned with reduce_sum / softmax, and
// avoids any subnormal-flush surprises if upstream pipelines are touched.
// Pattern: silu (single-input) + reduce_sum (output_shape != input_shape).

#include <cmath>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_HEReduceMax);

template <typename T_Ttype>
int heReduceMaxImpl(T_Ttype &out, const T_Ttype &in);

// Quant fallback - REQUIRED for HTP graph finalize (avoids err 1002)
DEF_PACKAGE_OP((heReduceMaxImpl<Tensor>), "HEReduceMax")

// fp32
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heReduceMaxImpl<PlainFloatTensor>),
                                  "HEReduceMax", SNAIL, Flags::RESOURCE_HVX)

// fp16
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heReduceMaxImpl<PlainFloat16Tensor>),
                                  "HEReduceMax", SNAIL, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("HEReduceMax", "in0"),
                      Flat("*"), MainMemory("*"))

template <typename T_Ttype>
int heReduceMaxImpl(T_Ttype &out, const T_Ttype &in) {
    const Idx B = in.dim(0);
    const Idx H = in.dim(1);
    const Idx W = in.dim(2);
    const Idx K = in.dim(3);

    for (Idx bi = 0; bi < B; bi++) {
      for (Idx hi = 0; hi < H; hi++) {
        for (Idx wi = 0; wi < W; wi++) {
          float acc = -INFINITY;
          for (Idx ki = 0; ki < K; ki++) {
            float v = (float)in(bi, hi, wi, ki);
            if (v > acc) acc = v;
          }
          out(bi, hi, wi, 0) = acc;
        }
      }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_HEReduceMax);
