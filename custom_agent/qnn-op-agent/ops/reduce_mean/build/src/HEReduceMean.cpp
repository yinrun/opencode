// HEReduceMean - reduce-mean along last axis with keepdim
// Input  shape [B, H, W, K]   (e.g. [1,1,128,896])
// Output shape [B, H, W, 1]
//
// Numerical: fp32 accumulator + fp32 division by K, single fp16 round on store.
// Pattern: clone of HEReduceSum (L8) with /K added before store.
// Matches numpy reference: x.astype(f32).mean(axis=-1, keepdim).astype(f16),
// which numpy implements as sum(f32)/K(f32) then cast f16 — same path.

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_HEReduceMean);

template <typename T_Ttype>
int heReduceMeanImpl(T_Ttype &out, const T_Ttype &in);

// Quant fallback - REQUIRED for HTP graph finalize (avoids err 1002)
DEF_PACKAGE_OP((heReduceMeanImpl<Tensor>), "HEReduceMean")

// fp32
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heReduceMeanImpl<PlainFloatTensor>),
                                  "HEReduceMean", SNAIL, Flags::RESOURCE_HVX)

// fp16
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heReduceMeanImpl<PlainFloat16Tensor>),
                                  "HEReduceMean", SNAIL, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("HEReduceMean", "in0"),
                      Flat("*"), MainMemory("*"))

template <typename T_Ttype>
int heReduceMeanImpl(T_Ttype &out, const T_Ttype &in) {
    const Idx B = in.dim(0);
    const Idx H = in.dim(1);
    const Idx W = in.dim(2);
    const Idx K = in.dim(3);
    const float Kf = (float)K;

    for (Idx bi = 0; bi < B; bi++) {
      for (Idx hi = 0; hi < H; hi++) {
        for (Idx wi = 0; wi < W; wi++) {
          float acc = 0.0f;
          for (Idx ki = 0; ki < K; ki++) {
            acc += (float)in(bi, hi, wi, ki);
          }
          // Use fp32 division (acc / K) rather than acc * (1/K) to match
          // numpy's sum/N path bit-exactly. 1/896 is not representable in fp32.
          out(bi, hi, wi, 0) = acc / Kf;
        }
      }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_HEReduceMean);
