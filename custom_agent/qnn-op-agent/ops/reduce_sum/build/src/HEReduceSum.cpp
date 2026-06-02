// HEReduceSum - reduce-sum along last axis with keepdim
// Input  shape [B, H, W, K]   (e.g. [1,1,128,896])
// Output shape [B, H, W, 1]
//
// Numerical: fp32 accumulator (fp16 sum of 896 values can overflow / lose ULPs).
// Pattern: silu (single-input) + matmul (output_shape != input_shape).
// AUTOSPLIT not added on first try — work is K=896 mac-equivalents per output
// element, total 128*896 = 115k loads, well under matmul N=4864 (~622k) which
// needed splitting. If err 1002 appears, add AUTOSPLIT along W (dim 2).

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_HEReduceSum);

template <typename T_Ttype>
int heReduceSumImpl(T_Ttype &out, const T_Ttype &in);

// Quant fallback - REQUIRED for HTP graph finalize (avoids err 1002)
DEF_PACKAGE_OP((heReduceSumImpl<Tensor>), "HEReduceSum")

// fp32
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heReduceSumImpl<PlainFloatTensor>),
                                  "HEReduceSum", SNAIL, Flags::RESOURCE_HVX)

// fp16
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heReduceSumImpl<PlainFloat16Tensor>),
                                  "HEReduceSum", SNAIL, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("HEReduceSum", "in0"),
                      Flat("*"), MainMemory("*"))

template <typename T_Ttype>
int heReduceSumImpl(T_Ttype &out, const T_Ttype &in) {
    const Idx B = in.dim(0);
    const Idx H = in.dim(1);
    const Idx W = in.dim(2);
    const Idx K = in.dim(3);

    for (Idx bi = 0; bi < B; bi++) {
      for (Idx hi = 0; hi < H; hi++) {
        for (Idx wi = 0; wi < W; wi++) {
          float acc = 0.0f;
          for (Idx ki = 0; ki < K; ki++) {
            acc += (float)in(bi, hi, wi, ki);
          }
          out(bi, hi, wi, 0) = acc;
        }
      }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_HEReduceSum);
