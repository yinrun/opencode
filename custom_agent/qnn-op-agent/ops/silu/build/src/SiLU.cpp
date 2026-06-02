// HESiLU - canonical Relu pattern, fp32 verified cosine 1.0
// fp16 variant included for forward compatibility but currently dispatches incorrectly
// (same issue as SDK canonical Relu.fp16 example - needs converter-generated model.cpp)
#include <cmath>
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_SiLU);

template <typename T_Ttype>
int siluImpl(T_Ttype &out, const T_Ttype &in);

// Quant fallback - REQUIRED for HTP to compile graph
DEF_PACKAGE_OP((siluImpl<Tensor>), "HESiLU")

// fp32 - verified working (cosine 1.0)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((siluImpl<PlainFloatTensor>),   "HESiLU", SNAIL, Flags::RESOURCE_HVX)

// fp16 - included but currently broken (same issue as canonical ReluFp16)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((siluImpl<PlainFloat16Tensor>), "HESiLU", SNAIL, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("HESiLU", "in0"), Flat("*"), MainMemory("*"))

template <typename T_Ttype>
int siluImpl(T_Ttype &out, const T_Ttype &in) {
    out.set_dims(in);
    for (Idx b = 0; b < in.dim(0); b++) {
      for (Idx h = 0; h < in.dim(1); h++) {
        for (Idx w = 0; w < in.dim(2); w++) {
          for (Idx d = 0; d < in.dim(3); d++) {
            float x = in(b, h, w, d);
            float sig = 1.0f / (1.0f + expf(-x));
            out(b, h, w, d) = x * sig;
          }
        }
      }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_SiLU);
