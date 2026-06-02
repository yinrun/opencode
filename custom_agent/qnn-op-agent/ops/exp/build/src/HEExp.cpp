// HEExp - element-wise exp y = exp(x)
// Pattern follows HESiLU/HEDiv: register Tensor (quant fallback)
// + PlainFloatTensor + PlainFloat16Tensor variants. Flat + MainMemory props.
#include <cmath>
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_HEExp);

template <typename T_Ttype>
int heExpImpl(T_Ttype &out, const T_Ttype &in);

// Quant fallback - REQUIRED for HTP graph finalize (avoids err 1002)
DEF_PACKAGE_OP((heExpImpl<Tensor>), "HEExp")

// fp32
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heExpImpl<PlainFloatTensor>),   "HEExp", SNAIL, Flags::RESOURCE_HVX)

// fp16
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heExpImpl<PlainFloat16Tensor>), "HEExp", SNAIL, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("HEExp", "in0"), Flat("*"), MainMemory("*"))

template <typename T_Ttype>
int heExpImpl(T_Ttype &out, const T_Ttype &in) {
    out.set_dims(in);
    for (Idx n = 0; n < in.dim(0); n++) {
      for (Idx h = 0; h < in.dim(1); h++) {
        for (Idx w = 0; w < in.dim(2); w++) {
          for (Idx d = 0; d < in.dim(3); d++) {
            float x = in(n, h, w, d);
            out(n, h, w, d) = expf(x);
          }
        }
      }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_HEExp);
