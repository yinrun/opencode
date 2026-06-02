#include <cmath>
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_HETanh);

template <typename T_Ttype>
int heTanhImpl(T_Ttype &out, const T_Ttype &in);

DEF_PACKAGE_OP((heTanhImpl<Tensor>), "HETanh")
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heTanhImpl<PlainFloatTensor>),   "HETanh", SNAIL, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heTanhImpl<PlainFloat16Tensor>), "HETanh", SNAIL, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("HETanh", "in0"), Flat("*"), MainMemory("*"))

// Tanh: y = tanh(x)
// Compute path: fp16 -> fp32 promote -> tanhf -> fp16 round
// Domain: all real numbers; saturates to +/-1 for |x| > ~5
template <typename T_Ttype>
int heTanhImpl(T_Ttype &out, const T_Ttype &in) {
    out.set_dims(in);
    for (Idx n = 0; n < in.dim(0); n++) {
      for (Idx h = 0; h < in.dim(1); h++) {
        for (Idx w = 0; w < in.dim(2); w++) {
          for (Idx d = 0; d < in.dim(3); d++) {
            float xv = in(n, h, w, d);
            out(n, h, w, d) = tanhf(xv);
          }
        }
      }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_HETanh);
