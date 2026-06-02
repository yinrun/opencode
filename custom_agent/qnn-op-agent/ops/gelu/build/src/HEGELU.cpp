#include <cmath>
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_HEGELU);

template <typename T_Ttype>
int heGeluImpl(T_Ttype &out, const T_Ttype &in);

DEF_PACKAGE_OP((heGeluImpl<Tensor>), "HEGELU")
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heGeluImpl<PlainFloatTensor>),   "HEGELU", SNAIL, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heGeluImpl<PlainFloat16Tensor>), "HEGELU", SNAIL, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("HEGELU", "in0"), Flat("*"), MainMemory("*"))

// GELU sigmoid approximation: y = x * sigmoid(1.702 * x)
// Compute path: fp16 -> fp32 promote -> sigmoid -> fp32 mul -> fp16 round
template <typename T_Ttype>
int heGeluImpl(T_Ttype &out, const T_Ttype &in) {
    out.set_dims(in);
    for (Idx n = 0; n < in.dim(0); n++) {
      for (Idx h = 0; h < in.dim(1); h++) {
        for (Idx w = 0; w < in.dim(2); w++) {
          for (Idx d = 0; d < in.dim(3); d++) {
            float xv = in(n, h, w, d);
            float sig = 1.0f / (1.0f + expf(-1.702f * xv));
            out(n, h, w, d) = xv * sig;
          }
        }
      }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_HEGELU);
