#include <cmath>
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_HEAtan);

template <typename T_Ttype>
int heAtanImpl(T_Ttype &out, const T_Ttype &in);

DEF_PACKAGE_OP((heAtanImpl<Tensor>), "HEAtan")
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heAtanImpl<PlainFloatTensor>),   "HEAtan", SNAIL, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heAtanImpl<PlainFloat16Tensor>), "HEAtan", SNAIL, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("HEAtan", "in0"), Flat("*"), MainMemory("*"))

template <typename T_Ttype>
int heAtanImpl(T_Ttype &out, const T_Ttype &in) {
    out.set_dims(in);
    for (Idx n = 0; n < in.dim(0); n++) {
      for (Idx h = 0; h < in.dim(1); h++) {
        for (Idx w = 0; w < in.dim(2); w++) {
          for (Idx d = 0; d < in.dim(3); d++) {
            float xv = in(n, h, w, d);
            out(n,h,w,d) = atanf(xv);
          }
        }
      }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_HEAtan);
