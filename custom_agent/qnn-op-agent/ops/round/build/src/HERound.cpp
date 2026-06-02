#include <cmath>
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_HERound);

template <typename T_Ttype>
int heRoundImpl(T_Ttype &out, const T_Ttype &in);

DEF_PACKAGE_OP((heRoundImpl<Tensor>), "HERound")
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heRoundImpl<PlainFloatTensor>),   "HERound", SNAIL, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heRoundImpl<PlainFloat16Tensor>), "HERound", SNAIL, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("HERound", "in0"), Flat("*"), MainMemory("*"))

template <typename T_Ttype>
int heRoundImpl(T_Ttype &out, const T_Ttype &in) {
    out.set_dims(in);
    for (Idx n = 0; n < in.dim(0); n++) {
      for (Idx h = 0; h < in.dim(1); h++) {
        for (Idx w = 0; w < in.dim(2); w++) {
          for (Idx d = 0; d < in.dim(3); d++) {
            float xv = in(n, h, w, d);
            out(n,h,w,d) = rintf(xv);  // banker's rounding (round half to even) to match numpy
          }
        }
      }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_HERound);
