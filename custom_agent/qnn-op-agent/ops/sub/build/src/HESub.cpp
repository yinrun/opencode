// HESub - element-wise subtract y = a - b (used in softmax: x - max(x))
// Mirror HEAdd: 3 variants (Tensor quant fallback + PlainFloat + PlainFloat16),
// Flat + MainMemory props. Two inputs, one output.
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_HESub);

template <typename T_Ttype>
int heSubImpl(T_Ttype &out, const T_Ttype &a, const T_Ttype &b);

// Quant fallback - REQUIRED for HTP graph finalize (avoids err 1002)
DEF_PACKAGE_OP((heSubImpl<Tensor>), "HESub")

// fp32
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heSubImpl<PlainFloatTensor>),   "HESub", SNAIL, Flags::RESOURCE_HVX)

// fp16
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heSubImpl<PlainFloat16Tensor>), "HESub", SNAIL, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("HESub", "in0", "in1"), Flat("*"), MainMemory("*"))

template <typename T_Ttype>
int heSubImpl(T_Ttype &out, const T_Ttype &a, const T_Ttype &b) {
    out.set_dims(a);
    for (Idx n = 0; n < a.dim(0); n++) {
      for (Idx h = 0; h < a.dim(1); h++) {
        for (Idx w = 0; w < a.dim(2); w++) {
          for (Idx d = 0; d < a.dim(3); d++) {
            float av = a(n, h, w, d);
            float bv = b(n, h, w, d);
            out(n, h, w, d) = av - bv;
          }
        }
      }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_HESub);
