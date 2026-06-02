// HEDiv - element-wise divide y = a / b
// Pattern follows HEAdd: register Tensor (quant fallback)
// + PlainFloatTensor + PlainFloat16Tensor variants. Flat + MainMemory props.
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_HEDiv);

template <typename T_Ttype>
int heDivImpl(T_Ttype &out, const T_Ttype &a, const T_Ttype &b);

// Quant fallback - REQUIRED for HTP graph finalize (avoids err 1002)
DEF_PACKAGE_OP((heDivImpl<Tensor>), "HEDiv")

// fp32
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heDivImpl<PlainFloatTensor>),   "HEDiv", SNAIL, Flags::RESOURCE_HVX)

// fp16
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heDivImpl<PlainFloat16Tensor>), "HEDiv", SNAIL, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("HEDiv", "in0", "in1"), Flat("*"), MainMemory("*"))

template <typename T_Ttype>
int heDivImpl(T_Ttype &out, const T_Ttype &a, const T_Ttype &b) {
    out.set_dims(a);
    for (Idx n = 0; n < a.dim(0); n++) {
      for (Idx h = 0; h < a.dim(1); h++) {
        for (Idx w = 0; w < a.dim(2); w++) {
          for (Idx d = 0; d < a.dim(3); d++) {
            float av = a(n, h, w, d);
            float bv = b(n, h, w, d);
            out(n, h, w, d) = av / bv;
          }
        }
      }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_HEDiv);
