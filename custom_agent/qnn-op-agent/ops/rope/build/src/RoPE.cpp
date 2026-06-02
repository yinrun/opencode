// HERoPE: pair-rotated Rotary Position Embedding (HuggingFace interleaved layout)
// out[..., 2i]   = x[..., 2i]   * cos[..., 2i]   - x[..., 2i+1] * sin[..., 2i]
// out[..., 2i+1] = x[..., 2i+1] * cos[..., 2i+1] + x[..., 2i]   * sin[..., 2i+1]
// cos/sin broadcast over heads dim (dim 1).
#include <cmath>
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_RoPE);

template <typename T_Ttype>
int ropeImpl(T_Ttype &out, const T_Ttype &x, const T_Ttype &cos_t, const T_Ttype &sin_t);

DEF_PACKAGE_OP((ropeImpl<Tensor>), "HERoPE")
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((ropeImpl<PlainFloatTensor>),   "HERoPE", SNAIL, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((ropeImpl<PlainFloat16Tensor>), "HERoPE", SNAIL, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("HERoPE", "x", "cos", "sin"), Flat("*"), MainMemory("*"))

template <typename T_Ttype>
int ropeImpl(T_Ttype &out, const T_Ttype &x, const T_Ttype &cos_t, const T_Ttype &sin_t) {
    out.set_dims(x);
    for (Idx b = 0; b < x.dim(0); b++) {
      for (Idx h = 0; h < x.dim(1); h++) {
        for (Idx w = 0; w < x.dim(2); w++) {
          for (Idx d = 0; d < x.dim(3); d += 2) {
            float xe = x(b, h, w, d);
            float xo = x(b, h, w, d + 1);
            float ce = cos_t(b, 0, w, d);
            float co = cos_t(b, 0, w, d + 1);
            float se = sin_t(b, 0, w, d);
            float so = sin_t(b, 0, w, d + 1);
            out(b, h, w, d)     = xe * ce - xo * se;
            out(b, h, w, d + 1) = xo * co + xe * so;
          }
        }
      }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_RoPE);
