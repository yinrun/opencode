// HESwiGLU: y = silu(gate) * up = gate * sigmoid(gate) * up
// Element-wise binary op, fused. Pattern mirrors HESiLU.
#include <cmath>
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_SwiGLU);

template <typename T_Ttype>
int swigluImpl(T_Ttype &out, const T_Ttype &gate, const T_Ttype &up);

// Quant fallback - REQUIRED for HTP graph compose.
DEF_PACKAGE_OP((swigluImpl<Tensor>), "HESwiGLU")

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((swigluImpl<PlainFloatTensor>),   "HESwiGLU", SNAIL, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((swigluImpl<PlainFloat16Tensor>), "HESwiGLU", SNAIL, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("HESwiGLU", "gate", "up"), Flat("*"), MainMemory("*"))

template <typename T_Ttype>
int swigluImpl(T_Ttype &out, const T_Ttype &gate, const T_Ttype &up) {
    out.set_dims(gate);
    for (Idx b = 0; b < gate.dim(0); b++) {
      for (Idx h = 0; h < gate.dim(1); h++) {
        for (Idx w = 0; w < gate.dim(2); w++) {
          for (Idx d = 0; d < gate.dim(3); d++) {
            float g = gate(b, h, w, d);
            float u = up(b, h, w, d);
            float sig = 1.0f / (1.0f + expf(-g));
            out(b, h, w, d) = (g * sig) * u;
          }
        }
      }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_SwiGLU);
