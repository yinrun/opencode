// HESoftmax: y = exp(x - max(x, dim=-1)) / sum(exp(x - max), dim=-1)
// Reduces over last dimension. Numerically stable form (max-subtract).
#include <cmath>
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_Softmax);

template <typename T_Ttype>
int softmaxImpl(T_Ttype &out, const T_Ttype &in);

DEF_PACKAGE_OP((softmaxImpl<Tensor>), "HESoftmax")
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((softmaxImpl<PlainFloatTensor>),   "HESoftmax", SNAIL, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((softmaxImpl<PlainFloat16Tensor>), "HESoftmax", SNAIL, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("HESoftmax", "in0"), Flat("*"), MainMemory("*"))

template <typename T_Ttype>
int softmaxImpl(T_Ttype &out, const T_Ttype &in) {
    out.set_dims(in);
    const Idx D = in.dim(3);
    for (Idx b = 0; b < in.dim(0); b++) {
      for (Idx h = 0; h < in.dim(1); h++) {
        for (Idx w = 0; w < in.dim(2); w++) {
          // Pass 1: max
          float mx = -1e30f;
          for (Idx d = 0; d < D; d++) {
            float v = in(b, h, w, d);
            if (v > mx) mx = v;
          }
          // Pass 2: sum exp(x - max)
          float sum = 0.0f;
          for (Idx d = 0; d < D; d++) {
            sum += expf((float)in(b, h, w, d) - mx);
          }
          float inv = 1.0f / sum;
          // Pass 3: write normalized
          for (Idx d = 0; d < D; d++) {
            out(b, h, w, d) = expf((float)in(b, h, w, d) - mx) * inv;
          }
        }
      }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_Softmax);
