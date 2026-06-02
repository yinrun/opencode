// HERmsNorm - RMSNorm custom op for HTP
// Pattern: scalar operator() loop (Hexagon compiler auto-vectorizes)
// Same proven pattern as HESiLU (cosine 0.99999994)
//
// Math: y[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]
// Inputs: in0 = x [B,H,W,D], in1 = weight [1,1,1,D]
// Output: y [B,H,W,D]

#include <cmath>
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_RMSNorm);

template <typename T_Ttype>
int rmsnormImpl(T_Ttype &out, const T_Ttype &in, const T_Ttype &weight);

// Quant fallback - REQUIRED for HTP to compile graph
DEF_PACKAGE_OP((rmsnormImpl<Tensor>), "HERmsNorm")

// fp32 variant
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((rmsnormImpl<PlainFloatTensor>),
                                  "HERmsNorm", SNAIL, Flags::RESOURCE_HVX)

// fp16 variant
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((rmsnormImpl<PlainFloat16Tensor>),
                                  "HERmsNorm", SNAIL, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("HERmsNorm", "in0", "in1"),
                      Flat("*"), MainMemory("*"))

template <typename T_Ttype>
int rmsnormImpl(T_Ttype &out, const T_Ttype &in, const T_Ttype &weight) {
    out.set_dims(in);

    const float eps = 1e-6f;
    const Idx B = in.dim(0);
    const Idx H = in.dim(1);
    const Idx W = in.dim(2);
    const Idx D = in.dim(3);

    for (Idx b = 0; b < B; b++) {
      for (Idx h = 0; h < H; h++) {
        for (Idx w = 0; w < W; w++) {
          // Pass 1: compute sum of squares in fp32
          float sum_sq = 0.0f;
          for (Idx d = 0; d < D; d++) {
            float x = in(b, h, w, d);
            sum_sq += x * x;
          }

          // rsqrt(mean(x^2) + eps)
          float rms_inv = 1.0f / sqrtf(sum_sq / (float)D + eps);

          // Pass 2: output = x * rms_inv * weight
          for (Idx d = 0; d < D; d++) {
            float x = in(b, h, w, d);
            float wt = weight(0, 0, 0, d);
            out(b, h, w, d) = x * rms_inv * wt;
          }
        }
      }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_RMSNorm);
