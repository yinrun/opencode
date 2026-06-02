// HEMatMul - matrix multiply C = A @ B
// V1: scalar fp32 accumulation, no HVX/HMX (correctness first)
//
// A: [1, 1, M, K]   in0
// B: [1, 1, K, N]   in1
// C: [1, 1, M, N]   out
//
// Layout assumption: row-major in last two dims (a(0,0,m,k), b(0,0,k,n)).
// For 4D tensors, dim(2)=M / dim(3)=K for A; dim(2)=K / dim(3)=N for B.
//
// Pattern follows HEAdd (multi-input) + RMSNorm (fp32 inner accumulation).
// Key difference from element-wise ops: out shape is [.., M, N], NOT same
// as any input. Output dims are constructed explicitly via size_t array.

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_HEMatMul);

template <typename T_Ttype>
int heMatMulImpl(T_Ttype &out, const T_Ttype &a, const T_Ttype &b);

// Quant fallback - REQUIRED for HTP graph finalize (avoids err 1002)
DEF_PACKAGE_OP((heMatMulImpl<Tensor>), "HEMatMul")

// fp32
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heMatMulImpl<PlainFloatTensor>),
                                  "HEMatMul", SNAIL, Flags::RESOURCE_HVX)

// fp16
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((heMatMulImpl<PlainFloat16Tensor>),
                                  "HEMatMul", SNAIL, Flags::RESOURCE_HVX)

DEF_TENSOR_PROPERTIES(Op("HEMatMul", "in0", "in1"),
                      Flat("*"), MainMemory("*"))

// Split along output dim 3 (N axis) for large matrices.
// HTP rejects (err 1002) the unsplit op when N is large (around N >= 4800
// for K=896 fp16 — the scheduler estimates the unsplit op exceeds budget).
// Slicing only B along its dim 3 produces the corresponding output column slice;
// A stays whole because every row of A is needed for every column slice.
//
// Chunk size 4096 chosen to stay within the working envelope (N=4096 verified
// pass, N=4800 verified fail). 4864 = 4096 + 768 — both chunks within budget.
DEF_PACKAGE_OPTIMIZATION(
    EARLY + 1,
    Op("HEMatMul", "A", "B"),
    GT(DIM_DEPTH("*"), 4096),
    AUTOSPLIT(3, "I", 4096, Op("HEMatMul", "A", TYPICAL_SLICE("B", "I"))))

template <typename T_Ttype>
int heMatMulImpl(T_Ttype &out, const T_Ttype &a, const T_Ttype &b) {
    const Idx B0 = a.dim(0);
    const Idx H  = a.dim(1);
    const Idx M  = a.dim(2);
    const Idx K  = a.dim(3);
    const Idx N  = b.dim(3);

    for (Idx bi = 0; bi < B0; bi++) {
      for (Idx hi = 0; hi < H; hi++) {
        for (Idx mi = 0; mi < M; mi++) {
          for (Idx ni = 0; ni < N; ni++) {
            float acc = 0.0f;
            for (Idx ki = 0; ki < K; ki++) {
              float av = a(bi, hi, mi, ki);
              float bv = b(bi, hi, ki, ni);
              acc += av * bv;
            }
            out(bi, hi, mi, ni) = acc;
          }
        }
      }
    }
    return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_HEMatMul);
