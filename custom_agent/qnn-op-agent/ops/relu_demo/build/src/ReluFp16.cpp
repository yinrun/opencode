//=============================================================================
// Relu FP16 HTP Op Package Demo
// Closely follows QNN SDK ExampleOpPackageReluFp16.cpp pattern
// Op name: HERelu (avoids conflict with built-in Relu)
//=============================================================================

#include <cmath>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_ReluFp16);

// op execute function declaration
template <typename T_Ttype>
int reluImplFp16(T_Ttype &out, const T_Ttype &in);

// Tensor properties: MainMemory (same as SDK's Relu.fp16)
DEF_TENSOR_PROPERTIES(Op("HERelu", "in0"), MainMemory("*", "in0"))

// Register implementations for both tensor types (SDK pattern)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((reluImplFp16<F16CroutonTensor>),
                                  "HERelu",
                                  FAST,
                                  Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((reluImplFp16<PlainFloat16Tensor>),
                                  "HERelu",
                                  FAST,
                                  Flags::RESOURCE_HVX)

/* execute function — fp16 relu using HVX, handles variable block sizes */
template <typename T_Ttype>
int reluImplFp16(T_Ttype &out, const T_Ttype &in) {
  debuglog("HERelu fp16 execute... dims=(%zdx%zdx%zdx%zd)",
           in.dim(0), in.dim(1), in.dim(2), in.dim(3));
  debuglog("in=%p out=%p blocktab_len=%zd", &in, &out, in.blocktab_len());
  out.set_dims(in);

  // Total fp16 elements and vectors
  size_t totalElements = in.dim(0) * in.dim(1) * in.dim(2) * in.dim(3);
  size_t totalVectors = (totalElements * 2 + 127) / 128;  // 2 bytes per fp16, 128 bytes per vector

  size_t inBlocks  = in.blocktab_len();
  auto inBlocktab  = in.blocktab_ptr();
  auto outBlocktab = out.blocktab_ptr();

  HVX_Vector vminval = Q6_Vh_vsplat_R(0x0);
  size_t vectorsDone = 0;
  for (uint32_t i = 0; i < inBlocks && vectorsDone < totalVectors; ++i) {
    auto inVptr  = (const HVX_Vector *)(inBlocktab[i]);
    auto outVptr = (HVX_Vector *)(outBlocktab[i]);
    // For F16CroutonTensor: 16 vectors per block
    // For PlainFloat16Tensor: all vectors in 1 block
    uint32_t vectorsInBlock = (inBlocks > 1) ? 16 : totalVectors;
    for (uint32_t j = 0; j < vectorsInBlock && vectorsDone < totalVectors; ++j) {
      HVX_Vector vin = inVptr[j];
      vin            = Q6_Vhf_vmax_VhfVhf(vin, vminval);
      outVptr[j]     = vin;
      vectorsDone++;
    }
  }
  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_ReluFp16);
