// Single-op model for QNN native MatMul baseline measurement
// Package "qti.aisw" + op type "MatMul" — pure built-in, no custom op package needed.
// Follows L1/L7 pattern: getGraphInfoFromModels + DO_GRAPH_NODE_VALIDATIONS=0,
// never call model.finalize() (qnn-net-run host driver finalizes itself).
#include "QnnModel.hpp"
#include "QnnOpDef.h"

#define DO_GRAPH_NODE_VALIDATIONS 0
using namespace qnn_wrapper_api;

extern "C" {
QNN_API
ModelError_t QnnModel_composeGraphs(
    Qnn_BackendHandle_t backendHandle,
    QNN_INTERFACE_VER_TYPE interface,
    Qnn_ContextHandle_t contextHandle,
    const GraphConfigInfo_t** graphsConfigInfo,
    const uint32_t numGraphsConfigInfo,
    GraphInfoPtr_t** graphsInfo,
    uint32_t* numGraphsInfo,
    bool debug,
    QnnLog_Callback_t logCallback,
    QnnLog_Level_t maxLogLevel) {
  ModelError_t err = MODEL_NO_ERROR;
  QnnModel model;
  const QnnGraph_Config_t** graphConfigs = nullptr;
  VALIDATE(getQnnGraphConfigFromInfo("QnnMatMul_graph",
      graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);
  VALIDATE(model.initialize(backendHandle, interface, contextHandle,
      "QnnMatMul_graph", debug, DO_GRAPH_NODE_VALIDATIONS,
      graphConfigs), err);

  // ---- input A: [1,1,128,896] fp16 ----
  uint32_t dims_a[] = {1, 1, 128, 896};
  VALIDATE(model.addTensor("a",
    (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
      .v1 = {.id = 0, .name = "a",
        .type = QNN_TENSOR_TYPE_APP_WRITE,
        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
        .dataType = QNN_DATATYPE_FLOAT_16,
        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
          QNN_QUANTIZATION_ENCODING_UNDEFINED,
          {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
        .rank = 4, .dimensions = dims_a,
        .memType = QNN_TENSORMEMTYPE_RAW,
        .clientBuf = {.data = nullptr, .dataSize = 0}}}), err);

  // ---- input B: [1,1,896,4864] fp16 ----
  uint32_t dims_b[] = {1, 1, 896, 4864};
  VALIDATE(model.addTensor("b",
    (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
      .v1 = {.id = 0, .name = "b",
        .type = QNN_TENSOR_TYPE_APP_WRITE,
        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
        .dataType = QNN_DATATYPE_FLOAT_16,
        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
          QNN_QUANTIZATION_ENCODING_UNDEFINED,
          {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
        .rank = 4, .dimensions = dims_b,
        .memType = QNN_TENSORMEMTYPE_RAW,
        .clientBuf = {.data = nullptr, .dataSize = 0}}}), err);

  // ---- params: transpose_in0=false, transpose_in1=false ----
  Qnn_Param_t matmul_params[] = {
    {.paramType = QNN_PARAMTYPE_SCALAR,
     .name = "transpose_in0",
     {.scalarParam = {.dataType = QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}},
    {.paramType = QNN_PARAMTYPE_SCALAR,
     .name = "transpose_in1",
     {.scalarParam = {.dataType = QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };

  // ---- output: [1,1,128,4864] fp16 (APP_READ) ----
  uint32_t dims_output[] = {1, 1, 128, 4864};
  const char* inputNames[] = {"a", "b"};

  Qnn_Tensor_t outputTensors[1];
  outputTensors[0].version = QNN_TENSOR_VERSION_1;
  outputTensors[0].v1.id = 0;
  outputTensors[0].v1.name = "output";
  outputTensors[0].v1.type = QNN_TENSOR_TYPE_APP_READ;
  outputTensors[0].v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  outputTensors[0].v1.dataType = QNN_DATATYPE_FLOAT_16;
  outputTensors[0].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  outputTensors[0].v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  outputTensors[0].v1.rank = 4;
  outputTensors[0].v1.dimensions = dims_output;
  outputTensors[0].v1.memType = QNN_TENSORMEMTYPE_RAW;
  outputTensors[0].v1.clientBuf.data = nullptr;
  outputTensors[0].v1.clientBuf.dataSize = 0;

  // qti.aisw is the built-in package; "MatMul" matches QNN_OP_MAT_MUL.
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
      "QnnMatMul_node",     // node name
      "qti.aisw",           // package name (built-in)
      "MatMul",             // op type
      matmul_params,        // params
      2,                    // numParams
      inputNames,           // input tensor names
      2,                    // numInputs
      outputTensors,        // output tensors
      1                     // numOutputs
  ), err);

  // Don't finalize here (L1).
  VALIDATE(getGraphInfoFromModels(&model, 1, graphsInfo), err);
  *numGraphsInfo = 1;
  return err;
}

QNN_API
ModelError_t QnnModel_freeGraphsInfo(
    GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo) {
  return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);
}
} // extern "C"
