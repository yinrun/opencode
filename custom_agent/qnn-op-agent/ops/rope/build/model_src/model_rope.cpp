// Single-graph model with TWO HERoPE nodes: one for q, one for k.
// Inputs:  q [1,14,128,64], k [1,2,128,64], cos [1,1,128,64], sin [1,1,128,64]
// Outputs: q_out [1,14,128,64], k_out [1,2,128,64]
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
  VALIDATE(getQnnGraphConfigFromInfo("RoPE_graph",
      graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);
  VALIDATE(model.initialize(backendHandle, interface, contextHandle,
      "RoPE_graph", debug, DO_GRAPH_NODE_VALIDATIONS,
      graphConfigs), err);

  uint32_t dims_q[]   = {1, 14, 128, 64};
  uint32_t dims_k[]   = {1, 2,  128, 64};
  uint32_t dims_cos[] = {1, 1,  128, 64};
  uint32_t dims_sin[] = {1, 1,  128, 64};

  // q
  VALIDATE(model.addTensor("q",
    (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
      .v1 = {.id = 0, .name = "q",
        .type = QNN_TENSOR_TYPE_APP_WRITE,
        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
        .dataType = QNN_DATATYPE_FLOAT_16,
        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
          QNN_QUANTIZATION_ENCODING_UNDEFINED,
          {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
        .rank = 4, .dimensions = dims_q,
        .memType = QNN_TENSORMEMTYPE_RAW,
        .clientBuf = {.data = nullptr, .dataSize = 0}}}), err);

  // k
  VALIDATE(model.addTensor("k",
    (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
      .v1 = {.id = 0, .name = "k",
        .type = QNN_TENSOR_TYPE_APP_WRITE,
        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
        .dataType = QNN_DATATYPE_FLOAT_16,
        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
          QNN_QUANTIZATION_ENCODING_UNDEFINED,
          {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
        .rank = 4, .dimensions = dims_k,
        .memType = QNN_TENSORMEMTYPE_RAW,
        .clientBuf = {.data = nullptr, .dataSize = 0}}}), err);

  // cos
  VALIDATE(model.addTensor("cos",
    (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
      .v1 = {.id = 0, .name = "cos",
        .type = QNN_TENSOR_TYPE_APP_WRITE,
        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
        .dataType = QNN_DATATYPE_FLOAT_16,
        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
          QNN_QUANTIZATION_ENCODING_UNDEFINED,
          {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
        .rank = 4, .dimensions = dims_cos,
        .memType = QNN_TENSORMEMTYPE_RAW,
        .clientBuf = {.data = nullptr, .dataSize = 0}}}), err);

  // sin
  VALIDATE(model.addTensor("sin",
    (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
      .v1 = {.id = 0, .name = "sin",
        .type = QNN_TENSOR_TYPE_APP_WRITE,
        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
        .dataType = QNN_DATATYPE_FLOAT_16,
        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
          QNN_QUANTIZATION_ENCODING_UNDEFINED,
          {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
        .rank = 4, .dimensions = dims_sin,
        .memType = QNN_TENSORMEMTYPE_RAW,
        .clientBuf = {.data = nullptr, .dataSize = 0}}}), err);

  // ---- Node 1: q rotation ----
  uint32_t dims_q_out[] = {1, 14, 128, 64};
  const char* qInputNames[] = {"q", "cos", "sin"};
  Qnn_Tensor_t qOutTensors[1];
  qOutTensors[0].version = QNN_TENSOR_VERSION_1;
  qOutTensors[0].v1.id = 0;
  qOutTensors[0].v1.name = "q_out";
  qOutTensors[0].v1.type = QNN_TENSOR_TYPE_APP_READ;
  qOutTensors[0].v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  qOutTensors[0].v1.dataType = QNN_DATATYPE_FLOAT_16;
  qOutTensors[0].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  qOutTensors[0].v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  qOutTensors[0].v1.rank = 4;
  qOutTensors[0].v1.dimensions = dims_q_out;
  qOutTensors[0].v1.memType = QNN_TENSORMEMTYPE_RAW;
  qOutTensors[0].v1.clientBuf.data = nullptr;
  qOutTensors[0].v1.clientBuf.dataSize = 0;

  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
      "RoPE_q_node",
      "HERoPEPackage",
      "HERoPE",
      nullptr, 0,
      qInputNames, 3,
      qOutTensors, 1
  ), err);

  // ---- Node 2: k rotation ----
  uint32_t dims_k_out[] = {1, 2, 128, 64};
  const char* kInputNames[] = {"k", "cos", "sin"};
  Qnn_Tensor_t kOutTensors[1];
  kOutTensors[0].version = QNN_TENSOR_VERSION_1;
  kOutTensors[0].v1.id = 0;
  kOutTensors[0].v1.name = "k_out";
  kOutTensors[0].v1.type = QNN_TENSOR_TYPE_APP_READ;
  kOutTensors[0].v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  kOutTensors[0].v1.dataType = QNN_DATATYPE_FLOAT_16;
  kOutTensors[0].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  kOutTensors[0].v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  kOutTensors[0].v1.rank = 4;
  kOutTensors[0].v1.dimensions = dims_k_out;
  kOutTensors[0].v1.memType = QNN_TENSORMEMTYPE_RAW;
  kOutTensors[0].v1.clientBuf.data = nullptr;
  kOutTensors[0].v1.clientBuf.dataSize = 0;

  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
      "RoPE_k_node",
      "HERoPEPackage",
      "HERoPE",
      nullptr, 0,
      kInputNames, 3,
      kOutTensors, 1
  ), err);

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
