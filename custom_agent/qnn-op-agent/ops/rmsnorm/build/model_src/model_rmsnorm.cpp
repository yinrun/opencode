// Single-op model: HERmsNorm for qnn-net-run testing
// Inputs: x [1,1,128,896] fp16, weight [1,1,1,896] fp16
// Output: y [1,1,128,896] fp16
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
  VALIDATE(getQnnGraphConfigFromInfo("RMSNorm_graph",
      graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);
  VALIDATE(model.initialize(backendHandle, interface, contextHandle,
      "RMSNorm_graph", debug, DO_GRAPH_NODE_VALIDATIONS,
      graphConfigs), err);

  // Input tensor: x [1, 1, 128, 896] fp16
  uint32_t dims_input[] = {1, 1, 128, 896};
  VALIDATE(model.addTensor("input",
    (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
      .v1 = {.id = 0, .name = "input",
        .type = QNN_TENSOR_TYPE_APP_WRITE,
        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
        .dataType = QNN_DATATYPE_FLOAT_16,
        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
          QNN_QUANTIZATION_ENCODING_UNDEFINED,
          {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
        .rank = 4, .dimensions = dims_input,
        .memType = QNN_TENSORMEMTYPE_RAW,
        .clientBuf = {.data = nullptr, .dataSize = 0}}}), err);

  // Weight tensor: w [1, 1, 1, 896] fp16
  uint32_t dims_weight[] = {1, 1, 1, 896};
  VALIDATE(model.addTensor("weight",
    (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
      .v1 = {.id = 0, .name = "weight",
        .type = QNN_TENSOR_TYPE_APP_WRITE,
        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
        .dataType = QNN_DATATYPE_FLOAT_16,
        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
          QNN_QUANTIZATION_ENCODING_UNDEFINED,
          {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
        .rank = 4, .dimensions = dims_weight,
        .memType = QNN_TENSORMEMTYPE_RAW,
        .clientBuf = {.data = nullptr, .dataSize = 0}}}), err);

  // Output tensor: y [1, 1, 128, 896] fp16
  uint32_t dims_output[] = {1, 1, 128, 896};

  const char* inputNames[] = {"input", "weight"};

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

  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,
      "RMSNorm_node",        // node name
      "HERmsNormPackage",    // package name
      "HERmsNorm",           // op type
      nullptr,               // params
      0,                     // numParams
      inputNames,            // input tensor names
      2,                     // numInputs
      outputTensors,         // output tensors
      1                      // numOutputs
  ), err);

  // Do NOT call model.finalize() — qnn-net-run calls QnnGraph_finalize itself
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
