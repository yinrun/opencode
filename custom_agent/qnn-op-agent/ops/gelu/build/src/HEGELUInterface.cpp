#include "HTP/QnnHtpCommon.h"
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "HTP/core/unique_types.h"
#include "QnnOpDef.h"
#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"
#include <array>
#include <string>

DEFINE_UNIQ_TY()
BEGIN_PKG_OPS_OPTS_LIST()
DECLARE_PKG_OPS_OPTS_LIST(PKG_HEGELU)
END_PKG_OPS_OPTS_LIST()

static constexpr auto sg_packageName = THIS_PKG_NAME_STR;
static constexpr auto sg_opName      = "HEGELU";
static std::array<const char*, 1> sg_opNames{{sg_opName}};

static Qnn_ApiVersion_t    sg_sdkApiVersion  = QNN_HTP_API_VERSION_INIT;
static Qnn_Version_t       sg_opsetVersion   = {1, 0, 0};
static QnnOpPackage_Info_t sg_packageInfo    = {
    sg_packageName, sg_opNames.data(), nullptr, (uint32_t)sg_opNames.size(),
    nullptr, 0, QNN_SDK_BUILD_ID, &sg_sdkApiVersion, nullptr, &sg_opsetVersion, {0}};

static QnnOpPackage_GlobalInfrastructure_t sg_globalInfra    = nullptr;
static bool                                sg_initialized     = false;
static QnnLog_Callback_t                   sg_logCallback     = nullptr;
static QnnLog_Level_t                      sg_maxLogLevel     = (QnnLog_Level_t)0;
static bool                                sg_logInitialized  = false;

INIT_PACKAGE_OP_DEF()
INIT_PACKAGE_OPTIMIZATION_DEF()
INIT_PACKAGE_PARAM_ORDER_DEF()

INIT_PKG_CORE_INIT_FUNC()

Qnn_ErrorHandle_t HEGELUInit(QnnOpPackage_GlobalInfrastructure_t infrastructure) {
    if (sg_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
    REGISTER_PACKAGE_PARAM_ORDERS()
    REGISTER_PACKAGE_AXIS_PARAMS()
    REGISTER_PACKAGE_PER_CHANNEL_QUANTIZED_OPS()
    sg_globalInfra = infrastructure;
    sg_initialized = true;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t HEGELUGetInfo(const QnnOpPackage_Info_t** info) {
    if (!sg_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    if (!info) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    *info = &sg_packageInfo;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t HEGELUValidateOpConfig(Qnn_OpConfig_t opConfig) {
    if (std::string(sg_packageName) != opConfig.v1.packageName)
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if (std::string(opConfig.v1.typeName) == sg_opName) {
        if (opConfig.v1.numOfInputs != 1 || opConfig.v1.numOfOutputs != 1)
            return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    } else {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t HEGELUCreateOpImpl(QnnOpPackage_GraphInfrastructure_t graphInfra,
                                    QnnOpPackage_Node_t node,
                                    QnnOpPackage_OpImpl_t* opImpl) {
    (void)graphInfra; (void)node; (void)opImpl;
    return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}

Qnn_ErrorHandle_t HEGELUFreeOpImpl(QnnOpPackage_OpImpl_t opImpl) {
    (void)opImpl;
    return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}

Qnn_ErrorHandle_t HEGELUTerminate() {
    if (!sg_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    sg_globalInfra = nullptr;
    sg_initialized = false;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t HEGELULogInitialize(QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel) {
    if (!callback) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    sg_logCallback = callback; sg_maxLogLevel = maxLogLevel; sg_logInitialized = true;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t HEGELULogSetLevel(QnnLog_Level_t maxLogLevel) {
    sg_maxLogLevel = maxLogLevel;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t HEGELULogTerminate() {
    sg_logCallback = nullptr; sg_maxLogLevel = (QnnLog_Level_t)0; sg_logInitialized = false;
    return QNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

Qnn_ErrorHandle_t HEGELUPackageInterfaceProvider(QnnOpPackage_Interface_t* interface) {
    if (!interface) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    interface->interfaceVersion      = {1, 4, 0};
    interface->v1_4.init             = HEGELUInit;
    interface->v1_4.terminate        = HEGELUTerminate;
    interface->v1_4.getInfo          = HEGELUGetInfo;
    interface->v1_4.validateOpConfig = HEGELUValidateOpConfig;
    interface->v1_4.createOpImpl     = HEGELUCreateOpImpl;
    interface->v1_4.freeOpImpl       = HEGELUFreeOpImpl;
    interface->v1_4.logInitialize    = HEGELULogInitialize;
    interface->v1_4.logSetLevel      = HEGELULogSetLevel;
    interface->v1_4.logTerminate     = HEGELULogTerminate;
    return QNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
