QNN_SDK_ROOT     := /home/yinrun/software/qualcomm/qairt/2.42.0.251225
QNN_INCLUDE      := $(QNN_SDK_ROOT)/include/QNN
QNN_TARGET_LIB   := $(QNN_SDK_ROOT)/lib/aarch64-android

HEXAGON_SDK      := /home/yinrun/software/qualcomm/Hexagon_SDK/6.4.0.2
HEXAGON_TOOLS    := $(HEXAGON_SDK)/tools/HEXAGON_Tools/19.0.04/Tools
ANDROID_NDK      := /home/yinrun/software/android-ndk-r29

PACKAGE_NAME := ReluDemoPackage
LIBRARY_NAME := libQnnHtpReluDemo.so
WORK         := build
SRC_DIR      := src

OP_SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OP_OBJS    := $(notdir $(patsubst %.cpp,%.o,$(OP_SOURCES)))

COMMON_CXX_FLAGS := \
    -std=c++17 \
    -I$(QNN_INCLUDE) \
    -fPIC -Wall -Wreorder \
    -Wno-missing-braces -Wno-unused-function \
    -Wno-format -Wno-unused-command-line-argument \
    -fvisibility=default -stdlib=libc++ \
    -DQNN_API="__attribute__((visibility(\"default\")))" \
    -D__QAIC_HEADER_EXPORT="__attribute__((visibility(\"default\")))" \
    -DTHIS_PKG_NAME=$(PACKAGE_NAME)

# ---- Hexagon v81 (SM8850 HTP) ----
HEXAGON_CXX := $(HEXAGON_TOOLS)/bin/hexagon-clang++
HEXAGON_FLAGS := $(COMMON_CXX_FLAGS) \
    -mv81 -mhvx -mhvx-length=128B -mhmx \
    -DUSE_OS_QURT -O2 -Wno-reorder -DPREPARE_DISABLED \
    -I$(HEXAGON_SDK)/rtos/qurt/computev81/include/qurt \
    -I$(HEXAGON_SDK)/rtos/qurt/computev81/include/posix \
    -I$(HEXAGON_SDK)/incs \
    -I$(HEXAGON_SDK)/incs/stddef

htp_v81: $(WORK)/hexagon-v81/$(LIBRARY_NAME)

$(WORK)/hexagon-v81:
	@mkdir -p $@

$(WORK)/hexagon-v81/%.o: $(SRC_DIR)/%.cpp | $(WORK)/hexagon-v81
	$(HEXAGON_CXX) $(HEXAGON_FLAGS) -MMD -c $< -o $@

$(WORK)/hexagon-v81/$(LIBRARY_NAME): $(patsubst %,$(WORK)/hexagon-v81/%,$(OP_OBJS))
	$(HEXAGON_CXX) -fPIC -std=c++17 -shared -o $@ $^

# ---- aarch64-android ----
AARCH64_CXX := $(ANDROID_NDK)/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ \
    --target=aarch64-none-linux-android21 \
    --sysroot=$(ANDROID_NDK)/toolchains/llvm/prebuilt/linux-x86_64/sysroot \
    -stdlib=libc++ -static-libstdc++

AARCH64_FLAGS := $(COMMON_CXX_FLAGS) \
    -D__HVXDBL__ \
    -I$(HEXAGON_TOOLS)/libnative/include \
    -ffast-math -DUSE_OS_LINUX -DANDROID \
    -fomit-frame-pointer \
    -Wno-invalid-offsetof -Wno-unused-variable -Wno-unused-parameter \
    -Wno-sign-compare -Wno-ignored-qualifiers -Wno-missing-field-initializers

AARCH64_LDFLAGS := -L$(QNN_TARGET_LIB) -lQnnHtp -lQnnHtpPrepare

htp_aarch64: $(WORK)/aarch64-android/$(LIBRARY_NAME)

$(WORK)/aarch64-android:
	@mkdir -p $@

$(WORK)/aarch64-android/%.o: $(SRC_DIR)/%.cpp | $(WORK)/aarch64-android
	$(AARCH64_CXX) $(AARCH64_FLAGS) -MMD -c $< -o $@

$(WORK)/aarch64-android/$(LIBRARY_NAME): $(patsubst %,$(WORK)/aarch64-android/%,$(OP_OBJS))
	$(AARCH64_CXX) -fPIC -std=c++17 -shared -o $@ $^ $(AARCH64_LDFLAGS)

# ---- Model .so (via qnn-model-lib-generator) ----
MODEL_SRC := model_src/model_relu.cpp
MODEL_LIB_DIR := model_lib

model: $(MODEL_LIB_DIR)/aarch64-android/libReluModel.so

$(MODEL_LIB_DIR)/aarch64-android/libReluModel.so: $(MODEL_SRC)
	@export PATH=$(ANDROID_NDK):$$PATH && \
	$(QNN_SDK_ROOT)/bin/x86_64-linux-clang/qnn-model-lib-generator \
	    -c $(MODEL_SRC) -t aarch64-android -l ReluModel -o $(MODEL_LIB_DIR)

# ---- Convenience targets ----
all: htp_v81 htp_aarch64

clean:
	rm -rf $(WORK) $(MODEL_LIB_DIR)

.PHONY: htp_v81 htp_aarch64 model all clean
