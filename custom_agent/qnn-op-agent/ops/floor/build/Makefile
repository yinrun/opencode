QNN_SDK_ROOT     := /home/yinrun/software/qualcomm/qairt/2.42.0.251225
QNN_INCLUDE      := $(QNN_SDK_ROOT)/include/QNN
QNN_TARGET_LIB   := $(QNN_SDK_ROOT)/lib/aarch64-android

HEXAGON_SDK      := /home/yinrun/software/qualcomm/Hexagon_SDK/6.4.0.2
HEXAGON_TOOLS    := $(HEXAGON_SDK)/tools/HEXAGON_Tools/19.0.04/Tools
ANDROID_NDK      := /home/yinrun/software/android-ndk-r29

PACKAGE_NAME := HEFloorPackage
LIBRARY_NAME := libQnnHtpFloor.so
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

# ---- Convenience targets ----
all: htp_v81 htp_aarch64 htp_x86

htp_x86: $(WORK)/x86_64-linux-clang/$(LIBRARY_NAME)

X86_LIBNATIVE := $(HEXAGON_TOOLS)/libnative
X86_CXX_FLAGS := $(COMMON_CXX_FLAGS) \
    -D__HVXDBL__ -I$(X86_LIBNATIVE)/include \
    -DUSE_OS_LINUX -fomit-frame-pointer -Wno-invalid-offsetof
X86_LDFLAGS := -Wl,--whole-archive -L$(X86_LIBNATIVE)/lib -lnative -Wl,--no-whole-archive -lpthread

$(WORK)/x86_64-linux-clang:
	@mkdir -p $@

$(WORK)/x86_64-linux-clang/%.o: $(SRC_DIR)/%.cpp | $(WORK)/x86_64-linux-clang
	clang++ $(X86_CXX_FLAGS) -MMD -c $< -o $@

$(WORK)/x86_64-linux-clang/$(LIBRARY_NAME): $(patsubst %,$(WORK)/x86_64-linux-clang/%,$(OP_OBJS))
	clang++ -fPIC -std=c++17 -shared -o $@ $^ $(X86_LDFLAGS)

clean:
	rm -rf $(WORK)

.PHONY: htp_v81 htp_aarch64 htp_x86 all clean
