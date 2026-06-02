#!/bin/bash
# Deploy and run Relu FP16 demo on SM8850 device
# Prerequisites: build op package (htp_v81 + htp_aarch64) and model .so

set -e

QNN_SDK="/home/yinrun/software/qualcomm/qairt/2.42.0.251225"
DEVICE_SERIAL="204cbd30"
DEVICE_DIR="/data/local/tmp/relu_test"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
TEST_DATA="$SCRIPT_DIR/test_data"

ADB="adb -s $DEVICE_SERIAL"

echo "=== Relu FP16 Demo: Deploy & Run ==="

# Check prerequisites
echo "[1/6] Checking build artifacts..."
HEX_SO="$BUILD_DIR/build/hexagon-v81/libQnnHtpReluDemo.so"
ARM_SO="$BUILD_DIR/build/aarch64-android/libQnnHtpReluDemo.so"
MODEL_SO="$BUILD_DIR/model_lib/aarch64-android/libReluModel.so"

for f in "$HEX_SO" "$ARM_SO" "$MODEL_SO" "$TEST_DATA/input.raw" "$TEST_DATA/input_list.txt"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        echo "Run: cd build && make all && make model"
        exit 1
    fi
done
echo "  All artifacts found."

# Kill any competing HTP processes
echo "[2/6] Ensuring HTP is free..."
$ADB shell "pkill -9 -f qnn_llama_runner" 2>/dev/null || true
$ADB shell "pkill -9 -f qnn-net-run" 2>/dev/null || true
sleep 3
echo "  Done."

# Push files to device
echo "[3/6] Pushing files to device..."
$ADB shell "mkdir -p $DEVICE_DIR/output"

# SDK libraries (skip if already present)
$ADB push "$QNN_SDK/bin/aarch64-android/qnn-net-run" /data/local/tmp/ 2>/dev/null || true
$ADB push "$QNN_SDK/lib/aarch64-android/libQnnHtp.so" /data/local/tmp/ 2>/dev/null || true
$ADB push "$QNN_SDK/lib/aarch64-android/libQnnHtpPrepare.so" /data/local/tmp/ 2>/dev/null || true
$ADB push "$QNN_SDK/lib/aarch64-android/libQnnHtpV81Stub.so" /data/local/tmp/ 2>/dev/null || true
$ADB push "$QNN_SDK/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so" /data/local/tmp/ 2>/dev/null || true

# Op package
$ADB push "$HEX_SO" /data/local/tmp/libHERelu_hex.so
$ADB push "$ARM_SO" "$DEVICE_DIR/libHERelu_arm.so"

# Model
$ADB push "$MODEL_SO" "$DEVICE_DIR/"

# Test data
$ADB push "$TEST_DATA/input.raw" "$DEVICE_DIR/"
$ADB push "$TEST_DATA/input_list.txt" "$DEVICE_DIR/"

$ADB shell "chmod +x /data/local/tmp/qnn-net-run"
echo "  Done."

# Run on device
echo "[4/6] Running qnn-net-run on device..."
# IMPORTANT: --use_native_input_files --use_native_output_files
# Without these, qnn-net-run reads .raw as fp32 and converts, corrupting fp16 data
$ADB shell "cd $DEVICE_DIR && \
  export LD_LIBRARY_PATH=\"/data/local/tmp:$DEVICE_DIR\" && \
  export ADSP_LIBRARY_PATH=\"/data/local/tmp;$DEVICE_DIR;/vendor/dsp/cdsp\" && \
  rm -rf output && mkdir output && \
  /data/local/tmp/qnn-net-run \
    --backend /data/local/tmp/libQnnHtp.so \
    --model $DEVICE_DIR/libReluModel.so \
    --input_list $DEVICE_DIR/input_list.txt \
    --output_dir $DEVICE_DIR/output \
    --use_native_input_files \
    --use_native_output_files \
    --op_packages $DEVICE_DIR/libHERelu_arm.so:InterfaceProvider:CPU,/data/local/tmp/libHERelu_hex.so:InterfaceProvider:HTP"

echo "  Execution complete."

# Pull output
echo "[5/6] Pulling output from device..."
mkdir -p "$SCRIPT_DIR/device_output"
$ADB pull "$DEVICE_DIR/output/Result_0/output_native.raw" "$SCRIPT_DIR/device_output/output.raw"
echo "  Output pulled."

# Validate
echo "[6/6] Validating output..."
python3 "$SCRIPT_DIR/validate_output.py" "$SCRIPT_DIR/device_output/output.raw"
