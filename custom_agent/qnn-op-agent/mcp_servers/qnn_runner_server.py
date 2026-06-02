"""qnn-runner MCP server — 单算子设备执行 + numpy 参考计算

Design:
- run_custom_op: 在设备 HTP 上执行自定义算子（通过预编译的 model .so + qnn-net-run）
- run_numpy_reference: 在 host 用 numpy fp32 计算参考输出
- generate_test_inputs: 生成确定性测试输入

单算子对比流程：
1. 用 numpy 计算参考输出（ground truth）
2. 用 qnn-net-run + custom op package 在设备 HTP 执行
3. 对比两者输出

对于 qnn-net-run 执行自定义算子，需要：
- model .so: 定义包含单个自定义 op 节点的 QNN graph
- op_packages: 自定义 op 的 hexagon .so + aarch64 .so
- backend: libQnnHtp.so
- input_list: 输入文件列表
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from fastmcp import FastMCP

mcp = FastMCP("qnn-runner")

SERIAL = (os.environ.get("ANDROID_SERIAL") or "").strip() or "204cbd30"
QNN_SDK = os.environ.get("QNN_SDK_242",
    "/home/yinrun/software/qualcomm/qairt/2.42.0.251225")
HEXAGON_SDK = os.environ.get("HEXAGON_SDK_ROOT",
    "/home/yinrun/software/qualcomm/Hexagon_SDK/6.4.0.2")
ANDROID_NDK = os.environ.get("ANDROID_NDK_ROOT",
    "/home/yinrun/software/android-ndk-r29")

DEVICE_WORK_DIR = "/data/local/tmp/qnn_op_test"
ADB = ["adb", "-s", SERIAL]
PROJECT_ROOT = Path(__file__).parent.parent


def _adb(args, timeout=120):
    try:
        env = {k: v for k, v in os.environ.items() if not (k == "ANDROID_SERIAL" and not v)}
        r = subprocess.run(ADB + args, capture_output=True, text=True, timeout=timeout, env=env)
        return {"rc": r.returncode, "out": r.stdout.strip(), "err": r.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"rc": -1, "out": "", "err": f"timeout {timeout}s"}


def _shell(cmd, timeout=60):
    return _adb(["shell", cmd], timeout)


def _push(local, remote):
    r = _adb(["push", str(local), remote])
    return r["rc"] == 0


def _pull(remote, local):
    os.makedirs(os.path.dirname(local), exist_ok=True)
    r = _adb(["pull", remote, str(local)])
    return r["rc"] == 0


# ============================================================
# Numpy reference implementations for each op
# ============================================================

def _numpy_rmsnorm(input_x: np.ndarray, weight: np.ndarray,
                   eps: float = 1e-6) -> np.ndarray:
    """RMSNorm: y = x * w / sqrt(mean(x^2) + eps)"""
    x = input_x.astype(np.float32)
    w = weight.astype(np.float32)
    mean_sq = np.mean(x * x, axis=-1, keepdims=True)
    rms_inv = 1.0 / np.sqrt(mean_sq + eps)
    out = x * rms_inv * w
    return out.astype(np.float16)


def _numpy_swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """SwiGLU: y = silu(gate) * up"""
    g = gate.astype(np.float32)
    u = up.astype(np.float32)
    silu = g * (1.0 / (1.0 + np.exp(-g)))  # silu(x) = x * sigmoid(x)
    return (silu * u).astype(np.float16)


def _numpy_rope(q: np.ndarray, k: np.ndarray,
                cos_cache: np.ndarray, sin_cache: np.ndarray):
    """RoPE: apply rotary position embedding to q and k."""
    def _rotate_half(x):
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return np.concatenate([-x2, x1], axis=-1)

    q_f = q.astype(np.float32)
    k_f = k.astype(np.float32)
    cos_f = cos_cache.astype(np.float32)
    sin_f = sin_cache.astype(np.float32)
    q_out = (q_f * cos_f + _rotate_half(q_f) * sin_f).astype(np.float16)
    k_out = (k_f * cos_f + _rotate_half(k_f) * sin_f).astype(np.float16)
    return q_out, k_out


def _numpy_softmax(x: np.ndarray) -> np.ndarray:
    """Softmax: numerically stable."""
    xf = x.astype(np.float32)
    xf = xf - np.max(xf, axis=-1, keepdims=True)
    e = np.exp(xf)
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float16)


def _numpy_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MatMul: C = A @ B"""
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)


NUMPY_OPS = {
    "rmsnorm": _numpy_rmsnorm,
    "swiglu": _numpy_swiglu,
    "rope": _numpy_rope,
    "softmax": _numpy_softmax,
    "matmul": _numpy_matmul,
}


@mcp.tool()
def generate_test_inputs(op_name: str, shapes: dict, dtype: str = "float16",
                         seed: int = 42) -> dict:
    """使用固定 seed 生成确定性测试输入，保存为 .raw 文件。
    shapes: {"input_name": [dim1, dim2, ...], ...}
    """
    rng = np.random.default_rng(seed)
    dt = {"float16": np.float16, "float32": np.float32}.get(dtype, np.float16)

    output_dir = str(PROJECT_ROOT / "baseline" / "test_inputs" / op_name)
    os.makedirs(output_dir, exist_ok=True)

    input_files = []
    for name, shape in shapes.items():
        data = rng.standard_normal(shape).astype(dt)
        file_path = os.path.join(output_dir, f"{name}.raw")
        data.tofile(file_path)
        input_files.append(file_path)

    return {"input_files": input_files, "seed": seed, "dtype": dtype}


@mcp.tool()
def run_numpy_reference(op_name: str, input_dir: str, output_dir: str,
                        params: dict = None) -> dict:
    """用 numpy 在 host 计算参考输出（fp32 精度，结果 cast 回 fp16）。

    Args:
        op_name: 算子名 (rmsnorm, swiglu, rope, softmax, matmul)
        input_dir: 包含 .raw 输入文件的目录
        output_dir: 输出 .raw 文件保存目录
        params: 算子参数 (如 eps)

    Returns:
        dict with output_files list and success status
    """
    if op_name not in NUMPY_OPS:
        return {"success": False, "output_files": [],
                "error": f"Unknown op: {op_name}. Supported: {list(NUMPY_OPS.keys())}"}

    os.makedirs(output_dir, exist_ok=True)
    params = params or {}

    try:
        if op_name == "rmsnorm":
            x = np.fromfile(os.path.join(input_dir, "input_x.raw"), dtype=np.float16)
            w = np.fromfile(os.path.join(input_dir, "weight.raw"), dtype=np.float16)
            # Reshape based on metadata
            meta_path = str(PROJECT_ROOT / "baseline" / "metadata.json")
            with open(meta_path) as f:
                meta = json.load(f)
            x_shape = meta["test_inputs"]["rmsnorm"]["shapes"]["input_x"]
            w_shape = meta["test_inputs"]["rmsnorm"]["shapes"]["weight"]
            x = x.reshape(x_shape)
            w = w.reshape(w_shape)
            eps = params.get("eps", 1e-6)
            out = _numpy_rmsnorm(x, w, eps)
            out_path = os.path.join(output_dir, "output.raw")
            out.tofile(out_path)
            return {"success": True, "output_files": [out_path],
                    "shape": list(out.shape), "dtype": "float16"}

        elif op_name == "swiglu":
            meta_path = str(PROJECT_ROOT / "baseline" / "metadata.json")
            with open(meta_path) as f:
                meta = json.load(f)
            gate = np.fromfile(os.path.join(input_dir, "gate.raw"), dtype=np.float16)
            up = np.fromfile(os.path.join(input_dir, "up.raw"), dtype=np.float16)
            gate = gate.reshape(meta["test_inputs"]["swiglu"]["shapes"]["gate"])
            up = up.reshape(meta["test_inputs"]["swiglu"]["shapes"]["up"])
            out = _numpy_swiglu(gate, up)
            out_path = os.path.join(output_dir, "output.raw")
            out.tofile(out_path)
            return {"success": True, "output_files": [out_path],
                    "shape": list(out.shape), "dtype": "float16"}

        elif op_name == "rope":
            meta_path = str(PROJECT_ROOT / "baseline" / "metadata.json")
            with open(meta_path) as f:
                meta = json.load(f)
            shapes = meta["test_inputs"]["rope"]["shapes"]
            q = np.fromfile(os.path.join(input_dir, "q.raw"), dtype=np.float16).reshape(shapes["q"])
            k = np.fromfile(os.path.join(input_dir, "k.raw"), dtype=np.float16).reshape(shapes["k"])
            cos_c = np.fromfile(os.path.join(input_dir, "cos_cache.raw"), dtype=np.float16).reshape(shapes["cos_cache"])
            sin_c = np.fromfile(os.path.join(input_dir, "sin_cache.raw"), dtype=np.float16).reshape(shapes["sin_cache"])
            q_out, k_out = _numpy_rope(q, k, cos_c, sin_c)
            q_path = os.path.join(output_dir, "q_out.raw")
            k_path = os.path.join(output_dir, "k_out.raw")
            q_out.tofile(q_path)
            k_out.tofile(k_path)
            return {"success": True, "output_files": [q_path, k_path],
                    "shapes": [list(q_out.shape), list(k_out.shape)], "dtype": "float16"}

        elif op_name == "softmax":
            meta_path = str(PROJECT_ROOT / "baseline" / "metadata.json")
            with open(meta_path) as f:
                meta = json.load(f)
            x = np.fromfile(os.path.join(input_dir, "input.raw"), dtype=np.float16)
            x = x.reshape(meta["test_inputs"]["softmax"]["shapes"]["input"])
            out = _numpy_softmax(x)
            out_path = os.path.join(output_dir, "output.raw")
            out.tofile(out_path)
            return {"success": True, "output_files": [out_path],
                    "shape": list(out.shape), "dtype": "float16"}

        elif op_name == "matmul":
            meta_path = str(PROJECT_ROOT / "baseline" / "metadata.json")
            with open(meta_path) as f:
                meta = json.load(f)
            shapes = meta["test_inputs"]["matmul"]["shapes"]
            a = np.fromfile(os.path.join(input_dir, "a.raw"), dtype=np.float16).reshape(shapes["a"])
            b = np.fromfile(os.path.join(input_dir, "b.raw"), dtype=np.float16).reshape(shapes["b"])
            out = _numpy_matmul(a, b)
            out_path = os.path.join(output_dir, "output.raw")
            out.tofile(out_path)
            return {"success": True, "output_files": [out_path],
                    "shape": list(out.shape), "dtype": "float16"}

    except Exception as e:
        return {"success": False, "output_files": [], "error": str(e)}


def _find_op_package_so(build_subdir: Path) -> Path | None:
    """Find the op-package .so in a per-arch build dir, excluding model libs.

    Op-package libs follow the convention `libQnnHtp<Op>.so`; model harness libs
    follow `libQnn<Op>Model.so`. We must filter the latter out — historically the
    runner wrote both into the same dir and `iterdir()` order decided which one
    got pushed as the stub (silu blocker root cause).
    """
    if not build_subdir.exists():
        return None
    candidates = [
        f for f in build_subdir.iterdir()
        if f.suffix == ".so" and "Model" not in f.name
    ]
    return candidates[0] if candidates else None


def _write_qnn_net_run_configs(local_dir: Path, graph_name: str) -> tuple[Path, Path]:
    """Generate the two config JSONs qnn-net-run needs for HTP graph compose.

    Without --config_file pointing at a backend_extensions block, HTP prepare
    rejects unknown graphs with `MODEL_GRAPH_ERROR` (the silu blocker). The
    minimal working pair: a top-level config that references the backend
    extensions library + a per-graph HTP config keyed by graph_name.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    htp_cfg = local_dir / "htp_config.json"
    htp_cfg.write_text(json.dumps({
        "graphs": [{
            "graph_names": [graph_name],
            "vtcm_mb": 8,
            "O": 3,
            "hvx_threads": 4,
            "fp16_relaxed_precision": 1,
        }],
    }, indent=2))

    top_cfg = local_dir / "net_run_config.json"
    top_cfg.write_text(json.dumps({
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": str(htp_cfg.name),
        },
    }, indent=2))
    return top_cfg, htp_cfg


@mcp.tool()
def run_custom_op_on_device(op_name: str, build_dir: str, input_dir: str,
                            output_dir: str, interface_provider: str = None) -> dict:
    """在设备 HTP 上执行自定义算子。

    需要：
    - build_dir 下有 build/hexagon-v81/<lib>.so 和 build/aarch64-android/<lib>.so
    - build_dir/model/<lib>.so（来自 build_single_op_model）
    - 设备上已有 qnn-net-run, libQnnHtp.so, libQnnHtpV81Stub.so, libQnnHtpPrepare.so,
      libQnnHtpNetRunExtensions.so

    设备布局:
        DEVICE_WORK_DIR/
            libQnn<Op>Model.so          # ARM model harness
            libQnnHtp<Op>.so            # ARM op-package stub
            net_run_config.json         # references backend_extensions
            htp_config.json             # graph_names binding
            htp/
                libQnnHtp<Op>.so        # Hexagon skel (loaded via ADSP_LIBRARY_PATH)

    Args:
        op_name: 算子名
        build_dir: 包含编译产物的目录 (ops/<op>/build/)
        input_dir: 包含 .raw 输入文件的目录
        output_dir: 本地输出目录
        interface_provider: op package 的 interface provider 函数名

    Returns:
        dict with success, output_files, latency_us, stderr
    """
    build_path = Path(build_dir)
    cap_name = op_name[0].upper() + op_name[1:]
    if not interface_provider:
        interface_provider = f"{cap_name}InterfaceProvider"

    # Discover op-package libs (filter out model harness libs)
    hexagon_so = _find_op_package_so(build_path / "build" / "hexagon-v81")
    aarch64_so = _find_op_package_so(build_path / "build" / "aarch64-android")
    if not hexagon_so or not aarch64_so:
        return {"success": False, "output_files": [], "latency_us": 0,
                "error": f"Missing op-package .so files. "
                         f"hex={hexagon_so}, arm={aarch64_so}"}

    # Discover model harness .so (preferred location: build/model/)
    model_so = build_path / "model" / f"libQnn{cap_name}Model.so"
    if not model_so.exists():
        # Legacy fallbacks
        for legacy in [build_path / "build" / "aarch64-android" / f"libQnn{cap_name}Model.so",
                       build_path / f"model_{op_name}.so",
                       build_path / f"model_{cap_name}.so"]:
            if legacy.exists():
                model_so = legacy
                break
    if not model_so.exists():
        return {"success": False, "output_files": [], "latency_us": 0,
                "error": f"Model .so not found at {model_so}. "
                         "Use build_single_op_model() to create it."}

    # Setup device dirs (ARM and Hex skel must NOT collide on shared name)
    htp_subdir = f"{DEVICE_WORK_DIR}/htp"
    _shell(f"rm -rf {DEVICE_WORK_DIR} && mkdir -p {htp_subdir}")

    # Push: ARM stub + model harness in work_dir, Hex skel in htp/ subdir
    _push(str(aarch64_so), f"{DEVICE_WORK_DIR}/{aarch64_so.name}")
    _push(str(hexagon_so), f"{htp_subdir}/{hexagon_so.name}")
    _push(str(model_so), f"{DEVICE_WORK_DIR}/{model_so.name}")

    # Generate and push qnn-net-run config files
    cfg_dir = build_path / "device_configs"
    graph_name = f"{cap_name}_graph"
    top_cfg, htp_cfg = _write_qnn_net_run_configs(cfg_dir, graph_name)
    _push(str(top_cfg), f"{DEVICE_WORK_DIR}/{top_cfg.name}")
    _push(str(htp_cfg), f"{DEVICE_WORK_DIR}/{htp_cfg.name}")

    # Push input files
    input_path = Path(input_dir)
    input_files_on_device = []
    for raw_file in sorted(input_path.glob("*.raw")):
        _push(str(raw_file), f"{DEVICE_WORK_DIR}/{raw_file.name}")
        input_files_on_device.append(f"{DEVICE_WORK_DIR}/{raw_file.name}")
    _shell(f"echo '{' '.join(input_files_on_device)}' > {DEVICE_WORK_DIR}/input_list.txt")

    # Run qnn-net-run.
    # Single op_packages entry pointing at the ARM stub; the HTP backend will
    # load the matching Hex skel via ADSP_LIBRARY_PATH (which points at htp/).
    op_pkg_arg = f"{DEVICE_WORK_DIR}/{aarch64_so.name}:{interface_provider}:HTP"
    cmd = (f"cd {DEVICE_WORK_DIR} && "
           f"export LD_LIBRARY_PATH=/data/local/tmp:{DEVICE_WORK_DIR}:$LD_LIBRARY_PATH && "
           f"export ADSP_LIBRARY_PATH=\"{htp_subdir};/data/local/tmp;/vendor/lib/rfsa/adsp;/dsp\" && "
           f"/data/local/tmp/qnn-net-run "
           f"--backend /data/local/tmp/libQnnHtp.so "
           f"--model {DEVICE_WORK_DIR}/{model_so.name} "
           f"--input_list {DEVICE_WORK_DIR}/input_list.txt "
           f"--output_dir {DEVICE_WORK_DIR}/output "
           f"--op_packages {op_pkg_arg} "
           # CRITICAL: without these, qnn-net-run treats .raw as fp32 by default,
           # corrupting fp16 input/output. See AGENTS.md L1.
           f"--use_native_input_files "
           f"--use_native_output_files")

    result = _shell(cmd, timeout=120)
    if result["rc"] != 0:
        return {"success": False, "output_files": [], "latency_us": 0,
                "stderr": result["err"], "stdout": result["out"],
                "command": cmd}

    # Pull outputs
    os.makedirs(output_dir, exist_ok=True)
    ls_result = _shell(f"ls {DEVICE_WORK_DIR}/output/Result_0/")
    output_files = []
    if ls_result["rc"] == 0:
        for fname in ls_result["out"].split():
            local_path = os.path.join(output_dir, fname)
            if _pull(f"{DEVICE_WORK_DIR}/output/Result_0/{fname}", local_path):
                output_files.append(local_path)

    # Parse latency from stdout
    latency_us = 0
    for line in result["out"].splitlines():
        if "execute" in line.lower() and "us" in line.lower():
            try:
                latency_us = int(''.join(c for c in line.split("us")[0].split()[-1] if c.isdigit()))
            except (ValueError, IndexError):
                pass

    return {"success": True, "output_files": output_files,
            "latency_us": latency_us, "stdout": result["out"]}


@mcp.tool()
def build_single_op_model(op_name: str, build_dir: str,
                          input_shapes: dict, output_shapes: dict,
                          op_package_name: str = None) -> dict:
    """为单个自定义算子生成并编译 model .so（用于 qnn-net-run）。

    生成一个最小的 C++ 文件，定义包含单个自定义 op 节点的 QNN graph，
    然后用 aarch64 NDK 编译为 .so。

    Args:
        op_name: 算子名 (如 "RMSNorm")
        build_dir: 输出目录 (ops/<op>/build/)
        input_shapes: {"input_name": [dims], ...}
        output_shapes: {"output_name": [dims], ...}
        op_package_name: op package 名 (如 "RMSNormPackage")

    Returns:
        dict with model_so path and success status
    """
    if not op_package_name:
        op_package_name = f"{op_name}Package"

    build_path = Path(build_dir)
    model_src_dir = build_path / "model_src"
    model_src_dir.mkdir(parents=True, exist_ok=True)

    # Generate model C++ source
    cpp_content = _gen_model_cpp(op_name, op_package_name,
                                 input_shapes, output_shapes)
    model_cpp = model_src_dir / f"model_{op_name.lower()}.cpp"
    model_cpp.write_text(cpp_content)

    # Compile for aarch64-android.
    # IMPORTANT: write the model .so into build/model/ — NOT build/aarch64-android/.
    # Putting it next to the op-package ARM stub broke run_custom_op_on_device's
    # .so discovery (silu blocker, May 2026).
    model_so_name = f"libQnn{op_name}Model.so"
    out_dir = build_path / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_so = out_dir / model_so_name

    ndk_clang = (f"{ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64"
                 f"/bin/clang++")
    sysroot = (f"{ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64"
               f"/sysroot")

    jni_dir = f"{QNN_SDK}/share/QNN/converter/jni"
    # Need to compile QnnModel.cpp, QnnWrapperUtils.cpp, and platform pal
    wrapper_srcs = [
        f"{jni_dir}/QnnModel.cpp",
        f"{jni_dir}/QnnWrapperUtils.cpp",
        f"{jni_dir}/linux/QnnModelPal.cpp",
    ]

    compile_cmd = [
        ndk_clang,
        "--target=aarch64-none-linux-android21",
        f"--sysroot={sysroot}",
        "-stdlib=libc++", "-static-libstdc++",
        "-std=c++17", "-fPIC", "-shared",
        "-O2", "-fvisibility=default",
        f"-I{QNN_SDK}/include/QNN",
        f"-I{jni_dir}",
        '-DQNN_API=__attribute__((visibility("default")))',
        "-Wno-sign-compare", "-Wno-unused-variable",
        "-Wno-unused-parameter",
        str(model_cpp),
    ] + wrapper_srcs + [
        "-o", str(model_so),
        f"-L{QNN_SDK}/lib/aarch64-android",
    ]

    try:
        result = subprocess.run(compile_cmd, capture_output=True,
                                text=True, timeout=60)
        if result.returncode != 0:
            return {"success": False, "model_so": "",
                    "error": result.stderr, "command": " ".join(compile_cmd)}
        return {"success": True, "model_so": str(model_so),
                "source": str(model_cpp)}
    except Exception as e:
        return {"success": False, "model_so": "", "error": str(e)}


def _gen_model_cpp(op_name: str, package_name: str,
                   input_shapes: dict, output_shapes: dict) -> str:
    """Generate minimal model .so C++ source for a single custom op."""
    dtype_enum = "QNN_DATATYPE_FLOAT_16"

    lines = [
        '// Auto-generated single-op model for qnn-net-run testing',
        '#include "QnnModel.hpp"',
        '#include "QnnOpDef.h"',
        '',
        '#define DO_GRAPH_NODE_VALIDATIONS 1',
        'using namespace qnn_wrapper_api;',
        '',
        'extern "C" {',
        'QNN_API',
        'ModelError_t QnnModel_composeGraphs(',
        '    Qnn_BackendHandle_t backendHandle,',
        '    QNN_INTERFACE_VER_TYPE interface,',
        '    Qnn_ContextHandle_t contextHandle,',
        '    const GraphConfigInfo_t** graphsConfigInfo,',
        '    const uint32_t numGraphsConfigInfo,',
        '    GraphInfoPtr_t** graphsInfo,',
        '    uint32_t* numGraphsInfo,',
        '    bool debug,',
        '    QnnLog_Callback_t logCallback,',
        '    QnnLog_Level_t maxLogLevel) {',
        '  ModelError_t err = MODEL_NO_ERROR;',
        '  QnnModel model;',
        '  const QnnGraph_Config_t** graphConfigs = nullptr;',
        f'  VALIDATE(getQnnGraphConfigFromInfo("{op_name}_graph",',
        '      graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);',
        f'  VALIDATE(model.initialize(backendHandle, interface, contextHandle,',
        f'      "{op_name}_graph", debug, DO_GRAPH_NODE_VALIDATIONS,',
        '      graphConfigs), err);',
        '',
    ]

    # Add input tensors
    all_input_names = []
    for name, shape in input_shapes.items():
        dims_str = ", ".join(str(d) for d in shape)
        lines.append(f'  uint32_t dims_{name}[] = {{{dims_str}}};')
        lines.append(f'  VALIDATE(model.addTensor("{name}",')
        lines.append('    (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,')
        lines.append(f'      .v1 = {{.id = 0, .name = "{name}",')
        lines.append('        .type = QNN_TENSOR_TYPE_APP_WRITE,')
        lines.append('        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,')
        lines.append(f'        .dataType = {dtype_enum},')
        lines.append('        .quantizeParams = {QNN_DEFINITION_UNDEFINED,')
        lines.append('          QNN_QUANTIZATION_ENCODING_UNDEFINED,')
        lines.append('          {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},')
        lines.append(f'        .rank = {len(shape)}, .dimensions = dims_{name},')
        lines.append('        .memType = QNN_TENSORMEMTYPE_RAW,')
        lines.append('        .clientBuf = {.data = nullptr, .dataSize = 0}}}), err);')
        lines.append('')
        all_input_names.append(name)

    # Output tensors: only declare dimension arrays (addNode will create the tensors)
    all_output_names = []
    for name, shape in output_shapes.items():
        dims_str = ", ".join(str(d) for d in shape)
        lines.append(f'  uint32_t dims_{name}[] = {{{dims_str}}};')
        lines.append('')
        all_output_names.append(name)

    # Add the op node using model.addNode
    # API: addNode(version, name, packageName, type, params, numParams,
    #              inputNames, numInputs, outputTensors, numOutputs)
    n_inputs = len(all_input_names)
    n_outputs = len(all_output_names)

    # Input names array (inputs referenced by name from addTensor)
    lines.append(f'  const char* inputNames[] = {{')
    for name in all_input_names:
        lines.append(f'    "{name}",')
    lines.append('  };')
    lines.append('')

    # Output tensors array (full Qnn_Tensor_t for outputs)
    lines.append(f'  Qnn_Tensor_t outputTensors[{n_outputs}];')
    for i, (name, shape) in enumerate(output_shapes.items()):
        lines.append(f'  outputTensors[{i}].version = QNN_TENSOR_VERSION_1;')
        lines.append(f'  outputTensors[{i}].v1.id = 0;')
        lines.append(f'  outputTensors[{i}].v1.name = "{name}";')
        lines.append(f'  outputTensors[{i}].v1.type = QNN_TENSOR_TYPE_APP_READ;')
        lines.append(f'  outputTensors[{i}].v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;')
        lines.append(f'  outputTensors[{i}].v1.dataType = {dtype_enum};')
        lines.append(f'  outputTensors[{i}].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;')
        lines.append(f'  outputTensors[{i}].v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;')
        lines.append(f'  outputTensors[{i}].v1.rank = {len(shape)};')
        lines.append(f'  outputTensors[{i}].v1.dimensions = dims_{name};')
        lines.append(f'  outputTensors[{i}].v1.memType = QNN_TENSORMEMTYPE_RAW;')
        lines.append(f'  outputTensors[{i}].v1.clientBuf.data = nullptr;')
        lines.append(f'  outputTensors[{i}].v1.clientBuf.dataSize = 0;')
    lines.append('')

    # Add node (op) using the wrapper API
    lines.append(f'  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,')
    lines.append(f'      "{op_name}_node",  // node name')
    lines.append(f'      "{package_name}",  // package name')
    lines.append(f'      "{op_name}",       // op type')
    lines.append(f'      nullptr,           // params')
    lines.append(f'      0,                 // numParams')
    lines.append(f'      inputNames,        // input tensor names')
    lines.append(f'      {n_inputs},        // numInputs')
    lines.append(f'      outputTensors,     // output tensors')
    lines.append(f'      {n_outputs}        // numOutputs')
    lines.append(f'  ), err);')
    lines.append('')
    lines.append('  VALIDATE(model.finalize(graphsInfo, numGraphsInfo), err);')
    lines.append('  return err;')
    lines.append('}')
    lines.append('')

    # Add freeGraphsInfo
    lines.append('QNN_API')
    lines.append('ModelError_t QnnModel_freeGraphsInfo(')
    lines.append('    GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo) {')
    lines.append('  return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);')
    lines.append('}')
    lines.append('} // extern "C"')

    return "\n".join(lines)


@mcp.tool()
def ensure_device_runtime() -> dict:
    """确保设备上有 qnn-net-run 和必要的 QNN 运行时库。

    检查并 push 缺失的文件到 /data/local/tmp/。
    """
    required_files = {
        "qnn-net-run": f"{QNN_SDK}/bin/aarch64-android/qnn-net-run",
        "libQnnHtp.so": f"{QNN_SDK}/lib/aarch64-android/libQnnHtp.so",
        "libQnnHtpPrepare.so": f"{QNN_SDK}/lib/aarch64-android/libQnnHtpPrepare.so",
        "libQnnHtpV81Stub.so": f"{QNN_SDK}/lib/aarch64-android/libQnnHtpV81Stub.so",
        "libQnnSystem.so": f"{QNN_SDK}/lib/aarch64-android/libQnnSystem.so",
        "libQnnHtpNetRunExtensions.so": f"{QNN_SDK}/lib/aarch64-android/libQnnHtpNetRunExtensions.so",
    }

    pushed = []
    already_present = []
    failed = []

    for name, local_path in required_files.items():
        # Check if already on device
        check = _shell(f"ls /data/local/tmp/{name} 2>/dev/null")
        if check["rc"] == 0 and name in check["out"]:
            already_present.append(name)
            continue
        # Push it
        if os.path.exists(local_path):
            if _push(local_path, f"/data/local/tmp/{name}"):
                pushed.append(name)
                if name == "qnn-net-run":
                    _shell("chmod +x /data/local/tmp/qnn-net-run")
            else:
                failed.append(name)
        else:
            failed.append(f"{name} (not found at {local_path})")

    return {"success": len(failed) == 0,
            "pushed": pushed, "already_present": already_present,
            "failed": failed}


if __name__ == "__main__":
    mcp.run()
