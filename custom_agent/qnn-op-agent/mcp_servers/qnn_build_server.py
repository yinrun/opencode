"""qnn-build MCP server — IDL 验证 + op package 生成 + NDK 编译"""
import os
import subprocess
from pathlib import Path
from fastmcp import FastMCP

mcp = FastMCP("qnn-build")

QNN_SDK = os.environ.get("QNN_SDK_242", "/home/yinrun/software/qualcomm/qairt/2.42.0.251225")
HEXAGON_SDK = os.environ.get("HEXAGON_SDK_ROOT", "")
ANDROID_NDK = os.environ.get("ANDROID_NDK_ROOT", "")


def _summarize_log(output: str, max_tail: int = 50) -> str:
    lines = output.splitlines()
    error_lines = [l for l in lines if any(k in l.lower() for k in ["error", "fatal", "undefined"])]
    tail = lines[-max_tail:] if len(lines) > max_tail else lines
    summary_parts = []
    if error_lines:
        summary_parts.append("=== ERRORS ===\n" + "\n".join(error_lines[:20]))
    summary_parts.append("=== LAST LINES ===\n" + "\n".join(tail))
    return "\n".join(summary_parts)


@mcp.tool()
def validate_idl(idl_path: str) -> dict:
    """验证 IDL（XML OpDef）定义的合法性。"""
    p = Path(idl_path)
    if not p.exists():
        return {"valid": False, "errors": [f"File not found: {idl_path}"]}

    content = p.read_text()
    errors = []
    if "<OpDefCollection" not in content:
        errors.append("Missing <OpDefCollection> root element")
    if "<Name>" not in content:
        errors.append("Missing <Name> element in OpDef")
    if "<Input>" not in content:
        errors.append("Missing <Input> element")
    if "<Output>" not in content:
        errors.append("Missing <Output> element")

    return {"valid": len(errors) == 0, "errors": errors}


@mcp.tool()
def gen_op_package(idl_path: str, output_dir: str, op_name: str) -> dict:
    """调用 qnn-op-package-generator 生成 op package 骨架。超时 5min。"""
    generator = os.path.join(QNN_SDK, "bin/x86_64-linux-clang/qnn-op-package-generator")
    if not os.path.exists(generator):
        return {"success": False, "package_dir": "", "log_summary": f"Generator not found: {generator}"}

    os.makedirs(output_dir, exist_ok=True)
    try:
        result = subprocess.run(
            [generator, "-p", idl_path, "-o", output_dir],
            capture_output=True, text=True, timeout=300)
        log = result.stdout + "\n" + result.stderr
        log_file = os.path.join(output_dir, f"gen_{op_name}.log")
        Path(log_file).write_text(log)
        return {"success": result.returncode == 0, "package_dir": output_dir,
                "log_summary": _summarize_log(log), "log_file": log_file}
    except subprocess.TimeoutExpired:
        return {"success": False, "package_dir": output_dir,
                "log_summary": "TIMEOUT: gen_op_package exceeded 5min"}


@mcp.tool()
def ndk_build(package_dir: str, target: str, sdk_version: str = "2.42") -> dict:
    """为指定目标编译 op package。target: hexagon-v81 | aarch64-android | x86_64-linux-clang。超时 5min。"""
    sdk = QNN_SDK if sdk_version == "2.42" else os.environ.get("QNN_SDK_242", QNN_SDK)

    target_map = {"hexagon-v81": "htp_v81", "aarch64-android": "htp_aarch64", "x86_64-linux-clang": "htp_x86"}
    make_target = target_map.get(target, target)

    env = os.environ.copy()
    env["QNN_SDK_ROOT"] = sdk
    if HEXAGON_SDK:
        env["HEXAGON_SDK_ROOT"] = HEXAGON_SDK
    if ANDROID_NDK:
        env["ANDROID_NDK_ROOT"] = ANDROID_NDK

    try:
        result = subprocess.run(
            ["make", make_target], cwd=package_dir,
            capture_output=True, text=True, timeout=300, env=env)
        log = result.stdout + "\n" + result.stderr

        # 持久化完整日志
        log_dir = os.path.join(package_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"build_{target}.log")
        Path(log_file).write_text(log)

        # 检查产物
        so_path = ""
        build_dir = os.path.join(package_dir, "build", target)
        if os.path.isdir(build_dir):
            for f in os.listdir(build_dir):
                if f.endswith(".so"):
                    so_path = os.path.join(build_dir, f)
                    break

        # 检查未定义符号
        undefined = []
        if so_path and os.path.exists(so_path):
            nm_result = subprocess.run(["nm", "-u", so_path], capture_output=True, text=True, timeout=30)
            undefined = [l.strip() for l in nm_result.stdout.splitlines() if l.strip()]

        return {"success": result.returncode == 0, "so_path": so_path,
                "log_summary": _summarize_log(log), "log_file": log_file,
                "undefined_symbols": undefined[:20]}
    except subprocess.TimeoutExpired:
        return {"success": False, "so_path": "", "log_summary": "TIMEOUT: ndk_build exceeded 5min",
                "log_file": "", "undefined_symbols": []}


if __name__ == "__main__":
    mcp.run()
