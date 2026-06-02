"""qnn-adb MCP server — 设备交互（push/pull/shell/health）"""
import os
import subprocess
import time
from fastmcp import FastMCP

mcp = FastMCP("qnn-adb")

def _resolve_serial() -> str:
    """每次调用都重新解析 serial：env > 单台设备自动选择。"""
    env_serial = (os.environ.get("ANDROID_SERIAL") or "").strip()
    if env_serial:
        return env_serial
    try:
        out = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""
    devices = [line.split("\t")[0] for line in out.splitlines()[1:]
               if line.strip() and "\tdevice" in line]
    return devices[0] if len(devices) == 1 else ""


def _run_adb(args: list, timeout: int = 60) -> dict:
    serial = _resolve_serial()
    cmd = ["adb"] + (["-s", serial] if serial else []) + args
    # 清理空 ANDROID_SERIAL，避免 adb 当成 "查找空 serial 设备" 失败
    env = {k: v for k, v in os.environ.items() if not (k == "ANDROID_SERIAL" and not v)}
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": f"timeout after {timeout}s"}
    except Exception as e:
        return {"exit_code": -1, "stdout": "", "stderr": str(e)}


@mcp.tool()
def adb_push(local_path: str, remote_path: str) -> dict:
    """推送文件到 Device。"""
    r = _run_adb(["push", local_path, remote_path], timeout=120)
    return {"success": r["exit_code"] == 0, "stderr": r["stderr"]}


@mcp.tool()
def adb_pull(remote_path: str, local_path: str) -> dict:
    """从 Device 拉取文件。"""
    r = _run_adb(["pull", remote_path, local_path], timeout=120)
    return {"success": r["exit_code"] == 0, "stderr": r["stderr"]}


@mcp.tool()
def adb_shell(command: str, timeout: int = 60) -> dict:
    """在 Device 上执行 shell 命令。强制超时。"""
    r = _run_adb(["shell", command], timeout=timeout)
    return {"exit_code": r["exit_code"], "stdout": r["stdout"], "stderr": r["stderr"]}


@mcp.tool()
def device_health() -> dict:
    """查询设备健康状态。热节流时阻塞等待（最长 5min）。连续 5 次失败报告不可用。"""
    for attempt in range(5):
        r = _run_adb(["shell", "echo ok"], timeout=10)
        if r["exit_code"] == 0:
            break
        time.sleep(5)
    else:
        return {"available": False, "thermal_state": "unknown",
                "battery_pct": -1, "serial": _resolve_serial(), "soc": "unknown"}

    # 查询热状态
    thermal = _run_adb(["shell", "cat /sys/class/thermal/thermal_zone0/temp"], timeout=10)
    temp_c = -1
    try:
        temp_c = int(thermal["stdout"].strip()) // 1000
    except (ValueError, AttributeError):
        pass

    thermal_state = "normal"
    if temp_c > 45:
        thermal_state = "warm"
    if temp_c > 55:
        thermal_state = "throttling"
        # 等待恢复（最长 5min）
        for _ in range(30):
            time.sleep(10)
            thermal = _run_adb(["shell", "cat /sys/class/thermal/thermal_zone0/temp"], timeout=10)
            try:
                temp_c = int(thermal["stdout"].strip()) // 1000
            except (ValueError, AttributeError):
                break
            if temp_c <= 45:
                thermal_state = "normal"
                break

    # 查询电量
    battery = _run_adb(["shell", "dumpsys battery | grep level"], timeout=10)
    battery_pct = -1
    try:
        battery_pct = int(battery["stdout"].split(":")[-1].strip())
    except (ValueError, IndexError, AttributeError):
        pass

    return {"available": True, "thermal_state": thermal_state,
            "battery_pct": battery_pct, "serial": _resolve_serial(), "soc": "SM8850",
            "temp_c": temp_c}


if __name__ == "__main__":
    mcp.run()
