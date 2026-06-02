"""tensor-compare MCP server — 数值对比 + ppl 评估"""
import numpy as np
from fastmcp import FastMCP

mcp = FastMCP("tensor-compare")


def _load_tensor(path: str, dtype: str = "float16") -> np.ndarray:
    dt = {"float16": np.float16, "float32": np.float32, "int8": np.int8, "uint8": np.uint8}
    return np.fromfile(path, dtype=dt.get(dtype, np.float16))


@mcp.tool()
def compare(a_path: str, b_path: str, dtype: str = "float16") -> dict:
    """计算两个 tensor 的数值对比指标（cosine similarity, max abs error, relative error）。"""
    a = _load_tensor(a_path, dtype).astype(np.float64)
    b = _load_tensor(b_path, dtype).astype(np.float64)

    if a.shape != b.shape:
        return {"cosine_similarity": 0.0, "max_abs_error": float("inf"),
                "relative_error": float("inf"), "shapes_match": False,
                "shape_a": list(a.shape), "shape_b": list(b.shape)}

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine = float(np.dot(a, b) / (norm_a * norm_b + 1e-12))
    max_abs = float(np.max(np.abs(a - b)))
    rel_err = float(np.mean(np.abs(a - b) / (np.abs(b) + 1e-8)))

    return {"cosine_similarity": cosine, "max_abs_error": max_abs,
            "relative_error": rel_err, "shapes_match": True,
            "shape_a": list(a.shape), "shape_b": list(b.shape)}


@mcp.tool()
def compare_pass(a_path: str, b_path: str, dtype: str = "float16",
                 cosine_min: float = 0.999, max_abs_max: float = 1e-3) -> dict:
    """对比并判定是否通过验收标准。返回 passed 和 failures 列表。"""
    metrics = compare(a_path, b_path, dtype)
    if not metrics["shapes_match"]:
        return {"passed": False, "failures": [{"metric": "shape", "actual": "mismatch",
                "threshold": "match"}], "metrics": metrics}

    failures = []
    if metrics["cosine_similarity"] < cosine_min:
        failures.append({"metric": "cosine_similarity",
                         "actual": metrics["cosine_similarity"], "threshold": cosine_min})
    if metrics["max_abs_error"] > max_abs_max:
        failures.append({"metric": "max_abs_error",
                         "actual": metrics["max_abs_error"], "threshold": max_abs_max})

    return {"passed": len(failures) == 0, "failures": failures, "metrics": metrics}


@mcp.tool()
def e2e_ppl(model_pte_path: str, dataset_path: str, max_samples: int = 100,
            baseline_ppl: float = 0.0) -> dict:
    """评估端到端 perplexity 并与 baseline 对比。
    注意：当前为 stub 实现，实际需要在设备上运行模型。"""
    # TODO: 实际实现需要调用设备端 runner 计算 ppl
    return {"ppl": 0.0, "samples_run": 0, "ppl_delta_pct": None,
            "passed": None, "note": "stub - not yet implemented"}


if __name__ == "__main__":
    mcp.run()
