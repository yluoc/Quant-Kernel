"""Project-wide API parity guardrails.

These tests intentionally inspect source files to catch integration drift:
- new scalar APIs without batch counterparts
- missing accelerator native-batch routing
- missing C API/C++ implementation parity
- missing test wiring for newly added APIs
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = ROOT / "python/quantkernel/engine.py"
STUB_PATH = ROOT / "python/quantkernel/__init__.pyi"
ACCEL_PATH = ROOT / "python/quantkernel/accelerator.py"
H_PATH = ROOT / "cpp/include/quantkernel/qk_api.h"
CPP_PATH = ROOT / "cpp/src/qk_api.cpp"
TESTS_DIR = ROOT / "python/tests"
_ENGINE_NON_BATCH_METHODS = {"abi_version", "plugin_get_api", "get_last_error", "clear_last_error"}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _engine_public_methods() -> tuple[list[str], list[str]]:
    text = _read(ENGINE_PATH)
    methods = re.findall(r"^    def ([a-zA-Z0-9_]+)\(", text, re.M)
    public = [
        m for m in methods
        if not m.startswith("_") and m not in {"get_accelerator", "price_batch", "native_batch_available"}
    ]
    scalars = sorted([m for m in public if not m.endswith("_batch")])
    batches = sorted([m for m in public if m.endswith("_batch")])
    return scalars, batches


def _stub_public_methods() -> tuple[set[str], set[str]]:
    text = _read(STUB_PATH)
    methods = set(re.findall(r"^\s{4}def ([a-zA-Z0-9_]+)\(", text, re.M))
    methods.discard("__init__")
    methods.discard("native_batch_available")
    methods.discard("get_accelerator")
    methods.discard("price_batch")
    scalars = {m for m in methods if not m.endswith("_batch")}
    batches = {m for m in methods if m.endswith("_batch")}
    return scalars, batches


def _accelerator_native_map() -> dict[str, str]:
    tree = ast.parse(_read(ACCEL_PATH))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "QuantAccelerator":
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == "_NATIVE_BATCH_METHODS"
                    and isinstance(stmt.value, ast.Dict)
                ):
                    out: dict[str, str] = {}
                    for k, v in zip(stmt.value.keys, stmt.value.values):
                        if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                            if isinstance(k.value, str) and isinstance(v.value, str):
                                out[k.value] = v.value
                    return out
    raise AssertionError("Could not locate QuantAccelerator._NATIVE_BATCH_METHODS")


def _c_api_funcs() -> tuple[set[str], set[str]]:
    h_text = _read(H_PATH)
    cpp_text = _read(CPP_PATH)
    h_funcs = set(re.findall(r"QK_EXPORT\s+(?:double|int32_t|void|const char\*)\s+(qk_[a-z0-9_]+)\s*\(", h_text))
    cpp_funcs = set(re.findall(r"\b(?:double|int32_t|void|const char\*)\s+(qk_[a-z0-9_]+)\s*\(", cpp_text))
    return h_funcs, cpp_funcs


def _all_python_test_text() -> str:
    parts: list[str] = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        if path.name in {"test_parity_guard.py", "test_perf_regression.py"}:
            continue
        parts.append(_read(path))
    return "\n".join(parts)


def test_engine_scalar_batch_pairs_are_complete() -> None:
    scalars, batches = _engine_public_methods()
    batch_set = set(batches)
    missing = [m for m in scalars if m not in _ENGINE_NON_BATCH_METHODS and f"{m}_batch" not in batch_set]
    assert not missing, f"Engine scalar APIs missing *_batch: {missing}"


def test_engine_and_stub_signatures_stay_in_sync() -> None:
    engine_scalars, engine_batches = _engine_public_methods()
    stub_scalars, stub_batches = _stub_public_methods()

    missing_stub_scalars = sorted(set(engine_scalars) - stub_scalars)
    missing_stub_batches = sorted(set(engine_batches) - stub_batches)

    assert not missing_stub_scalars, f"Stub missing scalar APIs: {missing_stub_scalars}"
    assert not missing_stub_batches, f"Stub missing batch APIs: {missing_stub_batches}"


def test_accelerator_native_batch_map_covers_all_scalar_methods() -> None:
    engine_scalars, _ = _engine_public_methods()
    native_map = _accelerator_native_map()
    routed_scalars = [m for m in engine_scalars if m not in _ENGINE_NON_BATCH_METHODS]

    missing = sorted([m for m in routed_scalars if m not in native_map])
    wrong_target = sorted([m for m in routed_scalars if native_map.get(m) != f"{m}_batch"])

    assert not missing, f"Accelerator _NATIVE_BATCH_METHODS missing scalar APIs: {missing}"
    assert not wrong_target, f"Accelerator _NATIVE_BATCH_METHODS has wrong batch target(s): {wrong_target}"


def test_c_api_header_and_cpp_implementation_stay_in_sync() -> None:
    h_funcs, cpp_funcs = _c_api_funcs()

    missing_cpp_impl = sorted([f for f in h_funcs if f not in cpp_funcs])
    extra_cpp_impl = sorted([
        f for f in cpp_funcs
        if f not in h_funcs and f not in {"qk_plugin_get_api", "qk_get_last_error", "qk_clear_last_error"}
    ])

    assert not missing_cpp_impl, f"C API declared but not implemented: {missing_cpp_impl}"
    assert not extra_cpp_impl, f"C API implemented but not declared: {extra_cpp_impl}"


def test_c_api_scalar_batch_pairs_are_complete() -> None:
    h_funcs, _ = _c_api_funcs()
    scalar = sorted([
        f for f in h_funcs
        if f.startswith("qk_")
        and not f.endswith("_batch")
        and f not in {"qk_abi_version", "qk_plugin_get_api", "qk_get_last_error", "qk_clear_last_error"}
    ])

    missing = sorted([f for f in scalar if f"{f}_batch" not in h_funcs])
    assert not missing, f"C API scalar functions missing *_batch variants: {missing}"


def test_accelerator_vectorized_methods_all_implemented() -> None:
    """Every method in _VECTORIZED_METHODS must be handled in _vectorized_price source."""
    accel_text = _read(ACCEL_PATH)
    tree = ast.parse(accel_text)
    vec_methods: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "QuantAccelerator":
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == "_VECTORIZED_METHODS"
                    and isinstance(stmt.value, ast.Set)
                ):
                    for elt in stmt.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            vec_methods.add(elt.value)
    assert vec_methods, "Could not locate _VECTORIZED_METHODS"

    missing = sorted(m for m in vec_methods if f'method == "{m}"' not in accel_text)
    assert not missing, f"_VECTORIZED_METHODS without _vectorized_price branch: {missing}"


def test_every_engine_api_has_explicit_python_test_wiring() -> None:
    """Fail when new APIs land without test references.

    This is intentionally strict: each public engine scalar+batch method should be
    referenced in at least one python test module.
    """
    scalars, batches = _engine_public_methods()
    corpus = _all_python_test_text()

    missing_scalars = sorted([m for m in scalars if m not in corpus])

    def _batch_root_name(batch_method: str) -> str:
        base = batch_method[:-6]
        return base[:-6] if base.endswith("_price") else base

    missing_batches = sorted([
        m for m in batches
        if (m not in corpus and _batch_root_name(m) not in corpus)
    ])

    assert not missing_scalars, f"Scalar APIs not referenced in python tests: {missing_scalars}"
    assert not missing_batches, f"Batch APIs not referenced in python tests: {missing_batches}"
