"""Library finder and ABI version check."""

import ctypes as ct
import os
import sys
from pathlib import Path

from ._abi import (
    ABI_MAJOR,
    ABI_MINOR,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _search_dirs() -> list[Path]:
    search_dirs: list[Path] = []

    env_path = os.environ.get("QK_LIB_PATH")
    if env_path:
        search_dirs.append(Path(env_path))

    root = _project_root()
    search_dirs.append(root / "build" / "cpp")
    search_dirs.append(root / "build")
    search_dirs.append(root)
    return search_dirs


def _kernel_names() -> list[str]:
    if sys.platform == "win32":
        return ["quantkernel.dll", "libquantkernel.dll"]
    if sys.platform == "darwin":
        return ["libquantkernel.dylib"]
    return ["libquantkernel.so"]


def _find_library(names: list[str], search_dirs: list[Path]) -> Path | None:
    for d in search_dirs:
        for name in names:
            candidate = d / name
            if candidate.exists():
                return candidate
    return None


def _find_library_or_raise(names: list[str], search_dirs: list[Path], purpose: str) -> Path:
    found = _find_library(names, search_dirs)
    if found is not None:
        return found

    raise OSError(
        f"Cannot find {purpose} shared library. "
        f"Searched: {[str(d) for d in search_dirs]}. "
        f"Set QK_LIB_PATH and build first."
    )


def _configure_function_signatures(lib: ct.CDLL) -> None:
    lib.qk_abi_version.restype = None
    lib.qk_abi_version.argtypes = [ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32)]
    
    lib.qk_cf_black_scholes_merton_price.restype = ct.c_double
    lib.qk_cf_black_scholes_merton_price.argtypes = [
        ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_double, ct.c_double, ct.c_int32,
    ]
    lib.qk_cf_black76_price.restype = ct.c_double
    lib.qk_cf_black76_price.argtypes = [
        ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int32,
    ]
    lib.qk_cf_bachelier_price.restype = ct.c_double
    lib.qk_cf_bachelier_price.argtypes = [
        ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int32,
    ]
    lib.qk_cf_heston_price_cf.restype = ct.c_double
    lib.qk_cf_heston_price_cf.argtypes = [
        ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_int32, ct.c_int32, ct.c_double,
    ]
    lib.qk_cf_merton_jump_diffusion_price.restype = ct.c_double
    lib.qk_cf_merton_jump_diffusion_price.argtypes = [
        ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32,
    ]
    lib.qk_cf_variance_gamma_price_cf.restype = ct.c_double
    lib.qk_cf_variance_gamma_price_cf.argtypes = [
        ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_double,
    ]
    lib.qk_cf_sabr_hagan_lognormal_iv.restype = ct.c_double
    lib.qk_cf_sabr_hagan_lognormal_iv.argtypes = [
        ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
    ]
    lib.qk_cf_sabr_hagan_black76_price.restype = ct.c_double
    lib.qk_cf_sabr_hagan_black76_price.argtypes = [
        ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int32,
    ]
    lib.qk_cf_dupire_local_vol.restype = ct.c_double
    lib.qk_cf_dupire_local_vol.argtypes = [
        ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_double, ct.c_double, ct.c_double, ct.c_double,
    ]

    # --- Fourier Transform methods ---
    ftm_common = [
        ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_double, ct.c_double, ct.c_int32,
    ]
    lib.qk_ftm_carr_madan_fft_price.restype = ct.c_double
    lib.qk_ftm_carr_madan_fft_price.argtypes = ftm_common + [ct.c_int32, ct.c_double, ct.c_double]
    lib.qk_ftm_cos_fang_oosterlee_price.restype = ct.c_double
    lib.qk_ftm_cos_fang_oosterlee_price.argtypes = ftm_common + [ct.c_int32, ct.c_double]
    lib.qk_ftm_fractional_fft_price.restype = ct.c_double
    lib.qk_ftm_fractional_fft_price.argtypes = ftm_common + [ct.c_int32, ct.c_double, ct.c_double, ct.c_double]
    lib.qk_ftm_lewis_fourier_inversion_price.restype = ct.c_double
    lib.qk_ftm_lewis_fourier_inversion_price.argtypes = ftm_common + [ct.c_int32, ct.c_double]
    lib.qk_ftm_hilbert_transform_price.restype = ct.c_double
    lib.qk_ftm_hilbert_transform_price.argtypes = ftm_common + [ct.c_int32, ct.c_double]

    # --- Integral Quadrature methods ---
    iqm_common = [
        ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_double, ct.c_double, ct.c_int32,
    ]
    lib.qk_iqm_gauss_hermite_price.restype = ct.c_double
    lib.qk_iqm_gauss_hermite_price.argtypes = iqm_common + [ct.c_int32]
    lib.qk_iqm_gauss_laguerre_price.restype = ct.c_double
    lib.qk_iqm_gauss_laguerre_price.argtypes = iqm_common + [ct.c_int32]
    lib.qk_iqm_gauss_legendre_price.restype = ct.c_double
    lib.qk_iqm_gauss_legendre_price.argtypes = iqm_common + [ct.c_int32, ct.c_double]
    lib.qk_iqm_adaptive_quadrature_price.restype = ct.c_double
    lib.qk_iqm_adaptive_quadrature_price.argtypes = iqm_common + [ct.c_double, ct.c_double, ct.c_int32, ct.c_double]

    tlm_argtypes = [
        ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32,
    ]
    lib.qk_tlm_crr_price.restype = ct.c_double
    lib.qk_tlm_crr_price.argtypes = tlm_argtypes
    lib.qk_tlm_jarrow_rudd_price.restype = ct.c_double
    lib.qk_tlm_jarrow_rudd_price.argtypes = tlm_argtypes
    lib.qk_tlm_tian_price.restype = ct.c_double
    lib.qk_tlm_tian_price.argtypes = tlm_argtypes
    lib.qk_tlm_leisen_reimer_price.restype = ct.c_double
    lib.qk_tlm_leisen_reimer_price.argtypes = tlm_argtypes
    lib.qk_tlm_trinomial_tree_price.restype = ct.c_double
    lib.qk_tlm_trinomial_tree_price.argtypes = tlm_argtypes
    lib.qk_tlm_derman_kani_const_local_vol_price.restype = ct.c_double
    lib.qk_tlm_derman_kani_const_local_vol_price.argtypes = tlm_argtypes

    # --- Finite Difference methods ---
    fdm_argtypes = [
        ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32,
        ct.c_int32,
    ]
    lib.qk_fdm_explicit_fd_price.restype = ct.c_double
    lib.qk_fdm_explicit_fd_price.argtypes = fdm_argtypes
    lib.qk_fdm_implicit_fd_price.restype = ct.c_double
    lib.qk_fdm_implicit_fd_price.argtypes = fdm_argtypes
    lib.qk_fdm_crank_nicolson_price.restype = ct.c_double
    lib.qk_fdm_crank_nicolson_price.argtypes = fdm_argtypes

    adi_argtypes = [
        ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32,
    ]
    lib.qk_fdm_adi_douglas_price.restype = ct.c_double
    lib.qk_fdm_adi_douglas_price.argtypes = adi_argtypes
    lib.qk_fdm_adi_craig_sneyd_price.restype = ct.c_double
    lib.qk_fdm_adi_craig_sneyd_price.argtypes = adi_argtypes
    lib.qk_fdm_adi_hundsdorfer_verwer_price.restype = ct.c_double
    lib.qk_fdm_adi_hundsdorfer_verwer_price.argtypes = adi_argtypes

    lib.qk_fdm_psor_price.restype = ct.c_double
    lib.qk_fdm_psor_price.argtypes = [
        ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32,
        ct.c_double, ct.c_double, ct.c_int32,
    ]

    # --- Monte Carlo methods ---
    mc_common = [
        ct.c_double, ct.c_double, ct.c_double, ct.c_double,
        ct.c_double, ct.c_double, ct.c_int32,
    ]
    lib.qk_mcm_standard_monte_carlo_price.restype = ct.c_double
    lib.qk_mcm_standard_monte_carlo_price.argtypes = mc_common + [ct.c_int32, ct.c_uint64]
    lib.qk_mcm_euler_maruyama_price.restype = ct.c_double
    lib.qk_mcm_euler_maruyama_price.argtypes = mc_common + [ct.c_int32, ct.c_int32, ct.c_uint64]
    lib.qk_mcm_milstein_price.restype = ct.c_double
    lib.qk_mcm_milstein_price.argtypes = mc_common + [ct.c_int32, ct.c_int32, ct.c_uint64]
    lib.qk_mcm_longstaff_schwartz_price.restype = ct.c_double
    lib.qk_mcm_longstaff_schwartz_price.argtypes = mc_common + [ct.c_int32, ct.c_int32, ct.c_uint64]
    lib.qk_mcm_quasi_monte_carlo_sobol_price.restype = ct.c_double
    lib.qk_mcm_quasi_monte_carlo_sobol_price.argtypes = mc_common + [ct.c_int32]
    lib.qk_mcm_quasi_monte_carlo_halton_price.restype = ct.c_double
    lib.qk_mcm_quasi_monte_carlo_halton_price.argtypes = mc_common + [ct.c_int32]
    lib.qk_mcm_multilevel_monte_carlo_price.restype = ct.c_double
    lib.qk_mcm_multilevel_monte_carlo_price.argtypes = mc_common + [ct.c_int32, ct.c_int32, ct.c_int32, ct.c_uint64]
    lib.qk_mcm_importance_sampling_price.restype = ct.c_double
    lib.qk_mcm_importance_sampling_price.argtypes = mc_common + [ct.c_int32, ct.c_double, ct.c_uint64]
    lib.qk_mcm_control_variates_price.restype = ct.c_double
    lib.qk_mcm_control_variates_price.argtypes = mc_common + [ct.c_int32, ct.c_uint64]
    lib.qk_mcm_antithetic_variates_price.restype = ct.c_double
    lib.qk_mcm_antithetic_variates_price.argtypes = mc_common + [ct.c_int32, ct.c_uint64]
    lib.qk_mcm_stratified_sampling_price.restype = ct.c_double
    lib.qk_mcm_stratified_sampling_price.argtypes = mc_common + [ct.c_int32, ct.c_uint64]


def _verify_abi(lib: ct.CDLL) -> None:
    major = ct.c_int32(-1)
    minor = ct.c_int32(-1)
    lib.qk_abi_version(ct.byref(major), ct.byref(minor))

    if major.value != ABI_MAJOR:
        raise RuntimeError(
            f"ABI major version mismatch: library={major.value}, expected={ABI_MAJOR}"
        )
    if minor.value < ABI_MINOR:
        raise RuntimeError(
            f"ABI minor version too old: library={minor.value}, expected>={ABI_MINOR}"
        )


def load_library() -> ct.CDLL:
    """Load the shared library and verify ABI version."""
    search_dirs = _search_dirs()
    path = _find_library_or_raise(_kernel_names(), search_dirs, "kernel")

    lib = ct.CDLL(str(path))
    _configure_function_signatures(lib)

    _verify_abi(lib)
    return lib
