"""Library finder and ABI version check."""

import ctypes as ct
import os
import sys
from pathlib import Path

from ._abi import (
    ABI_MAJOR,
    ABI_MINOR,
)

D = ct.c_double
I32 = ct.c_int32
U64 = ct.c_uint64
PD = ct.POINTER(ct.c_double)
PI32 = ct.POINTER(ct.c_int32)
PU64 = ct.POINTER(ct.c_uint64)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _search_dirs() -> list[Path]:
    search_dirs: list[Path] = []

    env_path = os.environ.get("QK_LIB_PATH")
    if env_path:
        search_dirs.append(Path(env_path))

    search_dirs.append(Path(__file__).resolve().parent)

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
        "Install a wheel for your platform, or set QK_LIB_PATH to a built library directory."
    )


_cf_common = [D, D, D, D, D, D, I32]
_cf_batch_base = [PD, PD, PD, PD, PD, PD, PI32]
_ftm_common = [D, D, D, D, D, D, I32]
_ftm_batch_base = [PD, PD, PD, PD, PD, PD, PI32]
_mc_common = [D, D, D, D, D, D, I32]
_mc_batch_base = [PD, PD, PD, PD, PD, PD, PI32]
_tlm_argtypes = [D, D, D, D, D, D, I32, I32, I32]
_tlm_batch_argtypes = [PD, PD, PD, PD, PD, PD, PI32, PI32, PI32, I32, PD]
_ram_common = [D, D, D, D, D, D, I32]
_agm_common = [D, D, D, D, D, D, I32]
_mlm_common = [D, D, D, D, D, D, I32]

_FUNCTION_SIGNATURES = [
    ("qk_abi_version", None, [ct.POINTER(I32), ct.POINTER(I32)]),
    ("qk_get_last_error", ct.c_char_p, []),
    ("qk_clear_last_error", None, []),

    ("qk_cf_black_scholes_merton_price", D, _cf_common),
    ("qk_cf_black76_price", D, [D, D, D, D, D, I32]),
    ("qk_cf_bachelier_price", D, [D, D, D, D, D, I32]),
    ("qk_cf_heston_price_cf", D, [D, D, D, D, D, D, D, D, D, D, I32, I32, D]),
    ("qk_cf_merton_jump_diffusion_price", D, [D, D, D, D, D, D, D, D, D, I32, I32]),
    ("qk_cf_variance_gamma_price_cf", D, [D, D, D, D, D, D, D, D, I32, I32, D]),
    ("qk_cf_sabr_hagan_lognormal_iv", D, [D, D, D, D, D, D, D]),
    ("qk_cf_sabr_hagan_black76_price", D, [D, D, D, D, D, D, D, D, I32]),
    ("qk_cf_dupire_local_vol", D, [D, D, D, D, D, D, D, D]),

    ("qk_cf_black_scholes_merton_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, I32, PD]),
    ("qk_cf_black76_price_batch", I32,
     [PD, PD, PD, PD, PD, PI32, I32, PD]),
    ("qk_cf_bachelier_price_batch", I32,
     [PD, PD, PD, PD, PD, PI32, I32, PD]),
    ("qk_cf_heston_price_cf_batch", I32,
     [PD, PD, PD, PD, PD, PD, PD, PD, PD, PD, PI32, PI32, PD, I32, PD]),
    ("qk_cf_merton_jump_diffusion_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PD, PD, PD, PI32, PI32, I32, PD]),
    ("qk_cf_variance_gamma_price_cf_batch", I32,
     [PD, PD, PD, PD, PD, PD, PD, PD, PI32, PI32, PD, I32, PD]),
    ("qk_cf_sabr_hagan_lognormal_iv_batch", I32,
     [PD, PD, PD, PD, PD, PD, PD, I32, PD]),
    ("qk_cf_sabr_hagan_black76_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PD, PD, PI32, I32, PD]),
    ("qk_cf_dupire_local_vol_batch", I32,
     [PD, PD, PD, PD, PD, PD, PD, PD, I32, PD]),

    ("qk_tlm_crr_price", D, _tlm_argtypes),
    ("qk_tlm_jarrow_rudd_price", D, _tlm_argtypes),
    ("qk_tlm_tian_price", D, _tlm_argtypes),
    ("qk_tlm_leisen_reimer_price", D, _tlm_argtypes),
    ("qk_tlm_trinomial_tree_price", D, _tlm_argtypes),
    ("qk_tlm_derman_kani_const_local_vol_price", D, _tlm_argtypes),
    ("qk_tlm_derman_kani_call_surface_price", D,
     [D, D, D, D, D, I32, PD, I32, PD, I32, PD, I32, I32]),

    ("qk_tlm_crr_price_batch", I32, _tlm_batch_argtypes),
    ("qk_tlm_jarrow_rudd_price_batch", I32, _tlm_batch_argtypes),
    ("qk_tlm_tian_price_batch", I32, _tlm_batch_argtypes),
    ("qk_tlm_leisen_reimer_price_batch", I32, _tlm_batch_argtypes),
    ("qk_tlm_trinomial_tree_price_batch", I32, _tlm_batch_argtypes),
    ("qk_tlm_derman_kani_const_local_vol_price_batch", I32, _tlm_batch_argtypes),

    ("qk_fdm_explicit_fd_price", D, [D, D, D, D, D, D, I32, I32, I32, I32]),
    ("qk_fdm_implicit_fd_price", D, [D, D, D, D, D, D, I32, I32, I32, I32]),
    ("qk_fdm_crank_nicolson_price", D, [D, D, D, D, D, D, I32, I32, I32, I32]),
    ("qk_fdm_adi_douglas_price", D, [D, D, D, D, D, D, D, D, D, D, I32, I32, I32, I32]),
    ("qk_fdm_adi_craig_sneyd_price", D, [D, D, D, D, D, D, D, D, D, D, I32, I32, I32, I32]),
    ("qk_fdm_adi_hundsdorfer_verwer_price", D, [D, D, D, D, D, D, D, D, D, D, I32, I32, I32, I32]),
    ("qk_fdm_psor_price", D, [D, D, D, D, D, D, I32, I32, I32, D, D, I32]),

    ("qk_mcm_standard_monte_carlo_price", D, _mc_common + [I32, U64]),
    ("qk_mcm_euler_maruyama_price", D, _mc_common + [I32, I32, U64]),
    ("qk_mcm_milstein_price", D, _mc_common + [I32, I32, U64]),
    ("qk_mcm_longstaff_schwartz_price", D, _mc_common + [I32, I32, U64]),
    ("qk_mcm_quasi_monte_carlo_sobol_price", D, _mc_common + [I32]),
    ("qk_mcm_quasi_monte_carlo_halton_price", D, _mc_common + [I32]),
    ("qk_mcm_multilevel_monte_carlo_price", D, _mc_common + [I32, I32, I32, U64]),
    ("qk_mcm_importance_sampling_price", D, _mc_common + [I32, D, U64]),
    ("qk_mcm_control_variates_price", D, _mc_common + [I32, U64]),
    ("qk_mcm_antithetic_variates_price", D, _mc_common + [I32, U64]),
    ("qk_mcm_stratified_sampling_price", D, _mc_common + [I32, U64]),

    ("qk_mcm_standard_monte_carlo_price_batch", I32,
     _mc_batch_base + [PI32, PU64, I32, PD]),
    ("qk_mcm_euler_maruyama_price_batch", I32,
     _mc_batch_base + [PI32, PI32, PU64, I32, PD]),
    ("qk_mcm_milstein_price_batch", I32,
     _mc_batch_base + [PI32, PI32, PU64, I32, PD]),
    ("qk_mcm_longstaff_schwartz_price_batch", I32,
     _mc_batch_base + [PI32, PI32, PU64, I32, PD]),
    ("qk_mcm_quasi_monte_carlo_sobol_price_batch", I32,
     _mc_batch_base + [PI32, I32, PD]),
    ("qk_mcm_quasi_monte_carlo_halton_price_batch", I32,
     _mc_batch_base + [PI32, I32, PD]),
    ("qk_mcm_multilevel_monte_carlo_price_batch", I32,
     _mc_batch_base + [PI32, PI32, PI32, PU64, I32, PD]),
    ("qk_mcm_importance_sampling_price_batch", I32,
     _mc_batch_base + [PI32, PD, PU64, I32, PD]),
    ("qk_mcm_control_variates_price_batch", I32,
     _mc_batch_base + [PI32, PU64, I32, PD]),
    ("qk_mcm_antithetic_variates_price_batch", I32,
     _mc_batch_base + [PI32, PU64, I32, PD]),
    ("qk_mcm_stratified_sampling_price_batch", I32,
     _mc_batch_base + [PI32, PU64, I32, PD]),

    ("qk_ftm_carr_madan_fft_price", D, _ftm_common + [I32, D, D]),
    ("qk_ftm_cos_fang_oosterlee_price", D, _ftm_common + [I32, D]),
    ("qk_ftm_fractional_fft_price", D, _ftm_common + [I32, D, D, D]),
    ("qk_ftm_lewis_fourier_inversion_price", D, _ftm_common + [I32, D]),
    ("qk_ftm_hilbert_transform_price", D, _ftm_common + [I32, D]),

    ("qk_ftm_carr_madan_fft_price_batch", I32,
     _ftm_batch_base + [PI32, PD, PD, I32, PD]),
    ("qk_ftm_cos_fang_oosterlee_price_batch", I32,
     _ftm_batch_base + [PI32, PD, I32, PD]),
    ("qk_ftm_fractional_fft_price_batch", I32,
     _ftm_batch_base + [PI32, PD, PD, PD, I32, PD]),
    ("qk_ftm_lewis_fourier_inversion_price_batch", I32,
     _ftm_batch_base + [PI32, PD, I32, PD]),
    ("qk_ftm_hilbert_transform_price_batch", I32,
     _ftm_batch_base + [PI32, PD, I32, PD]),

    ("qk_iqm_gauss_hermite_price", D, _cf_common + [I32]),
    ("qk_iqm_gauss_laguerre_price", D, _cf_common + [I32]),
    ("qk_iqm_gauss_legendre_price", D, _cf_common + [I32, D]),
    ("qk_iqm_adaptive_quadrature_price", D, _cf_common + [D, D, I32, D]),

    ("qk_ram_polynomial_chaos_expansion_price", D, _ram_common + [I32, I32]),
    ("qk_ram_radial_basis_function_price", D, _ram_common + [I32, D, D]),
    ("qk_ram_sparse_grid_collocation_price", D, _ram_common + [I32, I32]),
    ("qk_ram_proper_orthogonal_decomposition_price", D, _ram_common + [I32, I32]),

    ("qk_agm_pathwise_derivative_delta", D, _agm_common + [I32, U64]),
    ("qk_agm_likelihood_ratio_delta", D, _agm_common + [I32, U64, D]),
    ("qk_agm_aad_delta", D, _agm_common + [I32, D]),

    ("qk_mlm_deep_bsde_price", D, _mlm_common + [I32, I32, I32, D]),
    ("qk_mlm_pinns_price", D, _mlm_common + [I32, I32, I32, D]),
    ("qk_mlm_deep_hedging_price", D, _mlm_common + [I32, D, I32, U64]),
    ("qk_mlm_neural_sde_calibration_price", D, _mlm_common + [D, I32, D]),

    ("qk_fdm_explicit_fd_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PI32, PI32, PI32, I32, PD]),
    ("qk_fdm_implicit_fd_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PI32, PI32, PI32, I32, PD]),
    ("qk_fdm_crank_nicolson_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PI32, PI32, PI32, I32, PD]),
    ("qk_fdm_adi_douglas_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PD, PD, PD, PD, PI32, PI32, PI32, PI32, I32, PD]),
    ("qk_fdm_adi_craig_sneyd_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PD, PD, PD, PD, PI32, PI32, PI32, PI32, I32, PD]),
    ("qk_fdm_adi_hundsdorfer_verwer_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PD, PD, PD, PD, PI32, PI32, PI32, PI32, I32, PD]),
    ("qk_fdm_psor_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PI32, PI32, PD, PD, PI32, I32, PD]),

    ("qk_iqm_gauss_hermite_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PI32, I32, PD]),
    ("qk_iqm_gauss_laguerre_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PI32, I32, PD]),
    ("qk_iqm_gauss_legendre_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PI32, PD, I32, PD]),
    ("qk_iqm_adaptive_quadrature_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PD, PD, PI32, PD, I32, PD]),

    ("qk_ram_polynomial_chaos_expansion_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PI32, PI32, I32, PD]),
    ("qk_ram_radial_basis_function_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PI32, PD, PD, I32, PD]),
    ("qk_ram_sparse_grid_collocation_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PI32, PI32, I32, PD]),
    ("qk_ram_proper_orthogonal_decomposition_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PI32, PI32, I32, PD]),

    ("qk_agm_pathwise_derivative_delta_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PI32, PU64, I32, PD]),
    ("qk_agm_likelihood_ratio_delta_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PI32, PU64, PD, I32, PD]),
    ("qk_agm_aad_delta_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PI32, PD, I32, PD]),

    ("qk_mlm_deep_bsde_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PI32, PI32, PI32, PD, I32, PD]),
    ("qk_mlm_pinns_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PI32, PI32, PI32, PD, I32, PD]),
    ("qk_mlm_deep_hedging_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PI32, PD, PI32, PU64, I32, PD]),
    ("qk_mlm_neural_sde_calibration_price_batch", I32,
     [PD, PD, PD, PD, PD, PD, PI32, PD, PI32, PD, I32, PD]),
]


def _configure_function_signatures(lib: ct.CDLL) -> None:
    for name, restype, argtypes in _FUNCTION_SIGNATURES:
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes


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
