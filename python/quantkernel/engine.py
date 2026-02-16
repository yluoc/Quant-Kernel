"""QuantKernel high-level Python API."""

import math

from ._abi import QK_CALL, QK_PUT
from ._loader import load_library


class QuantKernel:
    """Thin wrapper for the QuantKernel shared library."""

    __slots__ = ('_lib', '_fn_cache')

    CALL = QK_CALL
    PUT = QK_PUT

    def __init__(self):
        self._lib = load_library()
        self._fn_cache = {}

    def _get_fn(self, fn_name: str):
        fn = self._fn_cache.get(fn_name)
        if fn is None:
            fn = getattr(self._lib, fn_name)
            self._fn_cache[fn_name] = fn
        return fn

    def _call_checked(self, fn_name: str, *args) -> float:
        out = self._get_fn(fn_name)(*args)
        if math.isnan(out):
            raise ValueError(f"{fn_name} returned NaN; check inputs.")
        return out

    def _tree_price(
        self, fn_name: str, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, steps: int, american_style: bool = False
    ) -> float:
        return self._get_fn(fn_name)(
            spot, strike, t, vol, r, q, option_type, steps, 1 if american_style else 0
        )

    def black_scholes_merton_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float, option_type: int
    ) -> float:
        return self._get_fn("qk_cf_black_scholes_merton_price")(
            spot, strike, t, vol, r, q, option_type
        )

    def black76_price(
        self, forward: float, strike: float, t: float, vol: float, r: float, option_type: int
    ) -> float:
        return self._get_fn("qk_cf_black76_price")(forward, strike, t, vol, r, option_type)

    def bachelier_price(
        self, forward: float, strike: float, t: float, normal_vol: float, r: float, option_type: int
    ) -> float:
        return self._get_fn("qk_cf_bachelier_price")(forward, strike, t, normal_vol, r, option_type)

    def heston_price_cf(
        self, spot: float, strike: float, t: float, r: float, q: float,
        v0: float, kappa: float, theta: float, sigma: float, rho: float,
        option_type: int, integration_steps: int = 1024, integration_limit: float = 120.0
    ) -> float:
        return self._call_checked(
            "qk_cf_heston_price_cf",
            spot, strike, t, r, q,
            v0, kappa, theta, sigma, rho,
            option_type, integration_steps, integration_limit
        )

    def merton_jump_diffusion_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        jump_intensity: float, jump_mean: float, jump_vol: float, max_terms: int, option_type: int
    ) -> float:
        return self._call_checked(
            "qk_cf_merton_jump_diffusion_price",
            spot, strike, t, vol, r, q,
            jump_intensity, jump_mean, jump_vol, max_terms, option_type
        )

    def variance_gamma_price_cf(
        self, spot: float, strike: float, t: float, r: float, q: float,
        sigma: float, theta: float, nu: float, option_type: int,
        integration_steps: int = 1024, integration_limit: float = 120.0
    ) -> float:
        return self._call_checked(
            "qk_cf_variance_gamma_price_cf",
            spot, strike, t, r, q,
            sigma, theta, nu, option_type,
            integration_steps, integration_limit
        )

    def sabr_hagan_lognormal_iv(
        self, forward: float, strike: float, t: float, alpha: float, beta: float, rho: float, nu: float
    ) -> float:
        return self._call_checked(
            "qk_cf_sabr_hagan_lognormal_iv",
            forward, strike, t, alpha, beta, rho, nu
        )

    def sabr_hagan_black76_price(
        self, forward: float, strike: float, t: float, r: float,
        alpha: float, beta: float, rho: float, nu: float, option_type: int
    ) -> float:
        return self._call_checked(
            "qk_cf_sabr_hagan_black76_price",
            forward, strike, t, r, alpha, beta, rho, nu, option_type
        )

    def dupire_local_vol(
        self, strike: float, t: float, call_price: float, dC_dT: float, dC_dK: float, d2C_dK2: float,
        r: float, q: float
    ) -> float:
        return self._call_checked(
            "qk_cf_dupire_local_vol",
            strike, t, call_price, dC_dT, dC_dK, d2C_dK2, r, q
        )

    def crr_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, steps: int, american_style: bool = False
    ) -> float:
        return self._tree_price(
            "qk_tlm_crr_price", spot, strike, t, vol, r, q, option_type, steps, american_style
        )

    def jarrow_rudd_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, steps: int, american_style: bool = False
    ) -> float:
        return self._tree_price(
            "qk_tlm_jarrow_rudd_price", spot, strike, t, vol, r, q, option_type, steps, american_style
        )

    def tian_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, steps: int, american_style: bool = False
    ) -> float:
        return self._tree_price(
            "qk_tlm_tian_price", spot, strike, t, vol, r, q, option_type, steps, american_style
        )

    def leisen_reimer_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, steps: int, american_style: bool = False
    ) -> float:
        return self._tree_price(
            "qk_tlm_leisen_reimer_price",
            spot, strike, t, vol, r, q, option_type, steps, american_style
        )

    def trinomial_tree_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, steps: int, american_style: bool = False
    ) -> float:
        return self._tree_price(
            "qk_tlm_trinomial_tree_price",
            spot, strike, t, vol, r, q, option_type, steps, american_style
        )

    def derman_kani_const_local_vol_price(
        self, spot: float, strike: float, t: float, local_vol: float, r: float, q: float,
        option_type: int, steps: int, american_style: bool = False
    ) -> float:
        return self._tree_price(
            "qk_tlm_derman_kani_const_local_vol_price",
            spot, strike, t, local_vol, r, q, option_type, steps, american_style
        )

    # --- Finite Difference methods ---

    def _fdm_price(
        self, fn_name: str, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, time_steps: int, spot_steps: int,
        american_style: bool = False
    ) -> float:
        return self._get_fn(fn_name)(
            spot, strike, t, vol, r, q, option_type,
            time_steps, spot_steps, 1 if american_style else 0
        )

    def explicit_fd_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, time_steps: int, spot_steps: int, american_style: bool = False
    ) -> float:
        return self._fdm_price(
            "qk_fdm_explicit_fd_price", spot, strike, t, vol, r, q,
            option_type, time_steps, spot_steps, american_style
        )

    def implicit_fd_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, time_steps: int, spot_steps: int, american_style: bool = False
    ) -> float:
        return self._fdm_price(
            "qk_fdm_implicit_fd_price", spot, strike, t, vol, r, q,
            option_type, time_steps, spot_steps, american_style
        )

    def crank_nicolson_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, time_steps: int, spot_steps: int, american_style: bool = False
    ) -> float:
        return self._fdm_price(
            "qk_fdm_crank_nicolson_price", spot, strike, t, vol, r, q,
            option_type, time_steps, spot_steps, american_style
        )

    def adi_douglas_price(
        self, spot: float, strike: float, t: float, r: float, q: float,
        v0: float, kappa: float, theta_v: float, sigma: float, rho: float,
        option_type: int, s_steps: int = 40, v_steps: int = 20, time_steps: int = 40
    ) -> float:
        return self._call_checked(
            "qk_fdm_adi_douglas_price",
            spot, strike, t, r, q,
            v0, kappa, theta_v, sigma, rho,
            option_type, s_steps, v_steps, time_steps
        )

    def adi_craig_sneyd_price(
        self, spot: float, strike: float, t: float, r: float, q: float,
        v0: float, kappa: float, theta_v: float, sigma: float, rho: float,
        option_type: int, s_steps: int = 40, v_steps: int = 20, time_steps: int = 40
    ) -> float:
        return self._call_checked(
            "qk_fdm_adi_craig_sneyd_price",
            spot, strike, t, r, q,
            v0, kappa, theta_v, sigma, rho,
            option_type, s_steps, v_steps, time_steps
        )

    def adi_hundsdorfer_verwer_price(
        self, spot: float, strike: float, t: float, r: float, q: float,
        v0: float, kappa: float, theta_v: float, sigma: float, rho: float,
        option_type: int, s_steps: int = 40, v_steps: int = 20, time_steps: int = 40
    ) -> float:
        return self._call_checked(
            "qk_fdm_adi_hundsdorfer_verwer_price",
            spot, strike, t, r, q,
            v0, kappa, theta_v, sigma, rho,
            option_type, s_steps, v_steps, time_steps
        )

    def psor_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        option_type: int, time_steps: int, spot_steps: int,
        omega: float = 1.2, tol: float = 1e-8, max_iter: int = 10000
    ) -> float:
        return self._get_fn("qk_fdm_psor_price")(
            spot, strike, t, vol, r, q, option_type,
            time_steps, spot_steps, omega, tol, max_iter
        )
