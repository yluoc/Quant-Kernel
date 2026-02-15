"""QuantKernel high-level Python API."""

import math

from ._abi import QK_CALL, QK_PUT
from ._loader import load_library


class QuantKernel:
    """Thin wrapper for the QuantKernel shared library."""

    CALL = QK_CALL
    PUT = QK_PUT

    def __init__(self):
        self._lib = load_library()

    def _call_checked(self, fn_name: str, *args: float | int) -> float:
        fn = getattr(self._lib, fn_name)
        out = fn(*args)
        if math.isnan(out):
            raise ValueError(f"{fn_name} returned NaN; check inputs.")
        return out

    def _tree_price(
        self, fn_name: str, spot: float, strike: float, t: float, vol: float,
        r: float, q: float, option_type: int, steps: int, american_style: bool = False
    ) -> float:
        out = self._call_checked(
            fn_name,
            float(spot), float(strike), float(t), float(vol),
            float(r), float(q), int(option_type), int(steps), 1 if american_style else 0
        )
        return out

    def black_scholes_merton_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float, option_type: int
    ) -> float:
        return self._call_checked(
            "qk_cf_black_scholes_merton_price",
            float(spot), float(strike), float(t), float(vol), float(r), float(q), int(option_type)
        )

    def black76_price(
        self, forward: float, strike: float, t: float, vol: float, r: float, option_type: int
    ) -> float:
        return self._call_checked(
            "qk_cf_black76_price",
            float(forward), float(strike), float(t), float(vol), float(r), int(option_type)
        )

    def bachelier_price(
        self, forward: float, strike: float, t: float, normal_vol: float, r: float, option_type: int
    ) -> float:
        return self._call_checked(
            "qk_cf_bachelier_price",
            float(forward), float(strike), float(t), float(normal_vol), float(r), int(option_type)
        )

    def heston_price_cf(
        self, spot: float, strike: float, t: float, r: float, q: float,
        v0: float, kappa: float, theta: float, sigma: float, rho: float,
        option_type: int, integration_steps: int = 1024, integration_limit: float = 120.0
    ) -> float:
        return self._call_checked(
            "qk_cf_heston_price_cf",
            float(spot), float(strike), float(t), float(r), float(q),
            float(v0), float(kappa), float(theta), float(sigma), float(rho),
            int(option_type), int(integration_steps), float(integration_limit)
        )

    def merton_jump_diffusion_price(
        self, spot: float, strike: float, t: float, vol: float, r: float, q: float,
        jump_intensity: float, jump_mean: float, jump_vol: float, max_terms: int, option_type: int
    ) -> float:
        return self._call_checked(
            "qk_cf_merton_jump_diffusion_price",
            float(spot), float(strike), float(t), float(vol), float(r), float(q),
            float(jump_intensity), float(jump_mean), float(jump_vol), int(max_terms), int(option_type)
        )

    def variance_gamma_price_cf(
        self, spot: float, strike: float, t: float, r: float, q: float,
        sigma: float, theta: float, nu: float, option_type: int,
        integration_steps: int = 1024, integration_limit: float = 120.0
    ) -> float:
        return self._call_checked(
            "qk_cf_variance_gamma_price_cf",
            float(spot), float(strike), float(t), float(r), float(q),
            float(sigma), float(theta), float(nu), int(option_type),
            int(integration_steps), float(integration_limit)
        )

    def sabr_hagan_lognormal_iv(
        self, forward: float, strike: float, t: float, alpha: float, beta: float, rho: float, nu: float
    ) -> float:
        return self._call_checked(
            "qk_cf_sabr_hagan_lognormal_iv",
            float(forward), float(strike), float(t),
            float(alpha), float(beta), float(rho), float(nu)
        )

    def sabr_hagan_black76_price(
        self, forward: float, strike: float, t: float, r: float,
        alpha: float, beta: float, rho: float, nu: float, option_type: int
    ) -> float:
        return self._call_checked(
            "qk_cf_sabr_hagan_black76_price",
            float(forward), float(strike), float(t), float(r),
            float(alpha), float(beta), float(rho), float(nu), int(option_type)
        )

    def dupire_local_vol(
        self, strike: float, t: float, call_price: float, dC_dT: float, dC_dK: float, d2C_dK2: float,
        r: float, q: float
    ) -> float:
        return self._call_checked(
            "qk_cf_dupire_local_vol",
            float(strike), float(t), float(call_price), float(dC_dT),
            float(dC_dK), float(d2C_dK2), float(r), float(q)
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
