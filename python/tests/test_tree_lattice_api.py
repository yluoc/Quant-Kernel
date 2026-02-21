"""Tree/lattice API checks."""

from quantkernel import QK_CALL, QK_PUT


def test_tree_methods_return_finite_prices(qk):
    args = dict(spot=100.0, strike=100.0, t=1.0, vol=0.2, r=0.03, q=0.01, option_type=QK_CALL, steps=300)
    prices = [
        qk.crr_price(**args),
        qk.jarrow_rudd_price(**args),
        qk.tian_price(**args),
        qk.leisen_reimer_price(**args),
        qk.trinomial_tree_price(**args),
        qk.derman_kani_const_local_vol_price(
            args["spot"], args["strike"], args["t"], args["vol"], args["r"], args["q"], args["option_type"], 14
        ),
    ]
    assert all(p > 0.0 for p in prices)


def test_american_put_not_below_european_put(qk):
    common = dict(spot=100.0, strike=110.0, t=1.0, vol=0.25, r=0.04, q=0.0, option_type=QK_PUT, steps=300)
    eur = qk.crr_price(**common, american_style=False)
    amr = qk.crr_price(**common, american_style=True)
    assert amr >= eur


def test_crr_respects_user_step_count(qk):
    args = dict(
        spot=100.0,
        strike=100.0,
        t=1.0,
        vol=0.2,
        r=0.03,
        q=0.01,
        option_type=QK_CALL,
        american_style=False,
    )

    step_1 = qk.crr_price(**args, steps=1)
    step_2 = qk.crr_price(**args, steps=2)

    assert abs(step_1 - step_2) > 1e-12


def test_derman_kani_call_surface_price_from_vanilla_surface(qk):
    spot = 100.0
    strike = 100.0
    t = 1.0
    vol = 0.2
    r = 0.03
    q = 0.01

    surface_strikes = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
    surface_maturities = [0.25, 0.5, 1.0, 1.5]

    surface_calls = []
    for tau in surface_maturities:
        row = [
            qk.black_scholes_merton_price(spot, k, tau, vol, r, q, QK_CALL)
            for k in surface_strikes
        ]
        surface_calls.append(row)

    dk_call = qk.derman_kani_call_surface_price(
        spot=spot,
        strike=strike,
        t=t,
        r=r,
        q=q,
        option_type=QK_CALL,
        surface_strikes=surface_strikes,
        surface_maturities=surface_maturities,
        surface_call_prices=surface_calls,
        steps=20,
        american_style=False,
    )
    bsm_call = qk.black_scholes_merton_price(spot, strike, t, vol, r, q, QK_CALL)
    assert abs(dk_call - bsm_call) < 4.0
