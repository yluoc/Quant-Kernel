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
