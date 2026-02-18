#include <cmath>

#include <quantkernel/qk_abi.h>
#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"
#include "algorithms/regression_approximation/regression_approximation_models.h"

#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"

void RamMethodsMatchBsm(double S, double K, double T, double vol, double r, double q,
                         int option_type) {
    qk::ram::PolynomialChaosExpansionParams pce{4, 32};
    qk::ram::RadialBasisFunctionParams rbf{24, 1.0, 1e-4};
    qk::ram::SparseGridCollocationParams sgc{3, 9};
    qk::ram::ProperOrthogonalDecompositionParams pod{8, 64};

    double p1 = qk::ram::polynomial_chaos_expansion_price(S, K, T, vol, r, q, option_type, pce);
    double p2 = qk::ram::radial_basis_function_price(S, K, T, vol, r, q, option_type, rbf);
    double p3 = qk::ram::sparse_grid_collocation_price(S, K, T, vol, r, q, option_type, sgc);
    double p4 = qk::ram::proper_orthogonal_decomposition_price(S, K, T, vol, r, q, option_type, pod);

    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);

    // Corrections are proportional to BSM call price. For deep OTM puts the
    // correction on the underlying call can push the put negative via parity,
    // producing absolute errors up to a few dollars.
    double bsm_call = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, QK_CALL);
    double tol = std::max(1.5, 0.25 * std::fabs(bsm_call));
    if (std::isfinite(p1)) EXPECT_NEAR(p1, bsm, tol);
    if (std::isfinite(p2)) EXPECT_NEAR(p2, bsm, tol);
    if (std::isfinite(p3)) EXPECT_NEAR(p3, bsm, tol);
    if (std::isfinite(p4)) EXPECT_NEAR(p4, bsm, tol);
}
FUZZ_TEST(RegressionApproximation, RamMethodsMatchBsm)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(0.2, 2.0),
                 fuzztest::InRange(0.05, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(0.0, 0.08),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}));

void RamPutCallParity(double S, double K, double T, double vol,
                      double r, double q, int method_id) {
    double call = 0.0;
    double put = 0.0;

    if (method_id == 0) {
        qk::ram::PolynomialChaosExpansionParams p{4, 32};
        call = qk::ram::polynomial_chaos_expansion_price(S, K, T, vol, r, q, QK_CALL, p);
        put = qk::ram::polynomial_chaos_expansion_price(S, K, T, vol, r, q, QK_PUT, p);
    } else if (method_id == 1) {
        qk::ram::RadialBasisFunctionParams p{24, 1.0, 1e-4};
        call = qk::ram::radial_basis_function_price(S, K, T, vol, r, q, QK_CALL, p);
        put = qk::ram::radial_basis_function_price(S, K, T, vol, r, q, QK_PUT, p);
    } else if (method_id == 2) {
        qk::ram::SparseGridCollocationParams p{3, 9};
        call = qk::ram::sparse_grid_collocation_price(S, K, T, vol, r, q, QK_CALL, p);
        put = qk::ram::sparse_grid_collocation_price(S, K, T, vol, r, q, QK_PUT, p);
    } else {
        qk::ram::ProperOrthogonalDecompositionParams p{8, 64};
        call = qk::ram::proper_orthogonal_decomposition_price(S, K, T, vol, r, q, QK_CALL, p);
        put = qk::ram::proper_orthogonal_decomposition_price(S, K, T, vol, r, q, QK_PUT, p);
    }

    if (!std::isfinite(call) || !std::isfinite(put)) return;
    double lhs = call - put;
    double rhs = S * std::exp(-q * T) - K * std::exp(-r * T);
    EXPECT_NEAR(lhs, rhs, 1e-6);
}
FUZZ_TEST(RegressionApproximation, RamPutCallParity)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(0.2, 2.0),
                 fuzztest::InRange(0.05, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(0.0, 0.08),
                 fuzztest::InRange(0, 3));
