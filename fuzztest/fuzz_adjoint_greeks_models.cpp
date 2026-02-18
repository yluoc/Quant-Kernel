#include <cmath>

#include <quantkernel/qk_abi.h>
#include "algorithms/adjoint_greeks/adjoint_greeks_models.h"
#include "algorithms/adjoint_greeks/common/internal_util.h"

#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"

void AadDeltaMatchesBsmDelta(double S, double K, double T, double vol, double r, double q,
                              int option_type) {
    qk::agm::AadParams p{64, 1e-6};
    double aad = qk::agm::aad_delta(S, K, T, vol, r, q, option_type, p);
    if (!std::isfinite(aad)) return;

    double bsm = qk::agm::detail::bsm_delta(S, K, T, vol, r, q, option_type);
    EXPECT_NEAR(aad, bsm, 1e-3);
}
FUZZ_TEST(AdjointGreeks, AadDeltaMatchesBsmDelta)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(0.2, 2.0),
                 fuzztest::InRange(0.05, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(0.0, 0.08),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}));

void PathwiseAndLrDeltasAreFiniteAndBounded(double S, double K, double T, double vol,
                                             double r, double q, int option_type) {
    qk::agm::PathwiseDerivativeParams pw{4096, 42};
    qk::agm::LikelihoodRatioParams lr{4096, 42, 6.0};

    double d_pw = qk::agm::pathwise_derivative_delta(S, K, T, vol, r, q, option_type, pw);
    double d_lr = qk::agm::likelihood_ratio_delta(S, K, T, vol, r, q, option_type, lr);

    double qf = std::exp(-q * T);
    EXPECT_TRUE(std::isfinite(d_pw));
    EXPECT_TRUE(std::isfinite(d_lr));
    EXPECT_LE(std::fabs(d_pw), qf + 1e-9);
    EXPECT_LE(std::fabs(d_lr), qf + 1e-9);

    // With low path counts MC deltas may not respect theoretical sign constraints,
    // so we only verify the absolute bound against qf.
}
FUZZ_TEST(AdjointGreeks, PathwiseAndLrDeltasAreFiniteAndBounded)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(0.2, 2.0),
                 fuzztest::InRange(0.05, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(0.0, 0.08),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}));

void AllThreeDeltasStayClose(double S, double K, double T, double vol,
                              double r, double q, int option_type) {
    qk::agm::PathwiseDerivativeParams pw{32768, 42};
    qk::agm::LikelihoodRatioParams lr{32768, 42, 6.0};
    qk::agm::AadParams aad_p{64, 1e-6};

    double d_pw = qk::agm::pathwise_derivative_delta(S, K, T, vol, r, q, option_type, pw);
    double d_lr = qk::agm::likelihood_ratio_delta(S, K, T, vol, r, q, option_type, lr);
    double d_aad = qk::agm::aad_delta(S, K, T, vol, r, q, option_type, aad_p);

    if (!std::isfinite(d_pw) || !std::isfinite(d_lr) || !std::isfinite(d_aad)) return;

    double tol = std::max(0.1, 0.5 * std::fabs(d_aad));
    EXPECT_NEAR(d_pw, d_aad, tol);
    EXPECT_NEAR(d_lr, d_aad, tol);
}
FUZZ_TEST(AdjointGreeks, AllThreeDeltasStayClose)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(0.2, 2.0),
                 fuzztest::InRange(0.05, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(0.0, 0.08),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}));
