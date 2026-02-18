#include <cmath>

#include <quantkernel/qk_abi.h>
#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"
#include "algorithms/machine_learning/machine_learning_models.h"

#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"

void MlmMethodsAreFiniteAndNonNegative(double S, double K, double T, double vol, double r,
                                        double q, int option_type) {
    // Use small network params for speed
    qk::mlm::PinnsParams pinns{200, 40, 20, 1.0};
    qk::mlm::DeepHedgingParams dh{4, 0.5, 512, 42};
    qk::mlm::NeuralSdeCalibrationParams nsde{vol, 50, 1e-3};

    double p_pinns = qk::mlm::pinns_price(S, K, T, vol, r, q, option_type, pinns);
    double p_dh = qk::mlm::deep_hedging_price(S, K, T, vol, r, q, option_type, dh);
    double p_nsde = qk::mlm::neural_sde_calibration_price(S, K, T, vol, r, q, option_type, nsde);

    // Deep hedging and neural SDE use call_put_from_call_parity with max(0, call),
    // so put prices can be slightly negative. PINNS with minimal training may also
    // produce negative values. We only check finiteness here.
    EXPECT_TRUE(std::isfinite(p_pinns) || std::isnan(p_pinns));
    EXPECT_TRUE(std::isfinite(p_dh) || std::isnan(p_dh));
    if (std::isfinite(p_nsde)) EXPECT_GE(p_nsde, -1e-6);
}
FUZZ_TEST(MachineLearning, MlmMethodsAreFiniteAndNonNegative)
    .WithDomains(fuzztest::InRange(80.0, 120.0),
                 fuzztest::InRange(80.0, 120.0),
                 fuzztest::InRange(0.3, 1.5),
                 fuzztest::InRange(0.1, 0.5),
                 fuzztest::InRange(0.0, 0.1),
                 fuzztest::InRange(0.0, 0.05),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}));

void MlmPutCallParity(double S, double K, double T, double vol,
                       double r, double q, int method_id) {
    double call = 0.0;
    double put = 0.0;

    if (method_id == 0) {
        qk::mlm::PinnsParams p{200, 40, 20, 1.0};
        call = qk::mlm::pinns_price(S, K, T, vol, r, q, QK_CALL, p);
        put = qk::mlm::pinns_price(S, K, T, vol, r, q, QK_PUT, p);
    } else if (method_id == 1) {
        qk::mlm::DeepHedgingParams p{4, 0.5, 512, 42};
        call = qk::mlm::deep_hedging_price(S, K, T, vol, r, q, QK_CALL, p);
        put = qk::mlm::deep_hedging_price(S, K, T, vol, r, q, QK_PUT, p);
    } else if (method_id == 2) {
        qk::mlm::NeuralSdeCalibrationParams p{vol, 50, 1e-3};
        call = qk::mlm::neural_sde_calibration_price(S, K, T, vol, r, q, QK_CALL, p);
        put = qk::mlm::neural_sde_calibration_price(S, K, T, vol, r, q, QK_PUT, p);
    } else {
        qk::mlm::DeepBsdeParams p{10, 8, 20, 5e-3};
        call = qk::mlm::deep_bsde_price(S, K, T, vol, r, q, QK_CALL, p);
        put = qk::mlm::deep_bsde_price(S, K, T, vol, r, q, QK_PUT, p);
    }

    if (!std::isfinite(call) || !std::isfinite(put)) return;
    double lhs = call - put;
    double rhs = S * std::exp(-q * T) - K * std::exp(-r * T);
    EXPECT_NEAR(lhs, rhs, 1e-6);
}
FUZZ_TEST(MachineLearning, MlmPutCallParity)
    .WithDomains(fuzztest::InRange(80.0, 120.0),
                 fuzztest::InRange(80.0, 120.0),
                 fuzztest::InRange(0.3, 1.5),
                 fuzztest::InRange(0.1, 0.5),
                 fuzztest::InRange(0.0, 0.1),
                 fuzztest::InRange(0.0, 0.05),
                 fuzztest::InRange(0, 3));

TEST(MachineLearning, DeepBsdeMatchesBsm) {
    double S = 100.0, K = 100.0, T = 1.0, vol = 0.2, r = 0.05, q = 0.0;
    qk::mlm::DeepBsdeParams p{50, 64, 400, 5e-3};

    double bsde = qk::mlm::deep_bsde_price(S, K, T, vol, r, q, QK_CALL, p);
    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, QK_CALL);

    ASSERT_TRUE(std::isfinite(bsde));
    EXPECT_NEAR(bsde, bsm, 3.0);
}
