#include <algorithm>
#include <cmath>

#include <quantkernel/qk_abi.h>
#include "algorithms/closed_form_semi_analytical/black_scholes_merton/black_scholes_merton.h"
#include "algorithms/fourier_transform_methods/fourier_transform_models.h"

#include "fuzztest/fuzztest.h"
#include "gtest/gtest.h"

void CarrMadanFftMatchesBsm(double S, double K, double T, double vol, double r, double q,
                            int option_type) {
    qk::ftm::CarrMadanFFTParams p{};
    p.grid_size = 1024;
    p.eta = 0.25;
    p.alpha = 1.5;

    double fft = qk::ftm::carr_madan_fft_price(S, K, T, vol, r, q, option_type, p);
    if (!std::isfinite(fft)) return;

    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);
    EXPECT_NEAR(fft, bsm, 0.25);
}
FUZZ_TEST(FourierTransform, CarrMadanFftMatchesBsm)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(0.2, 2.0),
                 fuzztest::InRange(0.05, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(0.0, 0.08),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}));

void CosMethodMatchesBsm(double S, double K, double T, double vol, double r, double q,
                         int option_type) {
    qk::ftm::COSMethodParams p{};
    p.n_terms = 256;
    p.truncation_width = 10.0;

    double cos = qk::ftm::cos_method_fang_oosterlee_price(S, K, T, vol, r, q, option_type, p);
    if (!std::isfinite(cos)) return;

    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);
    EXPECT_NEAR(cos, bsm, 2e-4);
}
FUZZ_TEST(FourierTransform, CosMethodMatchesBsm)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(0.2, 2.0),
                 fuzztest::InRange(0.05, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(0.0, 0.08),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}));

void FractionalFftMatchesBsm(double S, double K, double T, double vol, double r, double q,
                             int option_type) {
    qk::ftm::FractionalFFTParams p{};
    p.grid_size = 256;
    p.eta = 0.25;
    p.lambda = 0.05;
    p.alpha = 1.5;

    double frft = qk::ftm::fractional_fft_price(S, K, T, vol, r, q, option_type, p);
    if (!std::isfinite(frft)) return;

    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);
    EXPECT_NEAR(frft, bsm, 0.4);
}
FUZZ_TEST(FourierTransform, FractionalFftMatchesBsm)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(0.2, 2.0),
                 fuzztest::InRange(0.05, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(0.0, 0.08),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}));

void LewisAndHilbertAreFiniteAndNearBsm(double S, double K, double T, double vol, double r, double q,
                                        int option_type) {
    qk::ftm::LewisFourierInversionParams lewis_p{};
    lewis_p.integration_steps = 768;
    lewis_p.integration_limit = 180.0;
    qk::ftm::HilbertTransformParams hilbert_p{};
    hilbert_p.integration_steps = 1024;
    hilbert_p.integration_limit = 200.0;

    double lewis = qk::ftm::lewis_fourier_inversion_price(S, K, T, vol, r, q, option_type, lewis_p);
    double hilbert = qk::ftm::hilbert_transform_price(S, K, T, vol, r, q, option_type, hilbert_p);

    EXPECT_TRUE(std::isfinite(lewis));
    EXPECT_TRUE(std::isfinite(hilbert));
    EXPECT_GE(lewis, -1e-3);
    EXPECT_GE(hilbert, -1e-4);

    double bsm = qk::cfa::black_scholes_merton_price(S, K, T, vol, r, q, option_type);
    EXPECT_NEAR(lewis, bsm, 2e-3);
    EXPECT_NEAR(hilbert, bsm, 2e-4);
}
FUZZ_TEST(FourierTransform, LewisAndHilbertAreFiniteAndNearBsm)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(0.2, 2.0),
                 fuzztest::InRange(0.05, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(0.0, 0.08),
                 fuzztest::ElementOf({QK_CALL, QK_PUT}));

void FourierPutCallParity(double S, double K, double T, double vol, double r, double q,
                          int method_id) {
    double call = 0.0;
    double put = 0.0;

    if (method_id == 0) {
        qk::ftm::CarrMadanFFTParams p{1024, 0.25, 1.5};
        call = qk::ftm::carr_madan_fft_price(S, K, T, vol, r, q, QK_CALL, p);
        put = qk::ftm::carr_madan_fft_price(S, K, T, vol, r, q, QK_PUT, p);
    } else if (method_id == 1) {
        qk::ftm::COSMethodParams p{256, 10.0};
        call = qk::ftm::cos_method_fang_oosterlee_price(S, K, T, vol, r, q, QK_CALL, p);
        put = qk::ftm::cos_method_fang_oosterlee_price(S, K, T, vol, r, q, QK_PUT, p);
    } else if (method_id == 2) {
        qk::ftm::FractionalFFTParams p{256, 0.25, 0.05, 1.5};
        call = qk::ftm::fractional_fft_price(S, K, T, vol, r, q, QK_CALL, p);
        put = qk::ftm::fractional_fft_price(S, K, T, vol, r, q, QK_PUT, p);
    } else if (method_id == 3) {
        qk::ftm::LewisFourierInversionParams p{768, 180.0};
        call = qk::ftm::lewis_fourier_inversion_price(S, K, T, vol, r, q, QK_CALL, p);
        put = qk::ftm::lewis_fourier_inversion_price(S, K, T, vol, r, q, QK_PUT, p);
    } else {
        qk::ftm::HilbertTransformParams p{1024, 200.0};
        call = qk::ftm::hilbert_transform_price(S, K, T, vol, r, q, QK_CALL, p);
        put = qk::ftm::hilbert_transform_price(S, K, T, vol, r, q, QK_PUT, p);
    }

    if (!std::isfinite(call) || !std::isfinite(put)) return;
    double lhs = call - put;
    double rhs = S * std::exp(-q * T) - K * std::exp(-r * T);
    EXPECT_NEAR(lhs, rhs, 3e-3);

    EXPECT_GE(call, -0.05);
    EXPECT_GE(put, -0.05);
}
FUZZ_TEST(FourierTransform, FourierPutCallParity)
    .WithDomains(fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(60.0, 140.0),
                 fuzztest::InRange(0.2, 2.0),
                 fuzztest::InRange(0.05, 0.6),
                 fuzztest::InRange(0.0, 0.12),
                 fuzztest::InRange(0.0, 0.08),
                 fuzztest::InRange(0, 4));
