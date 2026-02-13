#include "mc_pricer.h"
#include "common/math_util.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace qk {

namespace {

inline uint64_t splitmix64_next(uint64_t& state) {
    state += UINT64_C(0x9E3779B97F4A7C15);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

inline double uniform_open01(uint64_t& state) {
    static constexpr double INV_2POW53 = 1.0 / 9007199254740992.0;
    uint64_t bits = splitmix64_next(state);
    return ((bits >> 11) + 0.5) * INV_2POW53;
}

inline double standard_normal(uint64_t& state) {
    static constexpr double TWO_PI = 6.2831853071795864769;
    double u1 = uniform_open01(state);
    double u2 = uniform_open01(state);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(TWO_PI * u2);
}

inline void write_error_row(QKMCOutput& out, int64_t i, int32_t error_code) {
    out.error_codes[i] = error_code;
    out.paths_used[i] = 0;
    write_nan(&out.price[i]);
    write_nan(&out.std_error[i]);
}

} // namespace

void mc_price_batch(const QKMCInput& in, QKMCOutput& out) {
    int64_t n = in.n;
    for (int64_t i = 0; i < n; ++i) {
        double S = in.spot[i];
        double K = in.strike[i];
        double T = in.time_to_expiry[i];
        double vol = in.volatility[i];
        double r = in.risk_free_rate[i];
        double q = in.dividend_yield[i];
        int32_t ot = in.option_type[i];
        int32_t paths = in.num_paths[i];
        uint64_t seed = in.rng_seed[i] ^ (UINT64_C(0x9E3779B97F4A7C15) * static_cast<uint64_t>(i + 1));

        if (!is_finite_safe(S) || !is_finite_safe(K) || !is_finite_safe(T) ||
            !is_finite_safe(vol) || !is_finite_safe(r) || !is_finite_safe(q)) {
            write_error_row(out, i, QK_ROW_ERR_NON_FINITE);
            continue;
        }
        if (S <= 0.0) {
            write_error_row(out, i, QK_ROW_ERR_NEGATIVE_S);
            continue;
        }
        if (K <= 0.0) {
            write_error_row(out, i, QK_ROW_ERR_NEGATIVE_K);
            continue;
        }
        if (T <= 0.0) {
            write_error_row(out, i, QK_ROW_ERR_NEGATIVE_T);
            continue;
        }
        if (vol <= 0.0) {
            write_error_row(out, i, QK_ROW_ERR_NEGATIVE_V);
            continue;
        }
        if (ot != QK_CALL && ot != QK_PUT) {
            write_error_row(out, i, QK_ROW_ERR_BAD_TYPE);
            continue;
        }
        if (paths <= 0) {
            write_error_row(out, i, QK_ROW_ERR_BAD_PATHS);
            continue;
        }

        double sqrt_t = std::sqrt(T);
        double drift = (r - q - 0.5 * vol * vol) * T;
        double diffusion = vol * sqrt_t;
        double discount = std::exp(-r * T);

        double mean = 0.0;
        double m2 = 0.0;
        for (int32_t p = 0; p < paths; ++p) {
            double z = standard_normal(seed);
            double st = S * std::exp(drift + diffusion * z);
            double payoff = (ot == QK_CALL)
                                ? std::max(st - K, 0.0)
                                : std::max(K - st, 0.0);
            double discounted_payoff = discount * payoff;

            double delta = discounted_payoff - mean;
            mean += delta / static_cast<double>(p + 1);
            m2 += delta * (discounted_payoff - mean);
        }

        double std_err = 0.0;
        if (paths > 1) {
            double sample_var = m2 / static_cast<double>(paths - 1);
            std_err = std::sqrt(std::max(sample_var, 0.0) / static_cast<double>(paths));
        }

        out.price[i] = mean;
        out.std_error[i] = std_err;
        out.paths_used[i] = paths;
        out.error_codes[i] = QK_ROW_OK;
    }
}

} // namespace qk
