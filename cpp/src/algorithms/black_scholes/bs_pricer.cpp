#include "bs_pricer.h"
#include "common/math_util.h"
#include <cmath>

namespace qk {

/* Single-row BS price (no validation — caller's responsibility) */
double bs_price_single(double S, double K, double T, double vol,
                       double r, double q, int32_t opt_type) {
    double sqrtT  = std::sqrt(T);
    double d1     = (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * sqrtT);
    double d2     = d1 - vol * sqrtT;
    double df     = std::exp(-r * T);
    double qf     = std::exp(-q * T);

    if (opt_type == QK_CALL) {
        return S * qf * norm_cdf(d1) - K * df * norm_cdf(d2);
    } else {
        return K * df * norm_cdf(-d2) - S * qf * norm_cdf(-d1);
    }
}

/* Batch Black-Scholes pricer with full greeks */
void bs_price_batch(const QKBSInput& in, QKBSOutput& out) {
    int64_t n = in.n;

    for (int64_t i = 0; i < n; ++i) {
        double S   = in.spot[i];
        double K   = in.strike[i];
        double T   = in.time_to_expiry[i];
        double vol = in.volatility[i];
        double r   = in.risk_free_rate[i];
        double q   = in.dividend_yield[i];
        int32_t ot = in.option_type[i];

        /* --- Per-row validation --- */
        if (!is_finite_safe(S) || !is_finite_safe(K) || !is_finite_safe(T) ||
            !is_finite_safe(vol) || !is_finite_safe(r) || !is_finite_safe(q)) {
            out.error_codes[i] = QK_ROW_ERR_NON_FINITE;
            write_nan(&out.price[i]);
            write_nan(&out.delta[i]);
            write_nan(&out.gamma[i]);
            write_nan(&out.vega[i]);
            write_nan(&out.theta[i]);
            write_nan(&out.rho[i]);
            continue;
        }
        if (S <= 0.0) {
            out.error_codes[i] = QK_ROW_ERR_NEGATIVE_S;
            write_nan(&out.price[i]); write_nan(&out.delta[i]);
            write_nan(&out.gamma[i]); write_nan(&out.vega[i]);
            write_nan(&out.theta[i]); write_nan(&out.rho[i]);
            continue;
        }
        if (K <= 0.0) {
            out.error_codes[i] = QK_ROW_ERR_NEGATIVE_K;
            write_nan(&out.price[i]); write_nan(&out.delta[i]);
            write_nan(&out.gamma[i]); write_nan(&out.vega[i]);
            write_nan(&out.theta[i]); write_nan(&out.rho[i]);
            continue;
        }
        if (T <= 0.0) {
            out.error_codes[i] = QK_ROW_ERR_NEGATIVE_T;
            write_nan(&out.price[i]); write_nan(&out.delta[i]);
            write_nan(&out.gamma[i]); write_nan(&out.vega[i]);
            write_nan(&out.theta[i]); write_nan(&out.rho[i]);
            continue;
        }
        if (vol <= 0.0) {
            out.error_codes[i] = QK_ROW_ERR_NEGATIVE_V;
            write_nan(&out.price[i]); write_nan(&out.delta[i]);
            write_nan(&out.gamma[i]); write_nan(&out.vega[i]);
            write_nan(&out.theta[i]); write_nan(&out.rho[i]);
            continue;
        }
        if (ot != QK_CALL && ot != QK_PUT) {
            out.error_codes[i] = QK_ROW_ERR_BAD_TYPE;
            write_nan(&out.price[i]); write_nan(&out.delta[i]);
            write_nan(&out.gamma[i]); write_nan(&out.vega[i]);
            write_nan(&out.theta[i]); write_nan(&out.rho[i]);
            continue;
        }

        /* --- Core Black-Scholes computation --- */
        double sqrtT = std::sqrt(T);
        double d1    = (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * sqrtT);
        double d2    = d1 - vol * sqrtT;
        double df    = std::exp(-r * T);
        double qf    = std::exp(-q * T);

        double Nd1   = norm_cdf(d1);
        double Nd2   = norm_cdf(d2);
        double nd1   = norm_pdf(d1);

        if (ot == QK_CALL) {
            out.price[i] = S * qf * Nd1 - K * df * Nd2;
            out.delta[i] = qf * Nd1;
            out.rho[i]   = K * T * df * Nd2 / 100.0;
            out.theta[i] = (-S * qf * nd1 * vol / (2.0 * sqrtT)
                            - r * K * df * Nd2
                            + q * S * qf * Nd1) / 365.0;
        } else {
            double Nmd1 = norm_cdf(-d1);
            double Nmd2 = norm_cdf(-d2);
            out.price[i] = K * df * Nmd2 - S * qf * Nmd1;
            out.delta[i] = -qf * Nmd1;
            out.rho[i]   = -K * T * df * Nmd2 / 100.0;
            out.theta[i] = (-S * qf * nd1 * vol / (2.0 * sqrtT)
                            + r * K * df * Nmd2
                            - q * S * qf * Nmd1) / 365.0;
        }

        /* Greeks common to both call and put */
        out.gamma[i] = qf * nd1 / (S * vol * sqrtT);
        out.vega[i]  = S * qf * nd1 * sqrtT / 100.0;

        out.error_codes[i] = QK_ROW_OK;
    }
}

} // namespace qk
