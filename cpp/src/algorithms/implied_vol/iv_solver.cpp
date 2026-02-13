#include "iv_solver.h"
#include "algorithms/black_scholes/bs_pricer.h"
#include "common/math_util.h"
#include <cmath>

namespace qk {

void iv_solve_batch(const QKIVInput& in, QKIVOutput& out) {
    int64_t n = in.n;

    for (int64_t i = 0; i < n; ++i) {
        double S   = in.spot[i];
        double K   = in.strike[i];
        double T   = in.time_to_expiry[i];
        double r   = in.risk_free_rate[i];
        double q   = in.dividend_yield[i];
        int32_t ot = in.option_type[i];
        double mkt = in.market_price[i];

        /* --- Per-row validation --- */
        if (!is_finite_safe(S) || !is_finite_safe(K) || !is_finite_safe(T) ||
            !is_finite_safe(r) || !is_finite_safe(q) || !is_finite_safe(mkt)) {
            out.error_codes[i] = QK_ROW_ERR_NON_FINITE;
            write_nan(&out.implied_vol[i]);
            out.iterations[i] = 0;
            continue;
        }
        if (S <= 0.0) {
            out.error_codes[i] = QK_ROW_ERR_NEGATIVE_S;
            write_nan(&out.implied_vol[i]); out.iterations[i] = 0;
            continue;
        }
        if (K <= 0.0) {
            out.error_codes[i] = QK_ROW_ERR_NEGATIVE_K;
            write_nan(&out.implied_vol[i]); out.iterations[i] = 0;
            continue;
        }
        if (T <= 0.0) {
            out.error_codes[i] = QK_ROW_ERR_NEGATIVE_T;
            write_nan(&out.implied_vol[i]); out.iterations[i] = 0;
            continue;
        }
        if (ot != QK_CALL && ot != QK_PUT) {
            out.error_codes[i] = QK_ROW_ERR_BAD_TYPE;
            write_nan(&out.implied_vol[i]); out.iterations[i] = 0;
            continue;
        }

        /* Check market price is positive and within theoretical bounds */
        if (mkt <= 0.0) {
            out.error_codes[i] = QK_ROW_ERR_BAD_PRICE;
            write_nan(&out.implied_vol[i]); out.iterations[i] = 0;
            continue;
        }

        double df = std::exp(-r * T);
        double qf = std::exp(-q * T);
        double upper_bound = (ot == QK_CALL) ? S * qf : K * df;
        if (mkt >= upper_bound) {
            out.error_codes[i] = QK_ROW_ERR_BAD_PRICE;
            write_nan(&out.implied_vol[i]); out.iterations[i] = 0;
            continue;
        }

        /* --- Newton-Raphson iteration --- */
        double vol = 0.25;  /* initial guess */
        double tol = in.tol > 0.0 ? in.tol : 1e-8;
        int32_t max_iter = in.max_iter > 0 ? in.max_iter : 100;

        bool converged = false;
        int32_t iter;
        for (iter = 0; iter < max_iter; ++iter) {
            double price = bs_price_single(S, K, T, vol, r, q, ot);
            double diff  = price - mkt;

            if (std::fabs(diff) < tol) {
                converged = true;
                break;
            }

            /* Vega (unscaled, dPrice/dVol) */
            double sqrtT = std::sqrt(T);
            double d1    = (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * sqrtT);
            double vega  = S * qf * norm_pdf(d1) * sqrtT;

            if (vega < 1e-15) {
                break; /* vega too small, Newton step undefined */
            }

            vol -= diff / vega;

            /* Clamp vol to reasonable range */
            if (vol < 1e-6) vol = 1e-6;
            if (vol > 10.0) vol = 10.0;
        }

        if (converged) {
            out.implied_vol[i] = vol;
            out.iterations[i]  = iter + 1;
            out.error_codes[i] = QK_ROW_OK;
        } else {
            out.error_codes[i] = QK_ROW_ERR_IV_NO_CONV;
            write_nan(&out.implied_vol[i]);
            out.iterations[i] = iter;
        }
    }
}

} // namespace qk
