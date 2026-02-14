#include "algorithms/closed_form_semi_analytical/dupire/dupire.h"

#include "algorithms/closed_form_semi_analytical/common/internal_util.h"

namespace qk::cfa {

double dupire_local_vol(double strike, double t, double call_price,
                        double dC_dT, double dC_dK, double d2C_dK2,
                        double r, double q) {
    if (!is_finite_safe(strike) || !is_finite_safe(t) || !is_finite_safe(call_price) ||
        !is_finite_safe(dC_dT) || !is_finite_safe(dC_dK) || !is_finite_safe(d2C_dK2) ||
        !is_finite_safe(r) || !is_finite_safe(q)) {
        return detail::nan_value();
    }
    if (strike <= 0.0 || t <= 0.0) return detail::nan_value();

    double num = dC_dT + (r - q) * strike * dC_dK + q * call_price;
    double den = 0.5 * strike * strike * d2C_dK2;
    if (den <= 0.0 || num < 0.0) return detail::nan_value();
    return std::sqrt(num / den);
}

double dupire_local_vol_fd(const std::function<double(double, double)>& call_price_fn,
                           double strike, double t, double r, double q,
                           double dK, double dT) {
    if (!call_price_fn || strike <= 0.0 || t <= 0.0 || dK <= 0.0 || dT <= 0.0) {
        return detail::nan_value();
    }

    double Kp = strike + dK;
    double Km = std::max(1e-8, strike - dK);

    double C0 = call_price_fn(strike, t);
    double CpK = call_price_fn(Kp, t);
    double CmK = call_price_fn(Km, t);
    double dC_dK = (CpK - CmK) / (Kp - Km);
    double half = 0.5 * (Kp - Km);
    double d2C_dK2 = (CpK - 2.0 * C0 + CmK) / (half * half);

    double dC_dT = 0.0;
    if (t > dT) {
        double CpT = call_price_fn(strike, t + dT);
        double CmT = call_price_fn(strike, t - dT);
        dC_dT = (CpT - CmT) / (2.0 * dT);
    } else {
        double CpT = call_price_fn(strike, t + dT);
        dC_dT = (CpT - C0) / dT;
    }

    return dupire_local_vol(strike, t, C0, dC_dT, dC_dK, d2C_dK2, r, q);
}

} // namespace qk::cfa
