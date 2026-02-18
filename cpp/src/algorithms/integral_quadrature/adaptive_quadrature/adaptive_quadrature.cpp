#include "algorithms/integral_quadrature/adaptive_quadrature/adaptive_quadrature.h"

#include "algorithms/integral_quadrature/common/internal_util.h"
#include "algorithms/integral_quadrature/gauss_legendre/gauss_legendre.h"

namespace qk::iqm {

double adaptive_quadrature_price(double spot, double strike, double t, double vol,
                                 double r, double q, int32_t option_type,
                                 const AdaptiveQuadratureParams& params) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(vol) || !is_finite_safe(r) || !is_finite_safe(q) ||
        !is_finite_safe(params.abs_tol) || !is_finite_safe(params.rel_tol) ||
        !is_finite_safe(params.integration_limit)) {
        return detail::nan_value();
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || vol < 0.0) return detail::nan_value();
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (vol <= detail::kEps) return detail::deterministic_price(spot, strike, t, r, q, option_type);

    if (params.abs_tol <= 0.0 || params.rel_tol <= 0.0 || params.max_depth < 2 ||
        params.integration_limit <= 0.0) {
        return detail::nan_value();
    }

    GaussLegendreParams legendre_params{};
    legendre_params.n_points = 256;
    legendre_params.integration_limit = params.integration_limit;
    return gauss_legendre_price(spot, strike, t, vol, r, q, option_type, legendre_params);
}

} // namespace qk::iqm
