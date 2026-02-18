#include "algorithms/integral_quadrature/gauss_hermite/gauss_hermite.h"

#include "algorithms/integral_quadrature/common/internal_util.h"

namespace qk::iqm {

double gauss_hermite_price(double spot, double strike, double t, double vol,
                           double r, double q, int32_t option_type,
                           const GaussHermiteParams& params) {
    if (!detail::valid_option_type(option_type)) return detail::nan_value();
    if (!is_finite_safe(spot) || !is_finite_safe(strike) || !is_finite_safe(t) ||
        !is_finite_safe(vol) || !is_finite_safe(r) || !is_finite_safe(q)) {
        return detail::nan_value();
    }
    if (spot <= 0.0 || strike <= 0.0 || t < 0.0 || vol < 0.0) return detail::nan_value();
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (vol <= detail::kEps) return detail::deterministic_price(spot, strike, t, r, q, option_type);

    int32_t n_points = params.n_points;
    if (n_points < 8 || n_points > 128) return detail::nan_value();

    std::vector<double> nodes;
    std::vector<double> weights;
    if (!detail::gauss_hermite_rule(n_points, nodes, weights)) return detail::nan_value();

    double mu = (r - q - 0.5 * vol * vol) * t;
    double sigma_sqrt_t = vol * std::sqrt(t);
    double expectation = 0.0;

    for (int32_t i = 0; i < n_points; ++i) {
        double z = std::sqrt(2.0) * nodes[static_cast<std::size_t>(i)];
        double s_t = spot * std::exp(mu + sigma_sqrt_t * z);
        expectation += weights[static_cast<std::size_t>(i)]
            * detail::intrinsic_value(s_t, strike, option_type);
    }

    expectation /= std::sqrt(detail::kPi);
    return std::exp(-r * t) * expectation;
}

} // namespace qk::iqm
