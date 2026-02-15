#include "algorithms/tree_lattice_methods/trinomial_tree/trinomial_tree.h"

#include "algorithms/tree_lattice_methods/common/internal_util.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace qk::tlm {

double trinomial_tree_price(double spot, double strike, double t, double vol, double r, double q,
                            int32_t option_type, int32_t steps, bool american_style) {
    if (!detail::valid_inputs(spot, strike, t, vol, steps, option_type) ||
        !is_finite_safe(r) || !is_finite_safe(q)) {
        return detail::nan_value();
    }
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);
    if (vol <= detail::kEps) {
        double fwd = spot * std::exp((r - q) * t);
        return std::exp(-r * t) * detail::intrinsic_value(fwd, strike, option_type);
    }

    double dt = t / static_cast<double>(steps);
    double disc = std::exp(-r * dt);

    double dx = vol * std::sqrt(2.0 * dt);
    double up = std::exp(dx);
    double inv_up = 1.0 / up;
    double edr = std::exp((r - q) * dt * 0.5);
    double ev = std::exp(vol * std::sqrt(dt * 0.5));
    double inv_ev = 1.0 / ev;
    double denom = ev - inv_ev;
    double pu_raw = (edr - inv_ev) / denom;
    double pd_raw = (ev - edr) / denom;
    double pu = pu_raw * pu_raw;
    double pd = pd_raw * pd_raw;
    double pm = std::max(0.0, 1.0 - pu - pd);
    double sum_p = pu + pm + pd;
    pu /= sum_p;
    pm /= sum_p;
    pd /= sum_p;

    std::size_t max_width = static_cast<std::size_t>(2 * steps + 1);
    std::vector<double> buf_a(max_width);
    std::vector<double> buf_b(max_width);
    double* values = buf_a.data();
    double* next = buf_b.data();

    double node_spot = spot * std::pow(inv_up, static_cast<double>(steps));
    double up2 = up;
    for (int32_t j = -steps; j <= steps; ++j) {
        values[j + steps] = detail::intrinsic_value(node_spot, strike, option_type);
        node_spot *= up2;
    }

    if (!american_style) {
        for (int32_t n = steps - 1; n >= 0; --n) {
            for (int32_t j = -n; j <= n; ++j) {
                int32_t center = j + (n + 1);
                next[j + n] = disc * (pu * values[center + 1] +
                                      pm * values[center] +
                                      pd * values[center - 1]);
            }
            double* tmp = values; values = next; next = tmp;
        }
    } else {
        for (int32_t n = steps - 1; n >= 0; --n) {
            double ns = spot * std::pow(inv_up, static_cast<double>(n));
            for (int32_t j = -n; j <= n; ++j) {
                int32_t center = j + (n + 1);
                double cont = disc * (pu * values[center + 1] +
                                      pm * values[center] +
                                      pd * values[center - 1]);
                next[j + n] = std::max(cont, detail::intrinsic_value(ns, strike, option_type));
                ns *= up2;
            }
            double* tmp = values; values = next; next = tmp;
        }
    }

    return values[0];
}

} // namespace qk::tlm
