#include "algorithms/tree_lattice_methods/implied_tree/derman_kani.h"

#include "algorithms/tree_lattice_methods/common/internal_util.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace qk::tlm {

namespace {

inline std::size_t surface_idx(int32_t maturity_idx, int32_t strike_idx, int32_t n_strikes) {
    return static_cast<std::size_t>(maturity_idx) * static_cast<std::size_t>(n_strikes) +
        static_cast<std::size_t>(strike_idx);
}

double local_vol_from_surface_derivatives(
    double strike, double call_price, double dC_dT, double dC_dK, double d2C_dK2, double r, double q
) {
    const double numerator = dC_dT + (r - q) * strike * dC_dK + q * call_price;
    const double denominator = 0.5 * strike * strike * d2C_dK2;
    if (!(denominator > 1e-14) || !(numerator > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double lv2 = numerator / denominator;
    if (!(lv2 > 0.0) || !is_finite_safe(lv2)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::sqrt(lv2);
}

double linear_interp_1d(
    const std::vector<double>& x,
    const std::vector<double>& y,
    int32_t x_offset,
    int32_t stride,
    double xq
) {
    const int32_t n = static_cast<int32_t>(x.size());
    if (n <= 0) return std::numeric_limits<double>::quiet_NaN();

    if (xq <= x.front()) {
        return y[static_cast<std::size_t>(x_offset)];
    }
    if (xq >= x.back()) {
        return y[static_cast<std::size_t>(x_offset + (n - 1) * stride)];
    }

    int32_t hi = static_cast<int32_t>(
        std::lower_bound(x.begin(), x.end(), xq) - x.begin()
    );
    int32_t lo = hi - 1;
    const double x0 = x[static_cast<std::size_t>(lo)];
    const double x1 = x[static_cast<std::size_t>(hi)];
    const double y0 = y[static_cast<std::size_t>(x_offset + lo * stride)];
    const double y1 = y[static_cast<std::size_t>(x_offset + hi * stride)];
    const double w = (xq - x0) / (x1 - x0);
    return (1.0 - w) * y0 + w * y1;
}

} // namespace

double derman_kani_implied_tree_price(
    double spot, double strike, double t, double r, double q,
    int32_t option_type,
    const std::function<double(double, double)>& local_vol_surface,
    ImpliedTreeConfig config
) {
    if (!detail::valid_option_type(option_type) || !is_finite_safe(spot) || !is_finite_safe(strike) ||
        !is_finite_safe(t) || !is_finite_safe(r) || !is_finite_safe(q) ||
        spot <= 0.0 || strike <= 0.0 || t < 0.0 || config.steps <= 0) {
        return detail::nan_value();
    }
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    int32_t N = config.steps;
    double dt = t / static_cast<double>(N);
    double disc = std::exp(-r * dt);

    double sigma_ref = local_vol_surface(spot, 0.0);
    if (!is_finite_safe(sigma_ref) || sigma_ref < 0.0) return detail::nan_value();
    sigma_ref = std::max(sigma_ref, detail::kEps);

    double dx = sigma_ref * std::sqrt(3.0 * dt);
    double up = std::exp(dx);
    double inv_up = 1.0 / up;
    double dx2 = dx * dx;
    double inv_dx = 1.0 / dx;
    double inv_dx2 = 1.0 / dx2;
    std::size_t max_width = static_cast<std::size_t>(2 * N + 1);
    std::vector<double> buf_a(max_width);
    std::vector<double> buf_b(max_width);
    double* values = buf_a.data();
    double* next = buf_b.data();

    double node_spot = spot * std::pow(inv_up, static_cast<double>(N));
    for (int32_t j = -N; j <= N; ++j) {
        values[j + N] = detail::intrinsic_value(node_spot, strike, option_type);
        node_spot *= up;
    }

    for (int32_t n = N - 1; n >= 0; --n) {
        double node_t = static_cast<double>(n) * dt;
        double ns = spot * std::pow(inv_up, static_cast<double>(n));
        for (int32_t j = -n; j <= n; ++j) {
            double sigma = local_vol_surface(ns, node_t);
            if (!is_finite_safe(sigma) || sigma < 0.0) return detail::nan_value();
            sigma = std::max(sigma, detail::kEps);

            double sigma2 = sigma * sigma;
            double mu = r - q - 0.5 * sigma2;
            double var_dt = sigma2 * dt;
            double mu_dt = mu * dt;

            double drift_var = (var_dt + mu_dt * mu_dt) * inv_dx2;
            double drift_dir = mu_dt * inv_dx;
            double pu = 0.5 * (drift_var + drift_dir);
            double pd = 0.5 * (drift_var - drift_dir);
            double pm = 1.0 - pu - pd;

            pu = std::max(0.0, pu);
            pd = std::max(0.0, pd);
            pm = std::max(0.0, pm);
            double sum_p = pu + pm + pd;
            pu /= sum_p;
            pm /= sum_p;
            pd /= sum_p;

            int32_t center = j + (n + 1);
            double cont = disc * (pu * values[center + 1] +
                                  pm * values[center] +
                                  pd * values[center - 1]);

            if (config.american_style) {
                next[j + n] = std::max(cont, detail::intrinsic_value(ns, strike, option_type));
            } else {
                next[j + n] = cont;
            }
            ns *= up;
        }
        double* tmp = values; values = next; next = tmp;
    }

    return values[0];
}

double derman_kani_implied_tree_price_from_call_surface(
    double spot, double strike, double t, double r, double q,
    int32_t option_type,
    const std::vector<double>& surface_strikes,
    const std::vector<double>& surface_maturities,
    const std::vector<double>& surface_call_prices,
    ImpliedTreeConfig config
) {
    if (!detail::valid_option_type(option_type) || !is_finite_safe(spot) || !is_finite_safe(strike) ||
        !is_finite_safe(t) || !is_finite_safe(r) || !is_finite_safe(q) ||
        spot <= 0.0 || strike <= 0.0 || t < 0.0 || config.steps <= 0) {
        return detail::nan_value();
    }
    if (t <= detail::kEps) return detail::intrinsic_value(spot, strike, option_type);

    const int32_t n_strikes = static_cast<int32_t>(surface_strikes.size());
    const int32_t n_maturities = static_cast<int32_t>(surface_maturities.size());
    if (n_strikes < 3 || n_maturities < 2) return detail::nan_value();
    if (static_cast<std::size_t>(n_strikes) * static_cast<std::size_t>(n_maturities) != surface_call_prices.size()) {
        return detail::nan_value();
    }

    for (int32_t k = 0; k < n_strikes; ++k) {
        const double K = surface_strikes[static_cast<std::size_t>(k)];
        if (!is_finite_safe(K) || K <= 0.0) return detail::nan_value();
        if (k > 0 && !(K > surface_strikes[static_cast<std::size_t>(k - 1)])) return detail::nan_value();
    }
    for (int32_t m = 0; m < n_maturities; ++m) {
        const double T = surface_maturities[static_cast<std::size_t>(m)];
        if (!is_finite_safe(T) || T <= 0.0) return detail::nan_value();
        if (m > 0 && !(T > surface_maturities[static_cast<std::size_t>(m - 1)])) return detail::nan_value();
    }
    for (double c : surface_call_prices) {
        if (!is_finite_safe(c) || c < 0.0) return detail::nan_value();
    }

    std::vector<double> local_vols(
        static_cast<std::size_t>(n_maturities) * static_cast<std::size_t>(n_strikes),
        std::numeric_limits<double>::quiet_NaN()
    );

    for (int32_t m = 0; m < n_maturities; ++m) {
        const double Tm = surface_maturities[static_cast<std::size_t>(m)];
        for (int32_t k = 0; k < n_strikes; ++k) {
            const double Kk = surface_strikes[static_cast<std::size_t>(k)];
            const double C = surface_call_prices[surface_idx(m, k, n_strikes)];

            double dC_dT = 0.0;
            if (m == 0) {
                const double T1 = surface_maturities[1];
                const double C1 = surface_call_prices[surface_idx(1, k, n_strikes)];
                dC_dT = (C1 - C) / (T1 - Tm);
            } else if (m == n_maturities - 1) {
                const double T0 = surface_maturities[static_cast<std::size_t>(m - 1)];
                const double C0 = surface_call_prices[surface_idx(m - 1, k, n_strikes)];
                dC_dT = (C - C0) / (Tm - T0);
            } else {
                const double T0 = surface_maturities[static_cast<std::size_t>(m - 1)];
                const double T1 = surface_maturities[static_cast<std::size_t>(m + 1)];
                const double C0 = surface_call_prices[surface_idx(m - 1, k, n_strikes)];
                const double C1 = surface_call_prices[surface_idx(m + 1, k, n_strikes)];
                dC_dT = (C1 - C0) / (T1 - T0);
            }

            double dC_dK = 0.0;
            if (k == 0) {
                const double K1 = surface_strikes[1];
                const double C1 = surface_call_prices[surface_idx(m, 1, n_strikes)];
                dC_dK = (C1 - C) / (K1 - Kk);
            } else if (k == n_strikes - 1) {
                const double K0 = surface_strikes[static_cast<std::size_t>(k - 1)];
                const double C0 = surface_call_prices[surface_idx(m, k - 1, n_strikes)];
                dC_dK = (C - C0) / (Kk - K0);
            } else {
                const double K0 = surface_strikes[static_cast<std::size_t>(k - 1)];
                const double K1 = surface_strikes[static_cast<std::size_t>(k + 1)];
                const double C0 = surface_call_prices[surface_idx(m, k - 1, n_strikes)];
                const double C1 = surface_call_prices[surface_idx(m, k + 1, n_strikes)];
                dC_dK = (C1 - C0) / (K1 - K0);
            }

            double d2C_dK2 = std::numeric_limits<double>::quiet_NaN();
            if (k > 0 && k < n_strikes - 1) {
                const double Km = surface_strikes[static_cast<std::size_t>(k - 1)];
                const double Kp = surface_strikes[static_cast<std::size_t>(k + 1)];
                const double hm = Kk - Km;
                const double hp = Kp - Kk;
                const double Cm = surface_call_prices[surface_idx(m, k - 1, n_strikes)];
                const double Cp = surface_call_prices[surface_idx(m, k + 1, n_strikes)];
                d2C_dK2 = 2.0 * ((Cp - C) / hp - (C - Cm) / hm) / (hp + hm);
            } else if (n_strikes >= 3) {
                const int32_t kk = (k == 0) ? 1 : (n_strikes - 2);
                const double Km = surface_strikes[static_cast<std::size_t>(kk - 1)];
                const double K0 = surface_strikes[static_cast<std::size_t>(kk)];
                const double Kp = surface_strikes[static_cast<std::size_t>(kk + 1)];
                const double hm = K0 - Km;
                const double hp = Kp - K0;
                const double Cm = surface_call_prices[surface_idx(m, kk - 1, n_strikes)];
                const double C0 = surface_call_prices[surface_idx(m, kk, n_strikes)];
                const double Cp = surface_call_prices[surface_idx(m, kk + 1, n_strikes)];
                d2C_dK2 = 2.0 * ((Cp - C0) / hp - (C0 - Cm) / hm) / (hp + hm);
            }

            double lv = local_vol_from_surface_derivatives(Kk, C, dC_dT, dC_dK, d2C_dK2, r, q);
            if (is_finite_safe(lv)) {
                lv = std::max(1e-4, std::min(5.0, lv));
                local_vols[surface_idx(m, k, n_strikes)] = lv;
            }
        }
    }

    double fallback = std::numeric_limits<double>::quiet_NaN();
    for (int32_t m = 0; m < n_maturities && !is_finite_safe(fallback); ++m) {
        int32_t nearest_k = 0;
        double best = std::fabs(surface_strikes[0] - spot);
        for (int32_t k = 1; k < n_strikes; ++k) {
            const double d = std::fabs(surface_strikes[static_cast<std::size_t>(k)] - spot);
            if (d < best) {
                best = d;
                nearest_k = k;
            }
        }
        const double lv = local_vols[surface_idx(m, nearest_k, n_strikes)];
        if (is_finite_safe(lv)) fallback = lv;
    }
    if (!is_finite_safe(fallback)) fallback = 0.2;

    for (int32_t m = 0; m < n_maturities; ++m) {
        for (int32_t k = 0; k < n_strikes; ++k) {
            const std::size_t id = surface_idx(m, k, n_strikes);
            if (is_finite_safe(local_vols[id])) continue;

            int32_t best_dm = 0;
            int32_t best_dk = 0;
            int32_t best_dist2 = std::numeric_limits<int32_t>::max();
            for (int32_t mm = 0; mm < n_maturities; ++mm) {
                for (int32_t kk = 0; kk < n_strikes; ++kk) {
                    const double lv = local_vols[surface_idx(mm, kk, n_strikes)];
                    if (!is_finite_safe(lv)) continue;
                    const int32_t dm = mm - m;
                    const int32_t dk = kk - k;
                    const int32_t dist2 = dm * dm + dk * dk;
                    if (dist2 < best_dist2) {
                        best_dist2 = dist2;
                        best_dm = dm;
                        best_dk = dk;
                    }
                }
            }
            if (best_dist2 == std::numeric_limits<int32_t>::max()) {
                local_vols[id] = fallback;
            } else {
                local_vols[id] = local_vols[surface_idx(m + best_dm, k + best_dk, n_strikes)];
            }
        }
    }

    auto local_vol_surface = [&](double s, double tau) -> double {
        if (!is_finite_safe(s) || s <= 0.0 || !is_finite_safe(tau)) return detail::nan_value();

        const double sq = std::max(surface_strikes.front(), std::min(surface_strikes.back(), s));
        const double tq = std::max(surface_maturities.front(), std::min(surface_maturities.back(), tau));

        int32_t m_hi = static_cast<int32_t>(
            std::lower_bound(surface_maturities.begin(), surface_maturities.end(), tq) - surface_maturities.begin()
        );
        if (m_hi <= 0) m_hi = 1;
        if (m_hi >= n_maturities) m_hi = n_maturities - 1;
        const int32_t m_lo = m_hi - 1;

        const double t0 = surface_maturities[static_cast<std::size_t>(m_lo)];
        const double t1 = surface_maturities[static_cast<std::size_t>(m_hi)];
        const double w_t = (tq - t0) / (t1 - t0);

        const double lv0 = linear_interp_1d(
            surface_strikes, local_vols, m_lo * n_strikes, 1, sq
        );
        const double lv1 = linear_interp_1d(
            surface_strikes, local_vols, m_hi * n_strikes, 1, sq
        );
        const double lv = (1.0 - w_t) * lv0 + w_t * lv1;
        return std::max(1e-4, std::min(5.0, lv));
    };

    return derman_kani_implied_tree_price(
        spot, strike, t, r, q, option_type, local_vol_surface, config
    );
}

} // namespace qk::tlm
